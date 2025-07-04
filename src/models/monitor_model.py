"""
Dataset-Agnostic Monitor Model for MinMax Training
Generates critical judgments of policy model outputs
"""
import torch
import logging
from typing import List, Dict, Any, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType

from ..config.model_config import MonitorModelConfig
from ..data.dataset_processor import DatasetProcessor

logger = logging.getLogger(__name__)

class MonitorModel:
    """
    Dataset-agnostic monitor model that generates critical judgments of policy outputs
    """
    
    def __init__(self, config: MonitorModelConfig, device: str = None):
        """
        Initialize monitor model
        
        Args:
            config: Monitor model configuration
            device: Single device to use (defaults to cuda:0 for PPO compatibility)
        """
        self.config = config
        self.target_device = device if device is not None else "cuda:0"
        if not torch.cuda.is_available():
            self.target_device = "cpu"
        
        logger.info(f"Loading monitor model: {config.model_name} on {self.target_device}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name,
            trust_remote_code=True,
            padding_side="left"
        )
        
        # Add pad token if missing
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Setup quantization if requested
        quantization_config = None
        if config.load_in_8bit:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False
            )
        
        # Use explicit device mapping instead of "auto"
        device_map = {"": self.target_device}  # Force all components to target device
        
        # Load model
        torch_dtype = getattr(torch, config.torch_dtype) if hasattr(torch, config.torch_dtype) else torch.bfloat16
        
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            device_map=device_map,
            quantization_config=quantization_config,
            torch_dtype=torch_dtype,
            trust_remote_code=True
        )
        
        # Setup LoRA if requested
        if config.use_lora:
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                target_modules=self._get_lora_target_modules()
            )
            self.model = get_peft_model(self.model, lora_config)
            logger.info("LoRA configuration applied to monitor model")
        
        # Set generation parameters
        self.generation_kwargs = {
            'max_new_tokens': config.max_new_tokens,
            'temperature': config.temperature,
            'do_sample': config.do_sample,
            'pad_token_id': self.tokenizer.pad_token_id,
            'eos_token_id': self.tokenizer.eos_token_id,
        }
        
        if config.top_p is not None:
            self.generation_kwargs['top_p'] = config.top_p
        
        # Update device to actual model device after loading
        self.device = self.target_device
        
        logger.info(f"Monitor model loaded successfully on {self.device}")
    
    def _get_lora_target_modules(self) -> List[str]:
        """Get LoRA target modules based on model architecture"""
        model_name = self.config.model_name.lower()
        
        if "mistral" in model_name or "mixtral" in model_name:
            return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        elif "llama" in model_name:
            return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        elif "gpt" in model_name:
            return ["c_attn", "c_proj", "c_fc"]
        else:
            # Default for most transformer models
            return ["q_proj", "k_proj", "v_proj", "o_proj"]
    
    def generate_critiques_batch(self, policy_outputs: List[str], 
                               questions: List[str], 
                               choices: List[List[str]],
                               stories: List[str],
                               dataset_processor: DatasetProcessor) -> List[str]:
        """
        Generate critical judgments for a batch of policy outputs
        
        Args:
            policy_outputs: List of policy model responses
            questions: List of questions  
            choices: List of answer choices for each question
            stories: list of story contexts
            dataset_processor: DatasetProcessor for prompt creation
            
        Returns:
            List of monitor critiques
        """
        logger.debug(f"Generating monitor critiques for batch size: {len(policy_outputs)}")
        
        # Parse answers from policy outputs for template
        parsed_answers = []
        for output in policy_outputs:
            # Simple parsing - extract A/B from beginning
            output_upper = output.strip().upper()
            if output_upper.startswith('A'):
                parsed_answers.append('A')
            elif output_upper.startswith('B'):
                parsed_answers.append('B')
            else:
                parsed_answers.append('A')  # Fallback
        
        critique_prompts = dataset_processor.create_monitor_critique_inputs_batch(
            stories, questions, choices, parsed_answers, policy_outputs
        )
        
        # Debug: Log first prompt to check format
        if critique_prompts:
            logger.debug(f"Monitor prompt example (first 300 chars): {critique_prompts[0][:300]}...")

        # Tokenize prompts
        inputs = self.tokenizer(
            critique_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_length
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate critiques
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                **self.generation_kwargs
            )
        
        # Extract generated critiques
        # Since we use left padding, all inputs have the same length
        input_length = inputs['input_ids'].size(1)
        
        # Extract generated tokens for all sequences at once
        generated_tokens_batch = outputs[:, input_length:]
        
        # Decode all sequences
        critiques = []
        for i in range(len(outputs)):
            critique = self.tokenizer.decode(generated_tokens_batch[i], skip_special_tokens=True)
            critiques.append(critique.strip())
        
        logger.debug("Monitor critiques generated successfully")
        return critiques
    
    def generate_batch(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, 
                      max_new_tokens: Optional[int] = None) -> torch.Tensor:
        """
        Generate batch of monitor critiques
        
        Args:
            input_ids: Input token ids
            attention_mask: Attention mask
            max_new_tokens: Override max_new_tokens if provided
            
        Returns:
            Generated token sequences
        """
        # Update generation kwargs if needed
        generation_kwargs = self.generation_kwargs.copy()
        if max_new_tokens is not None:
            generation_kwargs['max_new_tokens'] = max_new_tokens
        
        # Move inputs to model device
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **generation_kwargs
            )
        
        return outputs
    
    def get_model_for_training(self) -> torch.nn.Module:
        """Get the model for training purposes"""
        return self.model
    
    def get_tokenizer(self) -> AutoTokenizer:
        """Get the tokenizer"""
        return self.tokenizer
    
    def save_model(self, save_path: str):
        """
        Save the monitor model (handles both LoRA and full model saving)
        
        Args:
            save_path: Directory path to save the model
        """
        import os
        os.makedirs(save_path, exist_ok=True)
        
        # Save the model (LoRA adapters if using PEFT)
        self.model.save_pretrained(save_path)
        
        # Save the tokenizer
        self.tokenizer.save_pretrained(save_path)
        
        logger.info(f"Monitor model saved to {save_path}")
    
    def load_model(self, load_path: str):
        """Load a saved monitor model"""
        # This would implement loading logic for fine-tuned monitor models
        logger.info(f"Loading monitor model from {load_path}")
        # Implementation depends on whether using LoRA or full fine-tuning

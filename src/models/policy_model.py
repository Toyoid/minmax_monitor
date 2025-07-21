"""
Policy Model: Dataset-Agnostic Model for RLHF training
Supports various architectures (Mistral, GPT-2, LLaMA) with LoRA
"""
import torch
import logging
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig, 
    get_peft_model, 
    TaskType, 
    prepare_model_for_kbit_training
)
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class PolicyModel:
    def __init__(self, config=None, device=None):
        """
        Initialize policy model with configuration - Dataset Agnostic
        
        Args:
            config: PolicyModelConfig object with model parameters
            device: Target device for model loading (overrides device_map)
        """
        # Store config for external access
        self.config = config
        
        if config is None:
            # Fallback to defaults
            self.model_name = "mistralai/Mistral-7B-Instruct-v0.3"
            self.load_in_8bit = True
            self.lora_r = 16
            self.lora_alpha = 32
            self.lora_dropout = 0.1
            self.max_length = 512
            self.max_new_tokens = 50
            self.temperature = 0.7
            self.top_p = 0.9
            self.do_sample = True
            # Add use_lora configuration option  
            self.use_lora = True  # Default to True for backward compatibility
        else:
            self.model_name = config.model_name
            self.load_in_8bit = config.load_in_8bit
            self.lora_r = config.lora_r
            self.lora_alpha = config.lora_alpha
            self.lora_dropout = config.lora_dropout
            self.max_length = config.max_length
            self.max_new_tokens = config.max_new_tokens
            self.temperature = config.temperature
            self.top_p = config.top_p
            self.do_sample = config.do_sample
            # Add use_lora configuration option
            self.use_lora = getattr(config, 'use_lora', True)  # Default to True for backward compatibility
            
        # Set target device to cuda:0 for PPO compatibility
        self.target_device = device if device is not None else "cuda:0"
        if not torch.cuda.is_available():
            self.target_device = "cpu"
            
        logger.info(f"Policy model loading on: {self.target_device}")
        
        # Load tokenizer
        logger.info(f"Loading tokenizer for {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            use_fast=True,  # Use fast tokenizer to avoid sentencepiece issues
            trust_remote_code=True,
            padding_side='left'  # Essential for decoder-only models like DialoGPT/Mistral
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Ensure left padding for decoder-only architecture
        self.tokenizer.padding_side = 'left'
        logger.info(f"Tokenizer padding side set to: {self.tokenizer.padding_side}")
            
        # Configure quantization for memory efficiency
        if self.load_in_8bit:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
            )
        else:
            quantization_config = None
            
        # Load base model with explicit device mapping
        logger.info(f"Loading model {self.model_name} on {self.target_device}")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=quantization_config,
            device_map={"": self.target_device},  # Force single device
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            low_cpu_mem_usage=True  # Important for multi-user environment
        )
        
        # Select the appropriate LoRA target modules based on model type (only if using LoRA)
        if self.use_lora:
            if "gpt2" in self.model_name.lower() or "dialogpt" in self.model_name.lower():
                # GPT-2/DialoGPT architecture
                target_modules = ["c_attn", "c_proj"]
                logger.info(f"Using GPT-2/DialoGPT target modules: {target_modules}")
            elif "mistral" in self.model_name.lower():
                # Mistral architecture
                target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]
                logger.info(f"Using Mistral target modules: {target_modules}")
            elif "llama" in self.model_name.lower():
                # LLaMA architecture
                target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]
                logger.info(f"Using LLaMA target modules: {target_modules}")
            else:
                # Default to more common modules
                target_modules = ["query", "value", "key", "dense"]
                logger.warning(f"Unknown model architecture for {self.model_name}. Using default target modules: {target_modules}")
        
        # 3. Prepare the model for k-bit training (enables gradient checkpointing and correct dtype)
        if self.load_in_8bit:
            self.model = prepare_model_for_kbit_training(self.model)

        # Configure and apply LoRA (only if use_lora is True)
        if self.use_lora:
            # Configure LoRA
            lora_config = LoraConfig(
                r=self.lora_r,
                lora_alpha=self.lora_alpha,
                target_modules=target_modules,
                lora_dropout=self.lora_dropout,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )
            
            # Apply LoRA
            logger.info("Applying LoRA configuration")
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()    
        else:
            # For non-PEFT models, ensure all parameters are trainable for full fine-tuning
            logger.info("No PEFT/LoRA - using full model fine-tuning")
            for param in self.model.parameters():
                param.requires_grad = True
                
            # Verify trainable parameters
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self.model.parameters())
            logger.info(f"Policy model - Trainable parameters (full fine-tuning): {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")
        
        logger.info(f"Policy model loaded successfully on {self.target_device}")
        
    def generate_batch(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, 
                      max_new_tokens: Optional[int] = None, **generation_kwargs) -> torch.Tensor:
        """
        Dataset-agnostic batch generation - returns raw token outputs
        
        Args:
            input_ids: Tokenized input sequences [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            max_new_tokens: Maximum new tokens to generate
            **generation_kwargs: Additional generation parameters
            
        Returns:
            Generated token sequences [batch_size, total_seq_len]
        """
        if max_new_tokens is None:
            max_new_tokens = self.max_new_tokens
            
        # Merge generation parameters
        gen_kwargs = {
            'max_new_tokens': max_new_tokens,
            'do_sample': self.do_sample,
            'temperature': self.temperature,
            'top_p': self.top_p,
            'pad_token_id': self.tokenizer.eos_token_id,
            **generation_kwargs
        }
        
        # Move inputs to model device
        input_ids = input_ids.to(self.target_device)
        attention_mask = attention_mask.to(self.target_device)
        
        # Set model to evaluation mode for generation (prevents gradient checkpointing warnings)
        self.model.eval()
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **gen_kwargs
            )
            
        return outputs
        
    def get_model_for_training(self):
        """Return model for training"""
        return self.model
        
    def get_tokenizer(self):
        """Return tokenizer"""
        return self.tokenizer
    
    def get_device(self) -> str:
        """Get the device this model is loaded on"""
        return self.target_device
    
    def save_model(self, save_path: str):
        """
        Save the policy model (handles both LoRA and full model saving)
        
        Args:
            save_path: Directory path to save the model
        """
        import os
        os.makedirs(save_path, exist_ok=True)
        
        # Save the model (LoRA adapters if using PEFT)
        self.model.save_pretrained(save_path)
        
        # Save the tokenizer
        self.tokenizer.save_pretrained(save_path)
        
        logger.info(f"Policy model saved to {save_path}")
    
    def load_model(self, load_path: str):
        """
        Load a saved policy model
        
        Args:
            load_path: Directory path to load the model from
        """
        # This would implement loading logic for fine-tuned policy models
        logger.info(f"Loading policy model from {load_path}")
        # Implementation depends on whether using LoRA or full fine-tuning

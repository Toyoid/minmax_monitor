"""
Reward Model: Dataset-Agnostic Model for RLHF reward scoring
Supports multiple reward model types via plugin architecture
"""
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import logging
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)

class RewardModel:
    def __init__(self, config=None, device=None):
        """
        Initialize reward model with configuration - Single Device Mode
        
        Args:
            config: RewardModelConfig object with model parameters
            device: Target device for model loading (single device only)
        """
        if config is None:
            # Fallback to defaults
            self.model_name = "OpenAssistant/reward-model-deberta-v3-large-v2"
            self.max_length = 512
        else:
            self.model_name = config.model_name
            self.max_length = config.max_length
            
        # Set device configuration (simplified)
        self.device = device if device is not None else "cuda:1"  # Default to cuda:1 for reward
        if not torch.cuda.is_available():
            self.device = "cpu"
            
        logger.info(f"Loading reward model {self.model_name} on device: {self.device}")
        
        # Load model on single device with error handling
        try:
            logger.info(f"Loading reward model on {self.device}")
            
            # Get plugin for this model type
            from .reward_plugins import get_plugin
            self.plugin = get_plugin(self.model_name)
            logger.info(f"Using plugin for model type: {type(self.plugin).__name__}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                use_fast=True,
                trust_remote_code=True
            )
            
            # Load model on specific device
            if self.device == "cpu":
                device_map = "cpu"
                torch_dtype = torch.float32  # Use float32 for CPU
            else:
                device_map = {"": self.device}
                torch_dtype = torch.bfloat16
            
            # Get plugin-specific model kwargs
            model_kwargs = {
                "torch_dtype": torch_dtype,
                "device_map": device_map,
                "trust_remote_code": True,
                "low_cpu_mem_usage": True,
                **self.plugin.get_model_kwargs()  # Plugin-specific parameters
            }
            
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                **model_kwargs
            )
            self.model.eval()
            
            logger.info(f"Reward model loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load reward model on {self.device}: {e}")
            raise RuntimeError(f"Failed to load reward model: {e}")
        
    def get_device(self) -> str:
        """Get the device this model is loaded on"""
        return self.device
        
    def score_batch(self, conversations: List[List[Dict]]) -> torch.Tensor:
        """
        Score batch of conversations using plugin-based processing
        
        Args:
            conversations: List of conversation dictionaries with 'role' and 'content' keys
            
        Returns:
            Reward scores [batch_size] (processed by plugin)
        """
        try:
            # Use plugin to preprocess conversations
            formatted_texts = self.plugin.preprocess_conversations(conversations, self.tokenizer)

            # # print the formatted texts for debugging
            # print("\n======================================================")
            # for i, text in enumerate(formatted_texts):
            #     logger.debug(f"Reward Model Formatted Input Text {i}: \n{text}")
            # print("======================================================\n")
            
            # Tokenize the formatted texts
            inputs = self.tokenizer(
                formatted_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length
            ).to(self.device)
            
            # Get scores from model
            with torch.no_grad():
                outputs = self.model(**inputs)
                raw_scores = outputs.logits
            
            # Use plugin to postprocess scores
            processed_scores = self.plugin.postprocess_scores(raw_scores)
            return processed_scores.float().cpu()  # Return on CPU for consistency
                
        except Exception as e:
            logger.error(f"Error during reward model batch scoring on {self.device}: {e}")
            batch_size = len(conversations)
            return torch.zeros(batch_size, dtype=torch.float32)
    
    def get_model(self):
        """Return the underlying model"""
        return self.model

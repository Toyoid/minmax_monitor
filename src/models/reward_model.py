"""
Reward Model: Dataset-Agnostic Model for RLHF reward scoring
Designed for OpenAssistant reward model (single scalar output)
Supports multi-device inference for better performance
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
            
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                torch_dtype=torch_dtype,
                device_map=device_map,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            self.model.eval()
            
            logger.info(f"Reward model loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load reward model on {self.device}: {e}")
            raise RuntimeError(f"Failed to load reward model: {e}")
        
    def get_device(self) -> str:
        """Get the device this model is loaded on"""
        return self.device
        
    def score_batch(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Dataset-agnostic batch scoring - Single Device Mode
        Designed for OpenAssistant reward model which outputs single scalar scores
        
        Args:
            input_ids: Tokenized input sequences [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Raw reward scores [batch_size] (unormalized logits)
        """
        try:
            # Move inputs to model device
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            
            with torch.no_grad():
                outputs = self.model(input_ids, attention_mask)
                # OpenAssistant reward model outputs single scalar scores
                # logits shape: [batch_size, 1] or [batch_size]
                scores = outputs.logits.squeeze(-1)  # Remove last dimension if it's 1
                return scores.float().cpu()  # Return on CPU for consistency
                
        except Exception as e:
            logger.error(f"Error during reward model batch scoring on {self.device}: {e}")
            batch_size = input_ids.size(0)
            return torch.zeros(batch_size, dtype=torch.float32)
    
    def get_model(self):
        """Return the underlying model"""
        return self.model

"""
Skywork Reward Model Plugin
Handles Skywork reward model specific processing
"""
from typing import Dict, Any, List
import torch
from . import RewardModelPlugin, register_plugin

@register_plugin("skywork")
class SkyworkPlugin(RewardModelPlugin):
    """Plugin for Skywork reward models"""
    
    def get_model_kwargs(self) -> Dict[str, Any]:
        """Skywork models require specific loading parameters"""
        return {
            # "attn_implementation": "flash_attention_2",  # Disabled - requires flash-attn package
            "num_labels": 1,
        }
    
    def preprocess_conversations(self, conversations: List[List[Dict]], tokenizer) -> List[str]:
        """Apply chat template with BOS token removal for Skywork"""
        formatted_texts = []
        for conv in conversations:
            # Check if tokenizer has a chat template
            if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template is not None:
                try:
                    formatted = tokenizer.apply_chat_template(conv, tokenize=False)
                    
                    # Remove BOS token if present (Skywork requirement)
                    if (tokenizer.bos_token and 
                        formatted.startswith(tokenizer.bos_token)):
                        formatted = formatted[len(tokenizer.bos_token):]
                except Exception as e:
                    raise ValueError(f"Failed to apply chat template: {e}")
            else:
                # Manual formatting for models without chat template
                raise
                # formatted = self._manual_format_conversation(conv)
                
            formatted_texts.append(formatted)
        return formatted_texts
    
    def _manual_format_conversation(self, conversation: List[Dict]) -> str:
        """Manually format conversation for Skywork models"""
        formatted_parts = []
        for turn in conversation:
            role = turn.get('role', 'user')
            content = turn.get('content', '')
            if role == 'user':
                formatted_parts.append(f"<|user|>\n{content}\n")
            elif role == 'assistant':
                formatted_parts.append(f"<|assistant|>\n{content}\n")
        return "".join(formatted_parts)
    
    def postprocess_scores(self, raw_scores: torch.Tensor) -> torch.Tensor:
        """Skywork returns scores as-is, just squeeze if needed"""
        return raw_scores.squeeze(-1) if raw_scores.dim() > 1 else raw_scores

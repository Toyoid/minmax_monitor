"""
OpenAssistant Reward Model Plugin
Handles OpenAssistant reward model specific processing
"""
from typing import Dict, Any, List
import torch
from . import RewardModelPlugin, register_plugin

@register_plugin("openassistant")
class OpenAssistantPlugin(RewardModelPlugin):
    """Plugin for OpenAssistant reward models"""
    
    def get_model_kwargs(self) -> Dict[str, Any]:
        """OpenAssistant models don't need special loading parameters"""
        return {}
    
    def preprocess_conversations(self, conversations: List[List[Dict]], tokenizer) -> List[str]:
        """Apply chat template or fallback to manual formatting for OpenAssistant"""
        formatted_texts = []
        for conv in conversations:
            # Check if tokenizer has a chat template
            if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template is not None:
                formatted = tokenizer.apply_chat_template(conv, tokenize=False)
            else:
                # Manual formatting for models without chat template (like DeBERTa)
                formatted = self._manual_format_conversation(conv)
            formatted_texts.append(formatted)
        return formatted_texts
    
    def _manual_format_conversation(self, conversation: List[Dict]) -> str:
        """Manually format conversation for OpenAssistant models"""
        formatted_parts = []
        for turn in conversation:
            role = turn.get('role', 'user')
            content = turn.get('content', '')
            if role == 'user':
                formatted_parts.append(f"<|prompter|>{content}<|endoftext|>")
            elif role == 'assistant':
                formatted_parts.append(f"<|assistant|>{content}<|endoftext|>")
        return "".join(formatted_parts)
    
    def postprocess_scores(self, raw_scores: torch.Tensor) -> torch.Tensor:
        """OpenAssistant returns scores as-is, just squeeze if needed"""
        return raw_scores.squeeze(-1) if raw_scores.dim() > 1 else raw_scores

"""
Reward Model Plugin System
Provides extensible architecture for different reward model types
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List
import torch

class RewardModelPlugin(ABC):
    """Abstract base class for reward model plugins"""
    
    @abstractmethod
    def get_model_kwargs(self) -> Dict[str, Any]:
        """Return model-specific loading parameters"""
        pass
    
    @abstractmethod
    def preprocess_conversations(self, conversations: List[List[Dict]], tokenizer) -> List[str]:
        """Convert conversations to model-specific format"""
        pass
    
    @abstractmethod
    def postprocess_scores(self, raw_scores: torch.Tensor) -> torch.Tensor:
        """Post-process raw model scores if needed"""
        pass

# Plugin registry
REWARD_MODEL_PLUGINS = {}

def register_plugin(model_type: str):
    """Decorator to register plugins"""
    def decorator(plugin_class):
        REWARD_MODEL_PLUGINS[model_type] = plugin_class
        return plugin_class
    return decorator

def get_plugin(model_name: str) -> RewardModelPlugin:
    """Get appropriate plugin based on model name"""
    model_name_lower = model_name.lower()
    
    if "openassistant" in model_name_lower or "reward-model-deberta" in model_name_lower:
        return REWARD_MODEL_PLUGINS["openassistant"]()
    elif "skywork" in model_name_lower:
        return REWARD_MODEL_PLUGINS["skywork"]()
    else:
        supported_models = ["OpenAssistant/reward-model-deberta-v3-large-v2", "Skywork/Skywork-Reward-V2-Llama-3.1-8B"]
        raise ValueError(f"Unsupported reward model: {model_name}. Supported models: {', '.join(supported_models)}")

# Import plugins to register them
from .openassistant import OpenAssistantPlugin
from .skywork import SkyworkPlugin

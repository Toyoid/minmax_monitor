"""
Centralized Configuration for RLHF Deception Training
"""
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import json
import os

@dataclass
class PolicyModelConfig:
    """Configuration for Policy Model"""
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.3"
    load_in_8bit: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    device_map: str = "auto"
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True
    
    # Token parameters - will be set from TokenConfig
    max_new_tokens: int = 512
    max_length: int = 2048

@dataclass
class RewardModelConfig:
    """Configuration for Reward Model"""
    model_name: str = "OpenAssistant/reward-model-deberta-v3-large-v2"
    device_map: Optional[str] = "auto"
    torch_dtype: str = "bfloat16"
    
    # Token parameters - will be set from TokenConfig
    max_length: int = 512

@dataclass
class JudgeModelConfig:
    """Configuration for Judge Model"""
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.3"
    device_map: Optional[str] = "auto"
    temperature: float = 1.0
    do_sample: bool = False
    torch_dtype: str = "bfloat16"
    load_in_8bit: bool = True
    
    # Token parameters - will be set from TokenConfig
    max_new_tokens: int = 50
    max_length: int = 1024

@dataclass
class MonitorModelConfig:
    """Configuration for Monitor Model"""
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.3"
    load_in_8bit: bool = True
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    device_map: str = "auto"
    temperature: float = 0.8
    top_p: Optional[float] = 0.9
    do_sample: bool = True
    torch_dtype: str = "bfloat16"
    
    # Token parameters - will be set from TokenConfig
    max_new_tokens: int = 200
    max_length: int = 2048

@dataclass
class TrainingConfig:
    """Configuration for RLHF Training"""
    learning_rate: float = 1e-5
    batch_size: int = 16
    mini_batch_size: int = 4
    ppo_epochs: int = 2
    num_epochs: int = 8
    init_kl_coef: float = 0.02
    target_kl: float = 6.0
    cliprange: float = 0.2
    cliprange_value: float = 0.2
    vf_coef: float = 0.1
    max_grad_norm: float = 0.5
    reward_weight: float = 0.8
    judge_weight: float = 0.2
    eval_frequency: int = 2
    save_frequency: int = 5

@dataclass
class MinMaxTrainingConfig:
    """Configuration for MinMax Training with Dual PPO and TTUR"""
    # TTUR Configuration - Monitor (inner) updated more than Policy (outer)
    policy_learning_rate: float = 1e-5      # Lower LR for outer loop
    monitor_learning_rate: float = 3e-4     # Higher LR for inner loop
    monitor_updates_per_policy: int = 5     # Monitor updated 5x more frequently
    
    # Constraint and Dual Ascent Parameters
    constraint_penalty_lambda: float = 0.1  # Initial λ
    constraint_threshold: float = 0.5       # Violation threshold
    dual_ascent_step_size: float = 0.01     # λ update rate
    
    # Reward Combination
    reward_weight: float = 0.5
    judge_weight: float = 0.5
    
    # PPO Parameters
    batch_size: int = 16
    mini_batch_size: int = 4
    ppo_epochs: int = 1                     # Reduced for faster TTUR updates
    num_epochs: int = 8
    init_kl_coef: float = 0.02
    target_kl: float = 6.0
    cliprange: float = 0.2
    cliprange_value: float = 0.2
    vf_coef: float = 0.1
    max_grad_norm: float = 0.5
    eval_frequency: int = 1
    save_frequency: int = 2
    num_train_samples: Optional[int] = None

@dataclass
class EvaluationConfig:
    """Configuration for Deception Evaluation"""
    reward_threshold: float = 0.5
    judge_threshold: float = 0.5
    correctness_threshold: float = 0.3
    num_eval_samples: int = 100
    log_frequency: int = 10
    progress_frequency: int = 5

@dataclass
class DataConfig:
    """Configuration for Dataset and Token Management"""
    data_dir: str = "/data/lhx/minmax_monitor/dataset/qa_simple/data"
    max_story_length: int = 4096
    max_question_length: int = 256
    # Computed token parameters - will be synced from model configs
    reserved_generation_tokens: int = 512  # Computed from model configs
    safety_margin: int = 40
    max_story_tokens: Optional[int] = None  # Computed based on pipeline type

@dataclass
class RLHFConfig:
    """Master configuration for RLHF system"""
    policy_model: PolicyModelConfig = field(default_factory=PolicyModelConfig)
    reward_model: RewardModelConfig = field(default_factory=RewardModelConfig)
    judge_model: JudgeModelConfig = field(default_factory=JudgeModelConfig)
    monitor_model: MonitorModelConfig = field(default_factory=MonitorModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    minmax_training: MinMaxTrainingConfig = field(default_factory=MinMaxTrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    data: DataConfig = field(default_factory=DataConfig)
    save_dir: str = "./outputs"
    seed: int = 42

    def __post_init__(self):
        """Initialize token parameters in data config from model configs"""
        self._sync_token_parameters()

    def _sync_token_parameters(self):
        """Sync token parameters from model configs to data config"""
        # For RLHF training, use policy's max_new_tokens as reserved_generation_tokens
        self.data.reserved_generation_tokens = self.policy_model.max_new_tokens

    def compute_max_story_tokens(self, pipeline_type: str = "rlhf", 
                               template_overhead: int = 200) -> int:
        """
        Compute max_story_tokens based on pipeline type and model constraints
        
        Args:
            pipeline_type: "minmax" or "rlhf"
            template_overhead: Tokens used by prompt template (excluding story)
            
        Returns:
            Maximum tokens available for story content
        """
        if pipeline_type == "rlhf":
            # For RLHF: policy prompt only needs story + question + choices + template
            available_tokens = (self.policy_model.max_length - 
                              template_overhead - 
                              self.policy_model.max_new_tokens - 
                              self.data.safety_margin)
        else:  # MinMax mode for RLHFConfig (if monitor is available)
            # For MinMax: monitor prompt includes policy generation
            available_tokens = (self.monitor_model.max_length - 
                              template_overhead - 
                              self.policy_model.max_new_tokens -  # Policy generation in monitor input
                              self.monitor_model.max_new_tokens - # Monitor generation
                              self.data.safety_margin)
        
        self.data.max_story_tokens = max(available_tokens, 100)  # Minimum 100 tokens
        return self.data.max_story_tokens

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'RLHFConfig':
        """Create config from dictionary"""
        config = cls()
        
        for section, values in config_dict.items():
            if hasattr(config, section) and isinstance(values, dict):
                section_config = getattr(config, section)
                for key, value in values.items():
                    if hasattr(section_config, key):
                        setattr(section_config, key, value)
            elif hasattr(config, section):
                setattr(config, section, values)
        
        # Re-sync token parameters after loading from dict
        config._sync_token_parameters()
        return config
    
    @classmethod
    def from_file(cls, config_path: str) -> 'RLHFConfig':
        """Load config from JSON file"""
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        result = {}
        for field_name, field_obj in self.__dataclass_fields__.items():
            value = getattr(self, field_name)
            if hasattr(value, '__dict__'):
                result[field_name] = value.__dict__
            else:
                result[field_name] = value
        return result
    
    def save(self, config_path: str):
        """Save config to JSON file"""
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

@dataclass
class MinMaxConfig:
    """Main MinMax Configuration - focused configuration for minmax training"""
    policy_model: PolicyModelConfig = field(default_factory=PolicyModelConfig)
    monitor_model: MonitorModelConfig = field(default_factory=MonitorModelConfig)
    reward_model: RewardModelConfig = field(default_factory=RewardModelConfig)
    judge_model: JudgeModelConfig = field(default_factory=JudgeModelConfig)
    training: MinMaxTrainingConfig = field(default_factory=MinMaxTrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    data: DataConfig = field(default_factory=DataConfig)
    save_dir: str = "./outputs"
    seed: int = 42

    def __post_init__(self):
        """Initialize token parameters in data config from model configs"""
        self._sync_token_parameters()

    def _sync_token_parameters(self):
        """Sync token parameters from model configs to data config"""
        # Compute reserved_generation_tokens based on pipeline type
        # For MinMax training, use monitor's max_new_tokens (more restrictive)
        # For RLHF training, use policy's max_new_tokens
        self.data.reserved_generation_tokens = max(
            self.policy_model.max_new_tokens,
            self.monitor_model.max_new_tokens
        )

    def compute_max_story_tokens(self, pipeline_type: str = "minmax", 
                               template_overhead: int = 200) -> int:
        """
        Compute max_story_tokens based on pipeline type and model constraints
        
        Args:
            pipeline_type: "minmax" or "rlhf"
            template_overhead: Tokens used by prompt template (excluding story)
            
        Returns:
            Maximum tokens available for story content
        """
        if pipeline_type == "minmax":
            # For MinMax: monitor prompt includes policy generation, so account for both
            # Monitor prompt = story + question + choices + policy_response + template
            available_tokens = (self.monitor_model.max_length - 
                              template_overhead - 
                              self.policy_model.max_new_tokens -  # Policy generation in monitor input
                              self.monitor_model.max_new_tokens - # Monitor generation
                              self.data.safety_margin)
        else:  # RLHF
            # For RLHF: policy prompt only needs story + question + choices + template
            available_tokens = (self.policy_model.max_length - 
                              template_overhead - 
                              self.policy_model.max_new_tokens - 
                              self.data.safety_margin)
        
        self.data.max_story_tokens = max(available_tokens, 100)  # Minimum 100 tokens
        return self.data.max_story_tokens

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'MinMaxConfig':
        """Create config from dictionary"""
        config = cls()
        
        for section, values in config_dict.items():
            if hasattr(config, section) and isinstance(values, dict):
                section_config = getattr(config, section)
                for key, value in values.items():
                    if hasattr(section_config, key):
                        setattr(section_config, key, value)
            elif hasattr(config, section):
                setattr(config, section, values)
        
        # Re-sync token parameters after loading from dict
        config._sync_token_parameters()
        return config
    
    @classmethod
    def from_file(cls, config_path: str) -> 'MinMaxConfig':
        """Load config from JSON file"""
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        result = {}
        for field_name, field_obj in self.__dataclass_fields__.items():
            value = getattr(self, field_name)
            if hasattr(value, '__dict__'):
                result[field_name] = value.__dict__
            else:
                result[field_name] = value
        return result

def get_default_config() -> RLHFConfig:
    """Get default configuration"""
    return RLHFConfig()

def load_config_from_args(args) -> RLHFConfig:
    """Load configuration from command line arguments"""
    if hasattr(args, 'config') and args.config and os.path.exists(args.config):
        config = RLHFConfig.from_file(args.config)
    else:
        config = get_default_config()
    
    # Override with command line arguments
    if hasattr(args, 'save_dir') and args.save_dir:
        config.save_dir = args.save_dir
    if hasattr(args, 'num_epochs') and args.num_epochs:
        config.training.num_epochs = args.num_epochs
    if hasattr(args, 'batch_size') and args.batch_size:
        config.training.batch_size = args.batch_size
    if hasattr(args, 'learning_rate') and args.learning_rate:
        config.training.learning_rate = args.learning_rate
    
    return config

def dict_to_config(config_dict: dict, config_class):
    """Convert dictionary to config object"""
    # Filter to only include fields that exist in the config class
    valid_fields = {field.name for field in config_class.__dataclass_fields__.values()}
    filtered_dict = {k: v for k, v in config_dict.items() if k in valid_fields}
    return config_class(**filtered_dict)

def create_model_configs_from_dict(config_dict: dict):
    """Create model config objects from a dictionary"""
    policy_config = dict_to_config(config_dict.get('policy_model', {}), PolicyModelConfig)
    reward_config = dict_to_config(config_dict.get('reward_model', {}), RewardModelConfig)
    judge_config = dict_to_config(config_dict.get('judge_model', {}), JudgeModelConfig)
    monitor_config = dict_to_config(config_dict.get('monitor_model', {}), MonitorModelConfig)
    return policy_config, reward_config, judge_config, monitor_config

def create_minmax_model_configs_from_dict(config_dict: Dict[str, Any]):
    """Create model configs from dictionary for minmax training"""
    policy_config = PolicyModelConfig(**config_dict.get("policy_model", {}))
    monitor_config = MonitorModelConfig(**config_dict.get("monitor_model", {}))
    reward_config = RewardModelConfig(**config_dict.get("reward_model", {}))
    judge_config = JudgeModelConfig(**config_dict.get("judge_model", {}))
    
    return policy_config, monitor_config, reward_config, judge_config

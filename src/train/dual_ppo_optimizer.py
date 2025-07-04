"""
Dual PPO Optimizer for MinMax RLHF Training
Implements corrected TTUR with Monitor (inner loop) and Policy (outer loop)
"""
import torch
import logging
from typing import List, Dict, Any, Optional, Tuple

from ..data.dataset_processor import MinMaxOutput
from ..config.model_config import MinMaxTrainingConfig

# TRL imports for PPO training
try:
    from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
    TRL_AVAILABLE = True
except ImportError:
    TRL_AVAILABLE = False

logger = logging.getLogger(__name__)

class DualPPOMinMaxOptimizer:
    """
    Dual PPO optimizer with corrected TTUR for MinMax adversarial training
    - Monitor: Inner loop (updated more frequently)
    - Policy: Outer loop (updated less frequently)
    """
    
    def __init__(self, policy_model, monitor_model, config: MinMaxTrainingConfig, device_manager=None):
        """
        Initialize Dual PPO optimizer with corrected TTUR
        
        Args:
            policy_model: Policy model for PPO training
            monitor_model: Monitor model for PPO training
            config: MinMax training configuration
            device_manager: Device manager for multi-GPU coordination
        """
        self.policy_model = policy_model
        self.monitor_model = monitor_model
        self.config = config
        self.device_manager = device_manager
        
        if not TRL_AVAILABLE:
            raise RuntimeError("TRL is required for PPO training. Install with: pip install trl")
        
        # Initialize PPO trainers for both models
        self._setup_policy_ppo()
        self._setup_monitor_ppo()
        
        # TTUR configuration - Monitor updated more frequently
        self.monitor_updates_per_policy = getattr(config, 'monitor_updates_per_policy', 5)
        
        # Constraint multiplier for dual ascent
        self.lambda_multiplier = config.constraint_penalty_lambda
        self.constraint_threshold = getattr(config, 'constraint_threshold', 0.5)
        self.dual_ascent_step_size = getattr(config, 'dual_ascent_step_size', 0.01)
        
        logger.info(f"Dual PPO optimizer initialized:")
        logger.info(f"  - Policy LR: {config.policy_learning_rate} (outer loop)")
        logger.info(f"  - Monitor LR: {config.monitor_learning_rate} (inner loop)")
        logger.info(f"  - Monitor updates per policy: {self.monitor_updates_per_policy}")
        logger.info(f"  - Initial λ: {self.lambda_multiplier}")
    
    def _setup_policy_ppo(self):
        """Setup PPO trainer for policy model (outer loop)"""
        # Policy PPO Configuration - lower LR for outer loop
        policy_ppo_config = PPOConfig(
            model_name=self.policy_model.config.model_name,
            learning_rate=self.config.policy_learning_rate,
            batch_size=self.config.mini_batch_size,
            mini_batch_size=self.config.mini_batch_size,
            ppo_epochs=self.config.ppo_epochs,
            init_kl_coef=self.config.init_kl_coef,
            target_kl=self.config.target_kl,
            cliprange=self.config.cliprange,
            cliprange_value=self.config.cliprange_value,
            vf_coef=self.config.vf_coef,
            max_grad_norm=self.config.max_grad_norm,
            gradient_checkpointing=True,
            optimize_cuda_cache=True,
        )
        
        logger.info("Creating Policy PPO model on cuda:0")
        
        # Wrap policy model for PPO
        self.policy_ppo_model = AutoModelForCausalLMWithValueHead.from_pretrained(
            self.policy_model.get_model_for_training()
        )
        
        # Initialize Policy PPO trainer
        self.policy_ppo_trainer = PPOTrainer(
            config=policy_ppo_config,
            model=self.policy_ppo_model,
            tokenizer=self.policy_model.get_tokenizer(),
        )
        
        logger.info("✅ Policy PPO trainer initialized on cuda:0 (outer loop)")
    
    def _setup_monitor_ppo(self):
        """Setup PPO trainer for monitor model (inner loop)"""
        # Monitor PPO Configuration - higher LR for inner loop
        monitor_ppo_config = PPOConfig(
            model_name=self.monitor_model.config.model_name,
            learning_rate=self.config.monitor_learning_rate,
            batch_size=self.config.mini_batch_size,
            mini_batch_size=self.config.mini_batch_size,
            ppo_epochs=self.config.ppo_epochs,
            init_kl_coef=self.config.init_kl_coef,
            target_kl=self.config.target_kl,
            cliprange=self.config.cliprange,
            cliprange_value=self.config.cliprange_value,
            vf_coef=self.config.vf_coef,
            max_grad_norm=self.config.max_grad_norm,
            gradient_checkpointing=True,
            optimize_cuda_cache=True,
        )
        
        logger.info("Creating Monitor PPO model on cuda:0")
        
        # Wrap monitor model for PPO
        self.monitor_ppo_model = AutoModelForCausalLMWithValueHead.from_pretrained(
            self.monitor_model.get_model_for_training()
        )
        
        # Initialize Monitor PPO trainer
        self.monitor_ppo_trainer = PPOTrainer(
            config=monitor_ppo_config,
            model=self.monitor_ppo_model,
            tokenizer=self.monitor_model.get_tokenizer(),
        )
        
        logger.info("✅ Monitor PPO trainer initialized on cuda:0 (inner loop)")
    
    def step(self, minmax_output: MinMaxOutput) -> Dict[str, float]:
        """
        TTUR Step: Monitor (inner) updated more frequently than Policy (outer)
        
        Args:
            minmax_output: Output from minmax pipeline
            
        Returns:
            Dictionary with training metrics
        """
        # Extract PPO training data
        policy_queries, policy_responses = self._extract_policy_ppo_data(minmax_output)
        # Validate extracted policy data
        if len(policy_queries) == 0 or len(policy_responses) == 0:
            logger.error("No valid policy data extracted for PPO training")
            return {
                'policy_loss': 0.0, 'policy_kl': 0.0, 'policy_entropy': 0.0,
                'monitor_loss': 0.0, 'monitor_kl': 0.0, 'monitor_entropy': 0.0,
                'avg_violation': 0.0, 'lambda': self.lambda_multiplier,
                'policy_reward': 0.0, 'monitor_reward': 0.0
            }

        monitor_queries, monitor_responses = self._extract_monitor_ppo_data(minmax_output)
        # Validate extracted monitor data
        if len(monitor_queries) == 0 or len(monitor_responses) == 0:
            logger.error("No valid monitor data extracted for PPO training")
            return {
                'policy_loss': 0.0, 'policy_kl': 0.0, 'policy_entropy': 0.0,
                'monitor_loss': 0.0, 'monitor_kl': 0.0, 'monitor_entropy': 0.0,
                'avg_violation': 0.0, 'lambda': self.lambda_multiplier,
                'policy_reward': 0.0, 'monitor_reward': 0.0
            }
        
        # Use pipeline's computed rewards and constraints (no duplication!)
        combined_rewards = minmax_output.combined_scores
        truthfulness_penalties = minmax_output.truthfulness_penalty
        
        logger.debug(f"TTUR step - Combined rewards mean: {combined_rewards.mean():.4f}")
        logger.debug(f"TTUR step - Truthfulness penalties mean: {truthfulness_penalties.mean():.4f}")
        logger.debug(f"TTUR step - Current λ: {self.lambda_multiplier:.4f}")
        
        policy_metrics = {}
        monitor_metrics = {}
        # TTUR: Inner loop - Update monitor multiple times
        logger.debug(f"Inner loop: Updating monitor {self.monitor_updates_per_policy} times")
        for i in range(self.monitor_updates_per_policy):
            # Monitor objective: maximize E[-R(O,O') - λ·F(π_M)]
            monitor_rewards = -combined_rewards - self.lambda_multiplier * truthfulness_penalties
            
            # Convert rewards to list for PPO trainer
            monitor_reward_list = [torch.tensor(r.item(), dtype=torch.float32) for r in monitor_rewards]
            
            try:
                logger.debug(f"Monitor PPO step {i+1}: calling trainer.step()")
                
                step_metrics = self.monitor_ppo_trainer.step(
                    queries=monitor_queries,
                    responses=monitor_responses,
                    scores=monitor_reward_list
                )
                
                logger.debug(f"Monitor PPO step {i+1} completed successfully")
                
                if i == 0:  # Store metrics from first monitor update
                    monitor_metrics = step_metrics
                    
            except Exception as e:
                logger.warning(f"Monitor PPO step {i+1} failed: {e}")
                if i == 0:
                    monitor_metrics = {"ppo/loss/total": 0.0, "ppo/mean_kl": 0.0}
        
        # TTUR: Outer loop - Update policy once
        logger.debug("Outer loop: Updating policy 1 time")
        policy_rewards = combined_rewards
        
        # Convert rewards to list for PPO trainer
        policy_reward_list = [torch.tensor(r.item(), dtype=torch.float32) for r in policy_rewards]
        
        try:
            policy_metrics = self.policy_ppo_trainer.step(
                queries=policy_queries,
                responses=policy_responses,
                scores=policy_reward_list
            )
        except Exception as e:
            logger.warning(f"Policy PPO step failed: {e}")
            policy_metrics = {"ppo/loss/total": 0.0, "ppo/mean_kl": 0.0}
        
        # Update constraint multiplier using dual ascent
        avg_violation = truthfulness_penalties.mean().item()
        
        if avg_violation > self.constraint_threshold:
            self.lambda_multiplier *= (1.0 + self.dual_ascent_step_size)
        else:
            self.lambda_multiplier *= (1.0 - self.dual_ascent_step_size)
        
        # Clamp lambda to reasonable range
        self.lambda_multiplier = max(0.001, min(10.0, self.lambda_multiplier))
        
        logger.debug(f"Constraint update - Average violation: {avg_violation:.4f}")
        logger.debug(f"Constraint update - Updated λ: {self.lambda_multiplier:.4f}")
        
        return {
            'policy_loss': policy_metrics.get('ppo/loss/total', 0.0),
            'policy_kl': policy_metrics.get('ppo/mean_kl', 0.0),
            'policy_entropy': policy_metrics.get('ppo/mean_entropy', 0.0),
            'monitor_loss': monitor_metrics.get('ppo/loss/total', 0.0),
            'monitor_kl': monitor_metrics.get('ppo/mean_kl', 0.0),
            'monitor_entropy': monitor_metrics.get('ppo/mean_entropy', 0.0),
            'avg_violation': avg_violation,
            'lambda': self.lambda_multiplier,
            'policy_reward': policy_rewards.mean().item(),
            'monitor_reward': (-combined_rewards - self.lambda_multiplier * truthfulness_penalties).mean().item()
        }
    
    def _extract_policy_ppo_data(self, minmax_output: MinMaxOutput) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Extract queries and responses for Policy PPO training"""
        queries = []
        responses = []
        
        for i, input_len in enumerate(minmax_output.policy_input_lengths):
            full_sequence = minmax_output.generated_tokens[i]
            
            # Split into query (input) and response (generated)
            query = full_sequence[:input_len]
            response = full_sequence[input_len:]
            
            queries.append(query)
            responses.append(response)
        
        return queries, responses
    
    def _extract_monitor_ppo_data(self, minmax_output: MinMaxOutput) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Extract queries and responses for Monitor PPO training"""
        queries = []
        responses = []
        
        for i, input_len in enumerate(minmax_output.monitor_input_lengths):
            try:
                # Validate index bounds
                if i >= len(minmax_output.monitor_generated_tokens):
                    raise IndexError(f"Monitor input index {i} out of bounds")
                
                # The monitor_input_ids contains the full tokenized prompt + generated critique
                full_sequence = minmax_output.monitor_generated_tokens[i]
                
                # Validate sequence length
                if len(full_sequence) <= input_len:
                    raise ValueError(f"Monitor sequence {i} too short (len={len(full_sequence)}, input_len={input_len}), skipping")
                    
                
                # Split into query (input prompt) and response (generated critique)
                query = full_sequence[:input_len]
                response = full_sequence[input_len:]
                
                # Validate response is not empty
                if len(response) == 0:
                    raise ValueError(f"Monitor response {i} is empty")
                
                # Validate response contains non-padding tokens
                if self.monitor_model.get_tokenizer().pad_token_id is not None:
                    non_pad_tokens = (response != self.monitor_model.get_tokenizer().pad_token_id).sum()
                    if non_pad_tokens == 0:
                        raise ValueError(f"Monitor response {i} contains only padding tokens")
                
                queries.append(query)
                responses.append(response)
                
            except Exception as e:
                logger.warning(f"Error processing monitor data at index {i}: {e}")
                continue

        return queries, responses
    
    def get_training_state(self) -> Dict[str, Any]:
        """Get current training state for logging/checkpointing"""
        return {
            "lambda_constraint": self.lambda_multiplier,
            "constraint_threshold": self.constraint_threshold,
            "monitor_updates_per_policy": self.monitor_updates_per_policy,
            "policy_ppo_trainer_state": getattr(self.policy_ppo_trainer, 'state_dict', lambda: {})(),
            "monitor_ppo_trainer_state": getattr(self.monitor_ppo_trainer, 'state_dict', lambda: {})(),
        }
    
    def load_training_state(self, state: Dict[str, Any]):
        """Load training state from checkpoint"""
        self.lambda_multiplier = state.get("lambda_constraint", self.lambda_multiplier)
        self.constraint_threshold = state.get("constraint_threshold", self.constraint_threshold)
        self.monitor_updates_per_policy = state.get("monitor_updates_per_policy", self.monitor_updates_per_policy)
        
        if "policy_ppo_trainer_state" in state and hasattr(self.policy_ppo_trainer, 'load_state_dict'):
            self.policy_ppo_trainer.load_state_dict(state["policy_ppo_trainer_state"])
        
        if "monitor_ppo_trainer_state" in state and hasattr(self.monitor_ppo_trainer, 'load_state_dict'):
            self.monitor_ppo_trainer.load_state_dict(state["monitor_ppo_trainer_state"])
        
        logger.info("Dual PPO training state loaded from checkpoint")

    # Legacy compatibility methods (for gradual migration)
    def update_policy_ppo(self, minmax_output: MinMaxOutput) -> Dict[str, float]:
        """Legacy method - now handled by step()"""
        logger.warning("update_policy_ppo is deprecated, use step() instead")
        metrics = self.step(minmax_output)
        return {
            "policy_loss": metrics["policy_loss"],
            "policy_kl": metrics["policy_kl"],
            "policy_entropy": metrics["policy_entropy"],
            "policy_reward": metrics["policy_reward"],
        }
    
    def update_monitor_gradient(self, minmax_output: MinMaxOutput) -> Dict[str, float]:
        """Legacy method - now handled by step()"""
        logger.warning("update_monitor_gradient is deprecated, use step() instead")
        metrics = self.step(minmax_output)
        return {
            "monitor_loss": metrics["monitor_loss"],
            "monitor_reward_loss": metrics["monitor_reward"],
            "monitor_constraint_loss": metrics["avg_violation"],
            "lambda_constraint": metrics["lambda"],
        }
    
    def update_constraint_multiplier(self, avg_constraint_violation: float) -> float:
        """Legacy method - now handled by step()"""
        logger.warning("update_constraint_multiplier is deprecated, use step() instead")
        return self.lambda_multiplier

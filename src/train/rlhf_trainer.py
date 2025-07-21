#!/usr/bin/env python3
"""
Main RLHF Training Script  
"""
import os
import sys
import torch
import numpy as np
import logging
import json
import argparse
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.config.model_config import RLHFConfig
from src.config.dataset_config import QASimpleConfig
from src.data.qa_processor import create_qa_simple_dataloader
from src.data.dataset_processor import DatasetProcessor
from src.models.policy_model import PolicyModel
from src.models.reward_model import RewardModel
from src.models.judge_model import JudgeModel
from src.pipelines.rlhf_pipeline import RLHFPipeline
from src.utils.metrics_logger import MetricsLogger

# TRL imports for PPO training
try:
    from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
    TRL_AVAILABLE = True
except ImportError:
    print("TRL not available. Please install trl")
    TRL_AVAILABLE = False

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RLHFTrainer:
    """
    Dataset-agnostic RLHF trainer
    """
    
    def __init__(self, config_path: str, dataset_name: str = "qa_simple", eval_only: bool = False):
        # Setup GPU environment first
        from src.utils.env_setup import setup_gpu_environment
        from src.utils.device_manager import create_device_manager
        
        available_gpus = setup_gpu_environment()
        logger.info(f"Initializing RLHF trainer with {available_gpus} available GPUs")
        
        self.config_path = config_path
        self.dataset_name = dataset_name
        self.eval_only = eval_only
        
        # Load configuration
        with open(config_path) as f:
            config_dict = json.load(f)
        
        self.config = RLHFConfig.from_dict(config_dict)
        
        # Initialize device manager (simplified)
        self.device_manager = create_device_manager()
        
        self.device = torch.device(self.device_manager.get_policy_device())
        self.save_dir = self.config.save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Initialize MetricsLogger
        self._setup_metrics_logger()
        
        # Initialize components
        self._setup_dataset_config()
        self._setup_models()
        self._setup_pipeline()
        self._setup_data()
        
        if not eval_only and TRL_AVAILABLE:
            self._setup_ppo_trainer()
        
        logger.info("RLHF trainer initialized successfully")
    
    def _setup_metrics_logger(self):
        """Setup MetricsLogger with configurable backends"""
        # Extract logging config
        logging_config = getattr(self.config, 'logging', {})
        
        # Get backends list (default to wandb for backward compatibility)
        backends = getattr(logging_config, 'backends', ['wandb'])
        
        # Extract backend-specific configs
        wandb_config = getattr(logging_config, 'wandb_config', {})
        tensorboard_config = getattr(logging_config, 'tensorboard', {})
        
        # Create run name with timestamp
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"rlhf_{self.dataset_name}_{timestamp}"
        
        # Get project name from wandb config or use default
        project_name = wandb_config.get('project', 'rlhf')
        
        # Initialize MetricsLogger with multiple backends
        try:
            self.metrics_logger = MetricsLogger(
                project_name=project_name,
                run_name=run_name,
                config=self.config.to_dict(),
                trainer_type="rlhf",
                backends=backends,
                wandb_config=wandb_config,
                tensorboard_config=tensorboard_config
            )
            logger.info(f"MetricsLogger initialized for RLHF training with backends: {backends}")
        except Exception as e:
            logger.warning(f"Failed to initialize MetricsLogger: {e}. Training will continue without logging.")
            self.metrics_logger = None
    
    def _setup_dataset_config(self):
        """Setup dataset-specific configuration"""
        if self.dataset_name == "qa_simple":
            # Get data config from loaded configuration
            data_config = self.config.data
            
            # Create QASimpleConfig with values from config file
            self.dataset_config = QASimpleConfig(
                safety_margin=getattr(data_config, 'safety_margin', 40),
                story_truncation_strategy=getattr(data_config, 'story_truncation_strategy', 'intelligent'),
                answer_parsing_strategy=getattr(data_config, 'answer_parsing_strategy', 'ab_pattern'),
                response_extraction_strategy=getattr(data_config, 'response_extraction_strategy', 'auto'),
                answer_choices_format=getattr(data_config, 'answer_choices_format', 'AB'),
                num_choices=getattr(data_config, 'num_choices', 2)
            )
            self.data_dir = data_config.data_dir
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")
        
        logger.info(f"Dataset config: {self.dataset_name} with safety_margin={self.dataset_config.safety_margin}")
    
    def _setup_models(self):
        """Initialize all models with distributed device placement"""
        logger.info("Loading models with distributed device placement...")
        
        # Create model configs
        policy_config = self.config.policy_model
        reward_config = self.config.reward_model
        judge_config = self.config.judge_model
        
        # Initialize models on allocated devices
        self.policy_model = PolicyModel(
            policy_config, 
            device=self.device_manager.get_policy_device()
        )
        
        self.reward_model = RewardModel(
            reward_config,
            device=self.device_manager.get_reward_device()
        )
        
        self.judge_model = JudgeModel(
            judge_config,
            device=self.device_manager.get_judge_device()
        )
        
        logger.info("All models loaded successfully")
    
    def _setup_pipeline(self):
        """Setup the RLHF pipeline with device manager"""
        logger.info("Setting up RLHF pipeline...")
        
        # Create dataset processor
        self.processor = DatasetProcessor(
            self.dataset_config, 
            self.config.policy_model, 
            None,  # No monitor for RLHF
            "rlhf"
        )
        
        # Initialize pipeline with device manager
        self.pipeline = RLHFPipeline(
            self.policy_model,
            self.reward_model, 
            self.judge_model,
            self.processor,
            self.device_manager  
        )
        
        # Set reward weights from config
        self.pipeline.set_reward_weights(
            self.config.training.reward_weight,
            self.config.training.judge_weight
        )
        
        logger.info("RLHF pipeline initialized")
    
    def _setup_data(self):
        """Setup data loaders"""
        logger.info("Setting up data loaders...")
        
        batch_size = self.config.training.batch_size
        
        # Training data
        self.train_dataloader = create_qa_simple_dataloader(
            self.data_dir, "train", batch_size=batch_size, shuffle=True,
            dataset_processor=self.pipeline.processor
        )
        
        # Validation data
        self.val_dataloader = create_qa_simple_dataloader(
            self.data_dir, "val", batch_size=batch_size, shuffle=False,
            dataset_processor=self.pipeline.processor
        )
        
        logger.info(f"Data loaders created - Train: {len(self.train_dataloader.dataset)}, "
                   f"Val: {len(self.val_dataloader.dataset)} samples")
        
        # Log training sample limit if set
        num_train_samples = self.config.training.num_train_samples
        if num_train_samples is not None:
            logger.info(f"Training sample limit set to: {num_train_samples}")
        else:
            logger.info("No training sample limit set - will use full dataset")
    
    def _setup_ppo_trainer(self):
        """Setup PPO trainer for RLHF"""
        logger.info("Setting up PPO trainer...")
        
        # PPO Configuration with memory optimization
        ppo_config = PPOConfig(
            model_name=self.config.policy_model.model_name,
            learning_rate=self.config.training.learning_rate,
            batch_size=self.config.training.mini_batch_size,
            mini_batch_size=self.config.training.mini_batch_size,
            ppo_epochs=self.config.training.ppo_epochs,
            init_kl_coef=self.config.training.init_kl_coef,
            target_kl=self.config.training.target_kl,
            kl_penalty=self.config.training.kl_penalty,
            cliprange=self.config.training.cliprange,
            cliprange_value=self.config.training.cliprange_value,
            vf_coef=self.config.training.vf_coef,
            max_grad_norm=self.config.training.max_grad_norm,
            # Memory optimization settings
            gradient_checkpointing=True,  # Reduce memory at cost of compute
            optimize_cuda_cache=True,
        )
        
        # Get PPO device (may be different from policy device)
        ppo_device = self.device_manager.get_ppo_device()
        logger.info(f"PPO trainer will use device: {ppo_device}")
        
        # Get the model for PPO training and prepare it
        try:
            policy_model_for_training = self.policy_model.get_model_for_training()
            policy_model_for_training.train()  # Ensure training mode
            
            # Detect if this is actually a PEFT model
            is_peft_model = hasattr(policy_model_for_training, 'peft_config') or hasattr(policy_model_for_training, 'active_peft_config')
            logger.info(f"Policy model - PEFT detected: {is_peft_model}")
            
            # Verify trainable parameters before wrapping
            trainable_params = sum(p.numel() for p in policy_model_for_training.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in policy_model_for_training.parameters())
            logger.info(f"Policy model - Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")
            
            if trainable_params == 0:
                logger.error("Policy model has no trainable parameters! Check model configuration.")
                raise RuntimeError("Policy model has no trainable parameters")
            
            # Wrap policy model for PPO (correct way)
            self.ppo_model = AutoModelForCausalLMWithValueHead(policy_model_for_training)
            
            # Set the is_peft_model attribute that TRL expects
            self.ppo_model.is_peft_model = is_peft_model
            
            # Verify value head was added correctly
            ppo_trainable_after = sum(p.numel() for p in self.ppo_model.parameters() if p.requires_grad)
            logger.info(f"PPO model - Trainable parameters after value head: {ppo_trainable_after:,}")
            
        except Exception as e:
            logger.error(f"Failed to create PPO model: {e}")
            raise RuntimeError(f"PPO model creation failed: {e}")
        
        # Move PPO model to designated device if different from policy
        try:
            if ppo_device != self.device_manager.get_policy_device():
                logger.info(f"Moving PPO model from {self.device_manager.get_policy_device()} to {ppo_device}")
                self.ppo_model = self.ppo_model.to(ppo_device)
        except Exception as e:
            logger.error(f"Failed to move PPO model to device {ppo_device}: {e}")
            raise RuntimeError(f"Device movement failed: {e}")
        
        # Initialize PPO trainer
        try:
            self.ppo_trainer = PPOTrainer(
                config=ppo_config,
                model=self.ppo_model,
                tokenizer=self.policy_model.get_tokenizer(),
            )
        except Exception as e:
            logger.error(f"Failed to initialize PPO trainer: {e}")
            raise RuntimeError(f"PPO trainer initialization failed: {e}")
        
        logger.info("PPO trainer initialized")
    
    def _ppo_training_step(self, rlhf_output, combined_rewards_tensor):
        """
        Execute a single PPO training step
        
        Args:
            rlhf_output: Output from RLHF pipeline
            combined_rewards_tensor: Combined reward tensor
            
        Returns:
            PPO training statistics or None if failed
        """
        # Prepare data for PPO training
        queries = []
        responses = []
        scores = []
        
        for i in range(len(rlhf_output.policy_input_lengths)):
            input_len = rlhf_output.policy_input_lengths[i]
            full_sequence = rlhf_output.generated_tokens[i]
            
            # Query: original input prompt
            query_tensor = full_sequence[:input_len]
            # Response: newly generated tokens
            response_tensor = full_sequence[input_len:]
            
            # Move tensors to PPO device if needed
            ppo_device = self.device_manager.get_ppo_device()
            if ppo_device != str(query_tensor.device):
                query_tensor = query_tensor.to(ppo_device)
                response_tensor = response_tensor.to(ppo_device)
                combined_rewards_tensor = combined_rewards_tensor.to(ppo_device)
            
            queries.append(query_tensor)
            responses.append(response_tensor)
            scores.append(combined_rewards_tensor[i])
        
        # Single PPO training step for the batch
        try:
            train_stats = self.ppo_trainer.step(
                queries=queries,
                responses=responses, 
                scores=scores
            )
            return train_stats
        except Exception as ppo_error:
            logger.warning(f"PPO step failed: {ppo_error}")
            return None
    
    def train(self, num_epochs: Optional[int] = None):
        """
        Main training loop with comprehensive metrics logging
        """
        if self.eval_only:
            logger.info("Running in evaluation-only mode")
            return self.evaluate()
        
        if not TRL_AVAILABLE:
            raise RuntimeError("TRL is required for training. ")
        
        num_epochs = num_epochs or self.config.training.num_epochs
        logger.info(f"Starting RLHF training for {num_epochs} epochs")
        
        # Training metrics
        training_stats = {
            "epoch_combined_rewards": [],
            "epoch_reward_scores": [],
            "epoch_judge_scores": [], 
            "epoch_accuracies": [],
            "epoch_policy_losses": []
        }
        
        # Ensure models are in training mode at start
        try:
            self.ppo_model.train()
            logger.info("Models set to training mode")
        except Exception as e:
            logger.error(f"Failed to set models to training mode: {e}")
            raise RuntimeError(f"Model mode setting failed: {e}")
        
        for epoch in range(num_epochs):
            logger.info(f"=== Epoch {epoch + 1}/{num_epochs} ===")
            
            try:
                epoch_metrics = self._train_epoch()
            except Exception as e:
                logger.error(f"Error in epoch {epoch + 1}: {e}")
                # Continue with next epoch or break depending on severity
                continue
            
            # Log metrics
            training_stats["epoch_combined_rewards"].append(epoch_metrics["mean_combined_reward"])
            training_stats["epoch_reward_scores"].append(epoch_metrics["mean_reward_score"])
            training_stats["epoch_judge_scores"].append(epoch_metrics["mean_judge_score"])
            training_stats["epoch_accuracies"].append(epoch_metrics["mean_accuracy"])
            training_stats["epoch_policy_losses"].append(epoch_metrics["mean_policy_loss"])
            
            # Log epoch-level metrics to wandb
            if self.metrics_logger:
                epoch_wandb_metrics = {
                    'combined_reward': epoch_metrics["mean_combined_reward"],
                    'reward_score': epoch_metrics["mean_reward_score"],
                    'judge_score': epoch_metrics["mean_judge_score"],
                    'accuracy': epoch_metrics["mean_accuracy"],
                    'policy_loss': epoch_metrics["mean_policy_loss"],
                }
                self.metrics_logger.log_epoch_metrics(epoch_wandb_metrics, epoch + 1)
            
            # Evaluation
            if (epoch + 1) % self.config.training.eval_frequency == 0:
                logger.info("Running evaluation...")
                eval_metrics = self.evaluate()
                logger.info(f"Eval metrics: {eval_metrics}")
                
                # Log evaluation metrics to wandb
                if self.metrics_logger:
                    self.metrics_logger.log_evaluation_metrics(eval_metrics)
            
            # Save checkpoint
            if (epoch + 1) % self.config.training.save_frequency == 0:
                self._save_checkpoint(epoch + 1, training_stats)
        
        logger.info("Training completed!")
        return training_stats
    
    def _train_epoch(self):
        """Train for one epoch with clean metrics collection
        
        Metrics logged during training:
        1. accuracies (of policy)
        2. reward_scores (from the reward model)
        3. judge_scores
        4. combined_rewards 
        5. policy_losses
        6. judge deception metrics (computed by self.metrics_logger.compute_deception_metrics)
        """
        # Initialize epoch-level metric collectors
        accuracies = []
        reward_scores = []  # Will be computed from reward_scores_all
        judge_scores = []
        combined_rewards = []
        policy_losses = []
        
        # Set model to training mode
        try:
            self.ppo_model.train()
            logger.debug("PPO model set to training mode for epoch")
        except Exception as e:
            logger.error(f"Failed to set PPO model to training mode: {e}")
            raise RuntimeError(f"Training mode setting failed: {e}")
        
        # Check training sample limit
        num_train_samples = self.config.training.num_train_samples
        samples_processed = 0
        
        for batch_idx, batch in enumerate(self.train_dataloader):
            # Check sample limit
            if num_train_samples is not None:
                batch_size = len(batch.stories)
                if samples_processed >= num_train_samples:
                    logger.info(f"Reached training sample limit of {num_train_samples}. Stopping epoch.")
                    break                
                samples_processed += batch_size
            
            try:
                # Clear cache before each batch to prevent fragmentation
                torch.cuda.empty_cache()
                
                # Forward pass through pipeline
                rlhf_output = self.pipeline.forward_pass(
                    batch, 
                    max_new_tokens=self.config.policy_model.max_new_tokens // 4
                )
                
                # Compute combined reward tensor
                combined_rewards_tensor = self.pipeline.compute_combined_reward(rlhf_output)
                
                # Debug prints (following MinMax pattern)
                print("Combined rewards:")
                print(60 * "=")
                print(combined_rewards_tensor)
                print(60 * "=")
                
                # PPO training step
                train_stats = self._ppo_training_step(rlhf_output, combined_rewards_tensor)
                
                # Collect batch-level metrics 
                batch_accuracy = rlhf_output.ground_truth_correct.float().mean().item()
                batch_reward_score = rlhf_output.reward_scores.mean().item()
                batch_judge_score = rlhf_output.judge_scores.mean().item()
                batch_combined_reward = combined_rewards_tensor.mean().item()
                batch_policy_loss = train_stats.get("ppo/loss/total", 0.0) if train_stats else 0.0
                
                # Store epoch metrics
                accuracies.append(batch_accuracy)
                reward_scores.append(batch_reward_score)  # Add reward scores to epoch metrics
                judge_scores.append(batch_judge_score)
                combined_rewards.append(batch_combined_reward)
                policy_losses.append(batch_policy_loss)
                                
                # Log batch-level metrics (following MinMax pattern)
                if self.metrics_logger and batch_idx % self.config.evaluation.log_frequency == 0:
                    batch_metrics = {
                        'combined_reward': batch_combined_reward,
                        'judge_score': batch_judge_score,
                        'policy_accuracy': batch_accuracy,
                        'reward_score_mean': batch_reward_score,
                        'policy_loss': batch_policy_loss,
                    }
                    
                    # Compute judge deception metrics
                    judge_deception_metrics = self.metrics_logger.compute_deception_metrics(
                        rlhf_output.judge_scores, rlhf_output.ground_truth_correct
                    )
                    for key, value in judge_deception_metrics.items():
                        batch_metrics[f'judge_{key}'] = value
                    
                    # Log batch metrics
                    self.metrics_logger.log_training_batch_metrics(batch_metrics)
                
                if self.metrics_logger:
                    self.metrics_logger.global_train_step += 1
                
                # Log progress (use log_frequency consistently like MinMax)
                if batch_idx % self.config.evaluation.log_frequency == 0:
                    logger.info(f"Batch {batch_idx}: "
                              f"Combined_Reward={batch_combined_reward:.3f}, "
                              f"Policy_Loss={batch_policy_loss:.3f}, "
                              f"Judge={batch_judge_score:.3f}, "
                              f"Accuracy={batch_accuracy:.3f}")
                
            except Exception as e:
                logger.error(f"Error in batch {batch_idx}: {e}")
                continue
        
        return {
            "mean_combined_reward": float(np.mean(combined_rewards)) if combined_rewards else 0.0,
            "mean_reward_score": float(np.mean(reward_scores)) if reward_scores else 0.0,
            "mean_judge_score": float(np.mean(judge_scores)) if judge_scores else 0.0,
            "mean_accuracy": float(np.mean(accuracies)) if accuracies else 0.0,
            "mean_policy_loss": float(np.mean(policy_losses)) if policy_losses else 0.0
        }
    
    def evaluate(self):
        """Evaluate the model with comprehensive metrics collection"""
        logger.info("Running RLHF evaluation...")
        
        # Collect comprehensive evaluation data
        all_policy_correctness = []
        all_judge_scores = []
        all_reward_scores = []
        all_combined_scores = []
        all_parsed_answers = []
        batch_metrics_list = []
        
        # Set models to evaluation mode with proper error handling
        try:
            if hasattr(self, 'ppo_model'):
                self.ppo_model.eval()
                logger.debug("PPO model set to evaluation mode")
            
            # Also set underlying models to eval mode for generation
            self.policy_model.get_model_for_training().eval()
            self.reward_model.model.eval()
            self.judge_model.model.eval()
            logger.debug("All models set to evaluation mode")
        except Exception as e:
            logger.warning(f"Failed to set models to evaluation mode: {e}")
        
        # Evaluation loop with error handling
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_dataloader):
                if batch_idx >= self.config.evaluation.num_eval_samples // \
                   self.config.training.batch_size:
                    break
            
                try:
                    # Forward pass for evaluation (no training updates)
                    rlhf_output = self.pipeline.forward_pass(batch, max_new_tokens=50)
                    
                    # Compute combined rewards
                    combined_rewards = self.pipeline.compute_combined_reward(rlhf_output)
                    
                    # Collect data for comprehensive analysis
                    all_policy_correctness.extend(rlhf_output.ground_truth_correct.cpu().tolist())
                    all_judge_scores.extend(rlhf_output.judge_scores.cpu().tolist())
                    all_reward_scores.extend(rlhf_output.reward_scores.cpu().tolist())
                    all_combined_scores.extend(combined_rewards.cpu().tolist())
                    all_parsed_answers.extend(rlhf_output.parsed_answers)
                    
                    # Compute batch-level metrics for logging
                    batch_metrics = {
                        "accuracy": rlhf_output.ground_truth_correct.float().mean().item(),
                        "avg_reward_score": rlhf_output.reward_scores.mean().item(),
                        "avg_judge_score": rlhf_output.judge_scores.mean().item(),
                        "avg_combined_score": combined_rewards.mean().item(),
                        "parsing_success_rate": np.mean([1.0 if ans in ['A', 'B'] else 0.0 for ans in rlhf_output.parsed_answers]),
                        "batch_size": len(rlhf_output.parsed_answers),
                    }
                    batch_metrics_list.append(batch_metrics)
                    
                    logger.debug("Batch %d evaluation metrics: \n%s", batch_idx, json.dumps(batch_metrics, indent=2))
                    
                except Exception as e:
                    logger.error(f"Error in eval batch {batch_idx}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
        
        # Aggregate basic metrics
        if not batch_metrics_list:
            return {"error": "No successful evaluation batches"}
        
        # Convert to tensors for metrics computation
        policy_correctness_tensor = torch.tensor(all_policy_correctness, dtype=torch.float32)
        judge_scores_tensor = torch.tensor(all_judge_scores, dtype=torch.float32)
        
        # Compute comprehensive metrics using MetricsLogger
        aggregated_metrics = {
            "accuracy": float(policy_correctness_tensor.mean().item()),
            "avg_reward_score": float(np.mean(all_reward_scores)),
            "avg_judge_score": float(np.mean(all_judge_scores)),
            "avg_combined_score": float(np.mean(all_combined_scores)),
            "parsing_success_rate": np.mean([m["parsing_success_rate"] for m in batch_metrics_list]),
            "total_samples": sum([m["batch_size"] for m in batch_metrics_list]),
        }
        
        # Add judge deception metrics
        if self.metrics_logger:
            print("Judge Model: ")
            judge_deception_metrics = self.metrics_logger.compute_deception_metrics(
                judge_scores_tensor, policy_correctness_tensor
            )
            for key, value in judge_deception_metrics.items():
                aggregated_metrics[f"judge_{key}"] = value
            
            # Add answer distribution
            answer_distribution = self.metrics_logger.compute_answer_distribution(all_parsed_answers)
            aggregated_metrics.update(answer_distribution)

            # Log evaluation metrics
            self.metrics_logger.log_evaluation_metrics(aggregated_metrics)
        
        # Restore training mode after evaluation
        try:
            if hasattr(self, 'ppo_model'):
                self.ppo_model.train()
                logger.debug("PPO model restored to training mode")
            
            # Also restore underlying models to training mode  
            self.policy_model.get_model_for_training().train()
            self.reward_model.model.train()
            self.judge_model.model.train()
            logger.debug("All models restored to training mode")
        except Exception as e:
            logger.warning(f"Failed to restore models to training mode: {e}")
        
        return aggregated_metrics
    
    def _save_checkpoint(self, epoch: int, training_stats: Dict):
        """Save model checkpoint and training stats"""
        checkpoint_dir = os.path.join(self.save_dir, f"epoch_{epoch}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save model
        if hasattr(self, 'ppo_model'):
            self.ppo_model.save_pretrained(checkpoint_dir)
            self.policy_model.get_tokenizer().save_pretrained(checkpoint_dir)
        
        # Save training stats
        with open(os.path.join(checkpoint_dir, "training_stats.json"), 'w') as f:
            json.dump(training_stats, f, indent=2)
        
        # Save config
        with open(os.path.join(checkpoint_dir, "config.json"), 'w') as f:
            json.dump(self.config.to_dict(), f, indent=2)
        
        logger.info(f"Checkpoint saved to {checkpoint_dir}")

def main():
    parser = argparse.ArgumentParser(description="Dataset-Agnostic RLHF Training")
    parser.add_argument("--config", type=str, 
                       default="config/rlhf_config.json",
                       help="Path to configuration file (relative to project root)")
    parser.add_argument("--dataset", type=str, default="qa_simple",
                       choices=["qa_simple"],
                       help="Dataset to use for training")
    parser.add_argument("--eval-only", action="store_true",
                       help="Run evaluation only")
    parser.add_argument("--epochs", type=int, default=None,
                       help="Number of epochs to train")
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = RLHFTrainer(
        config_path=args.config,
        dataset_name=args.dataset,
        eval_only=args.eval_only
    )
    
    # Train or evaluate
    if args.eval_only:
        results = trainer.evaluate()
        print("Evaluation Results:")
        results = {k: (v.item() if hasattr(v, "item") else v)
                for k, v in results.items()}
        print(json.dumps(results, indent=2))
        # save evaluation results
        with open(os.path.join(trainer.save_dir, "evaluation_results.json"), 'w') as f:
            json.dump(results, f, indent=2)
    else:
        training_stats = trainer.train(num_epochs=args.epochs)
        print("Training completed!")
        print("Final training stats:")
        
        # Convert all values to JSON-serializable types
        final_stats = {}
        for k, v in training_stats.items():
            if v:  # If list is not empty
                final_val = v[-1] if isinstance(v, list) else v
                # Convert numpy types to Python native types
                if hasattr(final_val, 'item'):
                    final_stats[k] = final_val.item()
                elif isinstance(final_val, (np.float32, np.float64)):
                    final_stats[k] = float(final_val)
                elif isinstance(final_val, (np.int32, np.int64)):
                    final_stats[k] = int(final_val)
                else:
                    final_stats[k] = final_val
            else:
                final_stats[k] = 0.0
        
        print(json.dumps(final_stats, indent=2))

if __name__ == "__main__":
    main()

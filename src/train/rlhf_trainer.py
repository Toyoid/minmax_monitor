#!/usr/bin/env python3
"""
Dataset-Agnostic RLHF Training Script  
Main training pipeline using the new dataset-agnostic architecture
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

from src.config.model_config import RLHFConfig, create_model_configs_from_dict
from src.config.dataset_config import QASimpleConfig
from src.data.qa_processor import create_qa_simple_dataloader
from src.data.dataset_processor import DatasetProcessor, RawBatchData
from src.models.policy_model import PolicyModel
from src.models.reward_model import RewardModel
from src.models.judge_model import JudgeModel
from src.pipelines.rlhf_pipeline import RLHFPipeline

# TRL imports for PPO training
try:
    from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
    from trl.core import LengthSampler
    import wandb
    TRL_AVAILABLE = True
except ImportError:
    print("TRL not available. Please install with: pip install trl")
    TRL_AVAILABLE = False

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RLHFTrainer:
    """
    Dataset-agnostic RLHF trainer using the new pipeline architecture with multi-GPU support
    """
    
    def __init__(self, config_path: str, dataset_name: str = "qa_simple", eval_only: bool = False):
        # Setup GPU environment first
        from src.utils.env_setup import setup_gpu_environment, print_environment_info
        from src.utils.device_manager import create_device_manager
        
        available_gpus = setup_gpu_environment()
        logger.info(f"Initializing RLHF trainer with {available_gpus} available GPUs")
        
        self.config_path = config_path
        self.dataset_name = dataset_name
        self.eval_only = eval_only
        
        # Load configuration
        with open(config_path) as f:
            self.config_dict = json.load(f)
        
        # Initialize device manager (simplified)
        self.device_manager = create_device_manager()
        
        self.device = torch.device(self.device_manager.get_policy_device())
        self.save_dir = self.config_dict.get("save_dir", "./outputs")
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Initialize components
        self._setup_dataset_config()
        self._setup_models()
        self._setup_pipeline()
        self._setup_data()
        
        if not eval_only and TRL_AVAILABLE:
            self._setup_ppo_trainer()
        
        logger.info("Dataset-agnostic RLHF trainer initialized successfully")
    
    def _setup_dataset_config(self):
        """Setup dataset-specific configuration"""
        if self.dataset_name == "qa_simple":
            self.dataset_config = QASimpleConfig()
            self.data_dir = self.config_dict["data"]["data_dir"]
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")
        
        logger.info(f"Dataset config: {self.dataset_name}")
    
    def _setup_models(self):
        """Initialize all models with distributed device placement"""
        logger.info("Loading models with distributed device placement...")
        
        # Create model configs
        policy_config, reward_config, judge_config = create_model_configs_from_dict(self.config_dict)
        
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
        self.device_manager.print_memory_usage()
    
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
            self.device_manager  # Add device manager
        )
        
        logger.info("RLHF pipeline initialized")
    
    def _setup_data(self):
        """Setup data loaders"""
        logger.info("Setting up data loaders...")
        
        batch_size = self.config_dict["training"]["batch_size"]
        
        # Training data
        self.train_dataloader = create_qa_simple_dataloader(
            self.data_dir, "train", batch_size=batch_size, shuffle=True
        )
        
        # Validation data
        self.val_dataloader = create_qa_simple_dataloader(
            self.data_dir, "val", batch_size=batch_size, shuffle=False
        )
        
        logger.info(f"Data loaders created - Train: {len(self.train_dataloader.dataset)}, "
                   f"Val: {len(self.val_dataloader.dataset)} samples")
        
        # Log training sample limit if set
        num_train_samples = self.config_dict["training"].get("num_train_samples", None)
        if num_train_samples is not None:
            logger.info(f"Training sample limit set to: {num_train_samples}")
        else:
            logger.info("No training sample limit set - will use full dataset")
    
    def _setup_ppo_trainer(self):
        """Setup PPO trainer for RLHF"""
        logger.info("Setting up PPO trainer...")
        
        # PPO Configuration with memory optimization
        ppo_config = PPOConfig(
            model_name=self.config_dict["policy_model"]["model_name"],
            learning_rate=self.config_dict["training"]["learning_rate"],
            batch_size=self.config_dict["training"]["mini_batch_size"],
            mini_batch_size=self.config_dict["training"]["mini_batch_size"],
            ppo_epochs=self.config_dict["training"]["ppo_epochs"],
            init_kl_coef=self.config_dict["training"]["init_kl_coef"],
            target_kl=self.config_dict["training"]["target_kl"],
            cliprange=self.config_dict["training"]["cliprange"],
            cliprange_value=self.config_dict["training"]["cliprange_value"],
            vf_coef=self.config_dict["training"]["vf_coef"],
            max_grad_norm=self.config_dict["training"]["max_grad_norm"],
            # Memory optimization settings
            gradient_checkpointing=True,  # Reduce memory at cost of compute
            optimize_cuda_cache=True,
        )
        
        # Get PPO device (may be different from policy device)
        ppo_device = self.device_manager.get_ppo_device()
        logger.info(f"PPO trainer will use device: {ppo_device}")
        
        # Wrap policy model for PPO
        self.ppo_model = AutoModelForCausalLMWithValueHead.from_pretrained(
            self.policy_model.get_model_for_training()
        )
        
        # Move PPO model to designated device if different from policy
        if ppo_device != self.device_manager.get_policy_device():
            logger.info(f"Moving PPO model from {self.device_manager.get_policy_device()} to {ppo_device}")
            self.ppo_model = self.ppo_model.to(ppo_device)
        
        # Initialize PPO trainer
        self.ppo_trainer = PPOTrainer(
            config=ppo_config,
            model=self.ppo_model,
            tokenizer=self.policy_model.get_tokenizer(),
        )
        
        logger.info("PPO trainer initialized")
    
    def train(self, num_epochs: Optional[int] = None):
        """
        Main training loop
        """
        if self.eval_only:
            logger.info("Running in evaluation-only mode")
            return self.evaluate()
        
        if not TRL_AVAILABLE:
            raise RuntimeError("TRL is required for training. Install with: pip install trl")
        
        num_epochs = num_epochs or self.config_dict["training"]["num_epochs"]
        logger.info(f"Starting RLHF training for {num_epochs} epochs")
        
        # Training metrics
        training_stats = {
            "epoch_rewards": [],
            "epoch_judge_scores": [],
            "epoch_correctness": [],
            "epoch_losses": []
        }
        
        for epoch in range(num_epochs):
            logger.info(f"=== Epoch {epoch + 1}/{num_epochs} ===")
            
            epoch_metrics = self._train_epoch()
            
            # Log metrics
            training_stats["epoch_rewards"].append(epoch_metrics["mean_reward"])
            training_stats["epoch_judge_scores"].append(epoch_metrics["mean_judge_score"])
            training_stats["epoch_correctness"].append(epoch_metrics["mean_correctness"])
            training_stats["epoch_losses"].append(epoch_metrics["mean_loss"])
            
            # Evaluation
            if (epoch + 1) % self.config_dict["training"]["eval_frequency"] == 0:
                logger.info("Running evaluation...")
                eval_metrics = self.evaluate()
                logger.info(f"Eval metrics: {eval_metrics}")
            
            # Save checkpoint
            if (epoch + 1) % self.config_dict["training"]["save_frequency"] == 0:
                self._save_checkpoint(epoch + 1, training_stats)
        
        logger.info("Training completed!")
        return training_stats
    
    def _train_epoch(self):
        """Train for one epoch"""
        total_rewards = []
        total_judge_scores = []
        total_correctness = []
        total_losses = []
        
        self.ppo_model.train()
        
        # Check if we should limit the number of training samples
        num_train_samples = self.config_dict["training"].get("num_train_samples", None)
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
                
                # Compute combined reward
                combined_rewards = self._compute_combined_reward(rlhf_output)
                
                # Split large batch into mini-batches for PPO training
                mini_batch_size = self.config_dict["training"]["mini_batch_size"]
                batch_size = len(rlhf_output.policy_input_lengths)
                
                # Process in mini-batches
                for start_idx in range(0, batch_size, mini_batch_size):
                    end_idx = min(start_idx + mini_batch_size, batch_size)
                    
                    # Extract mini-batch data
                    mini_query_list = []
                    mini_response_list = []
                    mini_scores_list = []
                    
                    for i in range(start_idx, end_idx):
                        input_len = rlhf_output.policy_input_lengths[i]
                        full_sequence = rlhf_output.generated_tokens[i]
                        
                        # Query: original input prompt
                        query_tensor = full_sequence[:input_len]
                        # Response: newly generated tokens
                        response_tensor = full_sequence[input_len:]
                        
                        # Move tensors to PPO device if different from policy device
                        ppo_device = self.device_manager.get_ppo_device()
                        if ppo_device != str(query_tensor.device):
                            query_tensor = query_tensor.to(ppo_device)
                            response_tensor = response_tensor.to(ppo_device)
                        
                        mini_query_list.append(query_tensor)
                        mini_response_list.append(response_tensor)
                        mini_scores_list.append(torch.tensor(combined_rewards[i].item(), dtype=torch.float32, device=ppo_device))
                    
                    try:
                        # PPO training step on mini-batch
                        train_stats = self.ppo_trainer.step(
                            queries=mini_query_list,
                            responses=mini_response_list, 
                            scores=mini_scores_list
                        )
                        
                        if train_stats:
                            total_losses.append(train_stats.get("ppo/loss/total", 0))
                    except Exception as ppo_error:
                        logger.warning(f"PPO step failed for mini-batch {start_idx}-{end_idx}: {ppo_error}")
                        continue
                
                # Collect metrics (after all mini-batches)
                total_rewards.extend(combined_rewards.cpu().numpy())
                total_judge_scores.extend(rlhf_output.judge_scores.cpu().numpy())
                total_correctness.extend(rlhf_output.ground_truth_correct.cpu().numpy())
                
                # Note: total_losses is already collected in the mini-batch loop above
                
                # Log progress
                if batch_idx % self.config_dict["evaluation"]["log_frequency"] == 0:
                    logger.info(f"Batch {batch_idx}: "
                              f"Reward={np.mean(combined_rewards.cpu().numpy()):.3f}, "
                              f"Judge={np.mean(rlhf_output.judge_scores.cpu().numpy()):.3f}, "
                              f"Correct={np.mean(rlhf_output.ground_truth_correct.cpu().numpy()):.3f}")
                
            except Exception as e:
                logger.error(f"Error in batch {batch_idx}: {e}")
                continue
        
        return {
            "mean_reward": float(np.mean(total_rewards)) if total_rewards else 0.0,
            "mean_judge_score": float(np.mean(total_judge_scores)) if total_judge_scores else 0.0,
            "mean_correctness": float(np.mean(total_correctness)) if total_correctness else 0.0,
            "mean_loss": float(np.mean(total_losses)) if total_losses else 0.0
        }
    
    def _compute_combined_reward(self, rlhf_output):
        """Compute combined reward from multiple sources"""
        # Get weights from config
        reward_weight = self.config_dict["training"]["reward_weight"]
        judge_weight = self.config_dict["training"]["judge_weight"]
        
        # Normalize scores to [-1, 1] range
        # Reward scores: use tanh to map raw logits to [-1, 1]
        normalized_reward = torch.tanh(rlhf_output.reward_scores)
        
        # Judge scores: already in [-1, 1] range
        normalized_judge = rlhf_output.judge_scores
        
        # Combine rewards
        combined_reward = (reward_weight * normalized_reward + 
                          judge_weight * normalized_judge)
        
        return combined_reward
    
    def evaluate(self):
        """Evaluate the model"""
        logger.info("Running evaluation...")
        
        total_rewards = []
        total_judge_scores = []
        parsing_success_cnt = 0
        judge_correct_cnt = 0
        judge_positive_cnt = 0
        judge_false_pos_cnt = 0
        judge_false_neg_cnt = 0
        correct_predictions = 0
        total_predictions = 0
        
        self.ppo_model.eval() if hasattr(self, 'ppo_model') else None
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_dataloader):
                if batch_idx >= self.config_dict["evaluation"]["num_eval_samples"] // \
                   self.config_dict["training"]["batch_size"]:
                    break
                
                try:
                    # Forward pass
                    # rlhf_output = self.pipeline.forward_pass(batch, max_new_tokens=50)
                    batch_metrics = self.pipeline.evaluate_batch(batch, max_new_tokens=50)
                    
                    # Collect metrics
                    total_rewards.extend(batch_metrics['reward_scores'])
                    total_judge_scores.extend(batch_metrics['judge_scores'])
                    
                    # Count correct predictions
                    parsing_success_cnt += batch_metrics['parsing_success_rate'] * batch_metrics['batch_size']
                    judge_correct_cnt += batch_metrics['judge_accuracy'] * batch_metrics['batch_size']
                    judge_positive_cnt += batch_metrics['judge_positive_cnt']
                    judge_false_pos_cnt += batch_metrics['judge_false_pos_cnt']
                    judge_false_neg_cnt += batch_metrics['judge_false_neg_cnt']

                    correct_predictions += sum(batch_metrics['ground_truth_correct'])
                    total_predictions += batch_metrics['batch_size']

                except Exception as e:
                    logger.error(f"Error in eval batch {batch_idx}: {e}")
                    continue
        
        # Compute metrics
        eval_metrics = {
            "mean_reward": np.mean(total_rewards) if total_rewards else 0,
            "mean_judge_score": np.mean(total_judge_scores) if total_judge_scores else 0,
            "accuracy": correct_predictions / total_predictions if total_predictions > 0 else 0,
            "parsing_success_rate": parsing_success_cnt / total_predictions if total_predictions > 0 else 0,
            "judge_accuracy": judge_correct_cnt / total_predictions if total_predictions > 0 else 0,
            "judge_positive_rate": judge_positive_cnt / total_predictions if total_predictions > 0 else 0,
            "judge_false_pos_rate": judge_false_pos_cnt / total_predictions if total_predictions > 0 else 0,
            "judge_false_neg_rate": judge_false_neg_cnt / total_predictions if total_predictions > 0 else 0,
            "total_samples": total_predictions
        }
        
        return eval_metrics
    
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
            json.dump(self.config_dict, f, indent=2)
        
        logger.info(f"Checkpoint saved to {checkpoint_dir}")

def main():
    parser = argparse.ArgumentParser(description="Dataset-Agnostic RLHF Training")
    parser.add_argument("--config", type=str, 
                       default="/data/lhx/minmax_monitor/config/rlhf_config.json",
                       help="Path to configuration file")
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

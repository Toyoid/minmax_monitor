"""
MinMax Trainer for LLM-vs-Monitor Training
Main coordinator that orchestrates TTUR training between policy and monitor models
"""
import os
import sys
import torch
import numpy as np
import logging
import json
import argparse
from typing import Dict, Any, List, Optional
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from ..config.model_config import MinMaxConfig
from ..config.dataset_config import QASimpleConfig
from ..data.qa_processor import create_qa_simple_dataloader
from ..data.dataset_processor import DatasetProcessor
from ..models.policy_model import PolicyModel
from ..models.monitor_model import MonitorModel
from ..models.reward_model import RewardModel
from ..models.judge_model import JudgeModel
from ..pipelines.minmax_pipeline import MinMaxPipeline
from ..train.dual_ppo_optimizer import DualPPOMinMaxOptimizer

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MinMaxTrainer:
    """
    Main coordinator for MinMax LLM-vs-Monitor training
    Implements TTUR (Two-Timescale Update Rule) training dynamics
    """
    
    def __init__(self, config_path: str, dataset_name: str = "qa_simple", eval_only: bool = False):
        """
        Initialize MinMax trainer
        
        Args:
            config_path: Path to configuration file
            dataset_name: Dataset to use for training
            eval_only: Whether to run evaluation only
        """
        # Setup GPU environment first
        from ..utils.env_setup import setup_gpu_environment
        from ..utils.device_manager import create_device_manager
        
        available_gpus = setup_gpu_environment()
        logger.info(f"Initializing MinMax trainer with {available_gpus} available GPUs")
        
        self.config_path = config_path
        self.dataset_name = dataset_name
        self.eval_only = eval_only
        
        # Load configuration
        with open(config_path) as f:
            config_dict = json.load(f)
        
        self.config = MinMaxConfig.from_dict(config_dict)
        
        # Initialize device manager (simplified)
        self.device_manager = create_device_manager()
        
        self.device = torch.device(self.device_manager.get_policy_device())
        self.save_dir = self.config.save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Initialize components
        self._setup_dataset_config()
        self._setup_models()
        self._setup_pipeline()
        self._setup_data()
        
        if not eval_only:
            self._setup_optimizer()
        
        logger.info("MinMax trainer initialized successfully")
    
    def _setup_dataset_config(self):
        """Setup dataset-specific configuration"""
        if self.dataset_name == "qa_simple":
            self.dataset_config = QASimpleConfig()
            self.data_dir = self.config.data.data_dir
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")
        
        logger.info(f"Dataset config: {self.dataset_name}")
    
    def _setup_models(self):
        """Initialize all models with distributed device placement"""
        logger.info("Loading models with distributed device placement...")
        
        # Create model configs
        policy_config = self.config.policy_model
        monitor_config = self.config.monitor_model
        reward_config = self.config.reward_model
        judge_config = self.config.judge_model
        
        # Initialize models on allocated devices
        self.policy_model = PolicyModel(
            policy_config, 
            device=self.device_manager.get_policy_device()
        )
        
        self.monitor_model = MonitorModel(
            monitor_config,
            device=self.device_manager.get_monitor_device()
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
        """Setup the MinMax pipeline"""
        logger.info("Setting up MinMax pipeline...")
        
        # Create dataset processor
        self.processor = DatasetProcessor(
            self.dataset_config, 
            self.config.policy_model, 
            self.config.monitor_model, 
            "minmax"
        )
        
        # Initialize MinMax pipeline
        self.pipeline = MinMaxPipeline(
            self.policy_model,
            self.monitor_model,
            self.reward_model, 
            self.judge_model,
            self.processor,
            self.device_manager
        )
        
        logger.info("MinMax pipeline initialized")
    
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
    
    def _setup_optimizer(self):
        """Setup Dual PPO MinMax optimizer with TTUR"""
        logger.info("Setting up Dual PPO MinMax optimizer...")
        
        self.optimizer = DualPPOMinMaxOptimizer(
            self.policy_model,
            self.monitor_model,
            self.config.training,
            self.device_manager
        )
        
        logger.info("Dual PPO MinMax optimizer initialized")
    
    def train(self, num_epochs: Optional[int] = None) -> Dict[str, Any]:
        """
        Main training loop with TTUR dynamics
        
        Args:
            num_epochs: Number of epochs to train
            
        Returns:
            Training statistics
        """
        if self.eval_only:
            logger.info("Running in evaluation-only mode")
            return self.evaluate()
        
        num_epochs = num_epochs or self.config.training.num_epochs
        logger.info(f"Starting MinMax training for {num_epochs} epochs")
        
        # Training metrics
        training_stats = {
            "epoch_policy_losses": [],
            "epoch_monitor_losses": [],
            "epoch_combined_rewards": [],
            "epoch_truthfulness_penalties": [],
            "epoch_constraint_lambdas": [],
            "epoch_accuracies": []
        }
        
        for epoch in range(num_epochs):
            logger.info(f"=== Epoch {epoch + 1}/{num_epochs} ===")
            
            epoch_metrics = self._train_epoch()
            
            # Log metrics
            training_stats["epoch_policy_losses"].append(epoch_metrics["mean_policy_loss"])
            training_stats["epoch_monitor_losses"].append(epoch_metrics["mean_monitor_loss"])
            training_stats["epoch_combined_rewards"].append(epoch_metrics["mean_combined_reward"])
            training_stats["epoch_truthfulness_penalties"].append(epoch_metrics["mean_truthfulness_penalty"])
            training_stats["epoch_constraint_lambdas"].append(epoch_metrics["mean_lambda"])
            training_stats["epoch_accuracies"].append(epoch_metrics["mean_accuracy"])
            
            # Evaluation
            if (epoch + 1) % self.config.training.eval_frequency == 0:
                logger.info("Running evaluation...")
                eval_metrics = self.evaluate()
                logger.info(f"Eval accuracy: {eval_metrics['accuracy']:.3f}")
            
            # Save checkpoint
            if (epoch + 1) % self.config.training.save_frequency == 0:
                self._save_checkpoint(epoch + 1, training_stats)
        
        logger.info("MinMax training completed!")
        return training_stats
    
    def _train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch using TTUR dynamics
        
        Returns:
            Dictionary with epoch metrics
        """
        policy_losses = []
        monitor_losses = []
        combined_rewards = []
        truthfulness_penalties = []
        constraint_lambdas = []
        accuracies = []
        
        # Set models to training mode
        self.policy_model.get_model_for_training().train()
        self.monitor_model.get_model_for_training().train()
        
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
                # Clear cache before each batch
                torch.cuda.empty_cache()
                
                # Forward pass through pipeline
                minmax_output = self.pipeline.forward_pass(
                    batch, 
                    max_new_tokens=self.config.policy_model.max_new_tokens // 4
                )
                
                # TTUR: Single step handles both monitor (inner) and policy (outer) updates
                # Monitor updated more frequently than policy automatically
                all_metrics = self.optimizer.step(minmax_output)
                
                # Extract metrics for logging
                policy_losses.append(all_metrics["policy_loss"])
                monitor_losses.append(all_metrics["monitor_loss"])
                combined_rewards.append(all_metrics["policy_reward"])
                truthfulness_penalties.append(all_metrics["avg_violation"])
                constraint_lambdas.append(all_metrics["lambda"])
                accuracies.append(minmax_output.ground_truth_correct.float().mean().item())
                
                # Log progress
                if batch_idx % self.config.evaluation.log_frequency == 0:
                    logger.info(f"Batch {batch_idx}: "
                              f"Combined_Reward={combined_rewards[-1]:.3f}, "
                              f"Policy_Loss={policy_losses[-1]:.3f}, "
                              f"Monitor_Loss={monitor_losses[-1]:.3f}, "
                              f"Accuracy={accuracies[-1]:.3f}, "
                              f"λ={constraint_lambdas[-1]:.3f}")
                
            except Exception as e:
                logger.error(f"Error in batch {batch_idx}: {e}")
                continue
        
        return {
            "mean_policy_loss": float(np.mean(policy_losses)) if policy_losses else 0.0,
            "mean_monitor_loss": float(np.mean(monitor_losses)) if monitor_losses else 0.0,
            "mean_combined_reward": float(np.mean(combined_rewards)) if combined_rewards else 0.0,
            "mean_truthfulness_penalty": float(np.mean(truthfulness_penalties)) if truthfulness_penalties else 0.0,
            "mean_lambda": float(np.mean(constraint_lambdas)) if constraint_lambdas else 0.0,
            "mean_accuracy": float(np.mean(accuracies)) if accuracies else 0.0,
        }
    
    def evaluate(self) -> Dict[str, Any]:
        """Evaluate the MinMax models"""
        logger.info("Running MinMax evaluation...")
        
        # Set models to evaluation mode
        self.policy_model.get_model_for_training().eval()
        self.monitor_model.get_model_for_training().eval()
        
        all_metrics = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_dataloader):
                if batch_idx >= self.config.evaluation.num_eval_samples // \
                   self.config.training.batch_size:
                    break
                
                try:
                    # Evaluate batch
                    batch_metrics = self.pipeline.evaluate_batch(
                        batch, 
                        max_new_tokens=self.config.policy_model.max_new_tokens // 4
                    )
                    all_metrics.append(batch_metrics)
                    logger.info("Batch %d evaluation metrics: \n%s", batch_idx, json.dumps(batch_metrics, indent=2))
                    
                except Exception as e:
                    logger.error(f"Error in eval batch {batch_idx}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
        
        # Aggregate metrics
        if not all_metrics:
            return {"error": "No successful evaluation batches"}
        
        aggregated_metrics = {
            "accuracy": np.mean([m["accuracy"] for m in all_metrics]),
            "avg_reward_score": np.mean([m["avg_reward_score"] for m in all_metrics]),
            "avg_judge_score": np.mean([m["avg_judge_score"] for m in all_metrics]),
            "avg_combined_score": np.mean([m["avg_combined_score"] for m in all_metrics]),
            "avg_truthfulness_penalty": np.mean([m["avg_truthfulness_penalty"] for m in all_metrics]),
            "avg_critique_length": np.mean([m["avg_critique_length"] for m in all_metrics]),
            "parsing_success_rate": np.mean([m["parsing_success_rate"] for m in all_metrics]),
            "judge_accuracy": np.mean([m["judge_accuracy"] for m in all_metrics]),
            "total_samples": sum([m["batch_size"] for m in all_metrics]),
        }
        
        return aggregated_metrics
    
    def _save_checkpoint(self, epoch: int, training_stats: Dict):
        """Save model checkpoint and training stats"""
        checkpoint_dir = os.path.join(self.save_dir, f"epoch_{epoch}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save models
        self.policy_model.save_model(os.path.join(checkpoint_dir, "policy_model"))
        self.monitor_model.save_model(os.path.join(checkpoint_dir, "monitor_model"))
        
        # Save training stats
        with open(os.path.join(checkpoint_dir, "training_stats.json"), 'w') as f:
            json.dump(training_stats, f, indent=2)
        
        # Save config
        with open(os.path.join(checkpoint_dir, "config.json"), 'w') as f:
            json.dump(self.config.to_dict(), f, indent=2)
        
        # Save optimizer state
        optimizer_state = self.optimizer.get_training_state()
        torch.save(optimizer_state, os.path.join(checkpoint_dir, "optimizer_state.pt"))
        
        logger.info(f"MinMax checkpoint saved to {checkpoint_dir}")


def main():
    parser = argparse.ArgumentParser(description="MinMax LLM-vs-Monitor Training")
    parser.add_argument("--config", type=str, 
                       default="/data/lhx/minmax_monitor/config/minmax_config.json",
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
    trainer = MinMaxTrainer(
        config_path=args.config,
        dataset_name=args.dataset,
        eval_only=args.eval_only
    )
    
    # Train or evaluate
    if args.eval_only:
        results = trainer.evaluate()
        print("MinMax Evaluation Results:")
        results = {k: (v.item() if hasattr(v, "item") else v)
                  for k, v in results.items()}
        print(json.dumps(results, indent=2))
    else:
        training_stats = trainer.train(num_epochs=args.epochs)
        print("MinMax training completed!")
        print("Final training stats:")
        
        # Convert all values to JSON-serializable types
        final_stats = {}
        for k, v in training_stats.items():
            if v:  # If list is not empty
                final_val = v[-1] if isinstance(v, list) else v
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

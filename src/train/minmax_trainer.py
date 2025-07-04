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
from ..utils.metrics_logger import MetricsLogger

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
        
        # Initialize MetricsLogger
        self._setup_metrics_logger()
        
        # Initialize components
        self._setup_dataset_config()
        self._setup_models()
        self._setup_pipeline()
        self._setup_data()
        
        if not eval_only:
            self._setup_optimizer()
        
        logger.info("MinMax trainer initialized successfully")
    
    def _setup_metrics_logger(self):
        """Setup MetricsLogger for wandb integration"""
        # Extract wandb config from training config if available
        wandb_config = getattr(self.config, 'wandb', {})
        
        # Create run name with timestamp
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"minmax_{self.dataset_name}_{timestamp}"
        
        # Get project name from config or use default
        project_name = wandb_config.get('project', 'minmax-rlhf-training')
        
        # Initialize MetricsLogger
        try:
            self.metrics_logger = MetricsLogger(
                project_name=project_name,
                run_name=run_name,
                config=self.config.to_dict(),
                trainer_type="minmax",
                wandb_config=wandb_config
            )
            logger.info("MetricsLogger initialized for MinMax training")
        except Exception as e:
            logger.warning(f"Failed to initialize MetricsLogger: {e}. Training will continue without wandb logging.")
            self.metrics_logger = None
    
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
            
            # Log epoch-level metrics to wandb
            if self.metrics_logger:
                epoch_wandb_metrics = {
                    'policy_loss': epoch_metrics["mean_policy_loss"],
                    'monitor_loss': epoch_metrics["mean_monitor_loss"],
                    'combined_reward': epoch_metrics["mean_combined_reward"],
                    'truthfulness_penalty': epoch_metrics["mean_truthfulness_penalty"],
                    'lambda': epoch_metrics["mean_lambda"],
                    'accuracy': epoch_metrics["mean_accuracy"],
                }
                self.metrics_logger.log_epoch_metrics(epoch_wandb_metrics, epoch + 1)
            
            # Evaluation
            if (epoch + 1) % self.config.training.eval_frequency == 0:
                logger.info("Running evaluation...")
                eval_metrics = self.evaluate()
                logger.info(f"Eval accuracy: {eval_metrics['accuracy']:.3f}")
                
                # Log evaluation metrics to wandb
                if self.metrics_logger:
                    self.metrics_logger.log_evaluation_metrics(eval_metrics)
            
            # Save checkpoint
            if (epoch + 1) % self.config.training.save_frequency == 0:
                self._save_checkpoint(epoch + 1, training_stats)
        
        logger.info("MinMax training completed!")
        return training_stats
    
    def _train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch using TTUR dynamics with comprehensive metrics logging
        
        Returns:
            Dictionary with epoch metrics
        """
        # Initialize metric collectors
        policy_losses = []
        monitor_losses = []
        combined_rewards = []
        truthfulness_penalties = []
        constraint_lambdas = []
        accuracies = []
        
        # Additional metrics for comprehensive logging
        reward_scores_all = []
        judge_scores_all = []
        policy_correctness_all = []
        judge_correctness_all = []
        monitor_correctness_all = []
        parsed_answers_all = []
        
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
            
            # Clear cache before each batch
            torch.cuda.empty_cache()
            
            # Forward pass through pipeline
            minmax_output = self.pipeline.forward_pass(
                batch, 
                max_new_tokens=self.config.policy_model.max_new_tokens // 4
            )
            
            # Parse monitor judgments for additional metrics
            monitor_judgments = self.processor.parse_monitor_judgments_batch(minmax_output.monitor_critiques)
            monitor_correctness = self.processor.compute_monitor_correctness_batch(
                monitor_judgments, 
                minmax_output.ground_truth_correct
            )
            
            # TTUR: Single step handles both monitor (inner) and policy (outer) updates
            all_metrics = self.optimizer.step(minmax_output)
            
            # Collect basic training metrics
            policy_losses.append(all_metrics["policy_loss"])
            monitor_losses.append(all_metrics["monitor_loss"])
            combined_rewards.append(all_metrics["policy_reward"])
            truthfulness_penalties.append(all_metrics["avg_violation"])
            constraint_lambdas.append(all_metrics["lambda"])
            accuracies.append(minmax_output.ground_truth_correct.float().mean().item())
            
            # Collect additional metrics for comprehensive analysis
            reward_scores_all.extend(minmax_output.reward_scores.cpu().tolist())
            judge_scores_all.extend(minmax_output.judge_scores.cpu().tolist())
            policy_correctness_all.extend(minmax_output.ground_truth_correct.cpu().tolist())
            parsed_answers_all.extend(minmax_output.parsed_answers)
            
            # Judge correctness: binary threshold on judge scores
            judge_binary = (minmax_output.judge_scores > 0.5).float()
            judge_correctness_batch = (judge_binary == minmax_output.ground_truth_correct).float()
            judge_correctness_all.extend(judge_correctness_batch.cpu().tolist())
            
            # Monitor correctness
            monitor_correctness_all.extend(monitor_correctness.cpu().tolist())
            
            # Log comprehensive metrics to wandb every log_frequency batches
            if self.metrics_logger and batch_idx % self.config.evaluation.log_frequency == 0:
                # Compute batch-level metrics
                batch_metrics = {
                    'policy_loss': all_metrics["policy_loss"],
                    'monitor_loss': all_metrics["monitor_loss"],
                    'combined_reward': all_metrics["policy_reward"],
                    'truthfulness_penalty': all_metrics["avg_violation"],
                    'lambda': all_metrics["lambda"],
                    'policy_accuracy': minmax_output.ground_truth_correct.float().mean().item(),
                    'reward_score_mean': minmax_output.reward_scores.mean().item(),
                    'judge_score_mean': minmax_output.judge_scores.mean().item(),
                }
                
                # Compute deception metrics for judge
                judge_deception_metrics = self.metrics_logger.compute_deception_metrics(
                    minmax_output.judge_scores, minmax_output.ground_truth_correct
                )
                for key, value in judge_deception_metrics.items():
                    batch_metrics[f'judge_{key}'] = value
                
                # Compute deception metrics for monitor
                monitor_deception_metrics = self.metrics_logger.compute_deception_metrics(
                    monitor_correctness, minmax_output.ground_truth_correct
                )
                for key, value in monitor_deception_metrics.items():
                    batch_metrics[f'monitor_{key}'] = value
                
                # Log batch metrics
                self.metrics_logger.log_training_batch_metrics(batch_metrics)
            
            self.metrics_logger.global_train_step += 1
            
            # Log progress
            if batch_idx % self.config.evaluation.log_frequency == 0:
                logger.info(f"Batch {batch_idx}: "
                            f"Combined_Reward={combined_rewards[-1]:.3f}, "
                            f"Policy_Loss={policy_losses[-1]:.3f}, "
                            f"Monitor_Loss={monitor_losses[-1]:.3f}, "
                            f"Accuracy={accuracies[-1]:.3f}, "
                            f"λ={constraint_lambdas[-1]:.3f}")
                        
        return {
            "mean_policy_loss": float(np.mean(policy_losses)) if policy_losses else 0.0,
            "mean_monitor_loss": float(np.mean(monitor_losses)) if monitor_losses else 0.0,
            "mean_combined_reward": float(np.mean(combined_rewards)) if combined_rewards else 0.0,
            "mean_truthfulness_penalty": float(np.mean(truthfulness_penalties)) if truthfulness_penalties else 0.0,
            "mean_lambda": float(np.mean(constraint_lambdas)) if constraint_lambdas else 0.0,
            "mean_accuracy": float(np.mean(accuracies)) if accuracies else 0.0,
        }
    
    def evaluate(self) -> Dict[str, Any]:
        """Evaluate the MinMax models with comprehensive metrics collection"""
        logger.info("Running MinMax evaluation...")
        
        # Set models to evaluation mode
        self.policy_model.get_model_for_training().eval()
        self.monitor_model.get_model_for_training().eval()
        
        # Collect comprehensive evaluation data
        all_policy_correctness = []
        all_judge_scores = []
        all_monitor_correctness = []
        all_reward_scores = []
        all_combined_scores = []
        all_truthfulness_penalties = []
        all_parsed_answers = []
        batch_metrics_list = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_dataloader):
                if batch_idx >= self.config.evaluation.num_eval_samples // \
                   self.config.training.batch_size:
                    break
                
                try:
                    # Forward pass for evaluation (no training updates)
                    minmax_output = self.pipeline.forward_pass(
                        batch, 
                        max_new_tokens=self.config.policy_model.max_new_tokens // 4
                    )
                    
                    # Parse monitor judgments
                    monitor_judgments = self.processor.parse_monitor_judgments_batch(minmax_output.monitor_critiques)
                    monitor_correctness = self.processor.compute_monitor_correctness_batch(
                        monitor_judgments, 
                        minmax_output.ground_truth_correct
                    )
                    
                    # Collect data for comprehensive analysis
                    all_policy_correctness.extend(minmax_output.ground_truth_correct.cpu().tolist())
                    all_judge_scores.extend(minmax_output.judge_scores.cpu().tolist())
                    all_monitor_correctness.extend(monitor_correctness.cpu().tolist())
                    all_reward_scores.extend(minmax_output.reward_scores.cpu().tolist())
                    all_combined_scores.extend(minmax_output.combined_scores.cpu().tolist())
                    all_truthfulness_penalties.extend(minmax_output.truthfulness_penalty.cpu().tolist())
                    all_parsed_answers.extend(minmax_output.parsed_answers)
                    
                    # Compute batch-level metrics for logging
                    batch_metrics = {
                        "accuracy": minmax_output.ground_truth_correct.float().mean().item(),
                        "avg_reward_score": minmax_output.reward_scores.mean().item(),
                        "avg_judge_score": minmax_output.judge_scores.mean().item(),
                        "avg_combined_score": minmax_output.combined_scores.mean().item(),
                        "avg_truthfulness_penalty": minmax_output.truthfulness_penalty.mean().item(),
                        "avg_critique_length": np.mean([len(critique.split()) for critique in minmax_output.monitor_critiques]),
                        "parsing_success_rate": np.mean([1.0 if ans in ['A', 'B'] else 0.0 for ans in minmax_output.parsed_answers]),
                        "batch_size": len(minmax_output.parsed_answers),
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
        monitor_correctness_tensor = torch.tensor(all_monitor_correctness, dtype=torch.float32)
        
        # Compute comprehensive metrics using MetricsLogger
        aggregated_metrics = {
            "accuracy": float(policy_correctness_tensor.mean().item()),
            "avg_reward_score": float(np.mean(all_reward_scores)),
            "avg_judge_score": float(np.mean(all_judge_scores)),
            "avg_combined_score": float(np.mean(all_combined_scores)),
            "avg_truthfulness_penalty": float(np.mean(all_truthfulness_penalties)),
            "avg_critique_length": np.mean([m["avg_critique_length"] for m in batch_metrics_list]),
            "parsing_success_rate": np.mean([m["parsing_success_rate"] for m in batch_metrics_list]),
            "total_samples": sum([m["batch_size"] for m in batch_metrics_list]),
        }
        
        # Add judge deception metrics
        if self.metrics_logger:
            judge_deception_metrics = self.metrics_logger.compute_deception_metrics(
                judge_scores_tensor, policy_correctness_tensor
            )
            for key, value in judge_deception_metrics.items():
                aggregated_metrics[f"judge_{key}"] = value
            
            # Add monitor deception metrics
            monitor_deception_metrics = self.metrics_logger.compute_deception_metrics(
                monitor_correctness_tensor, policy_correctness_tensor
            )
            for key, value in monitor_deception_metrics.items():
                aggregated_metrics[f"monitor_{key}"] = value
            
            # Add answer distribution
            answer_distribution = self.metrics_logger.compute_answer_distribution(all_parsed_answers)
            aggregated_metrics.update(answer_distribution)
        
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

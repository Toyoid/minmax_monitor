"""
Centralized Metrics Logger for RLHF and MinMax Training
Handles wandb logging and comprehensive metrics computation including deception analysis
"""
import wandb
import torch
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Union
from collections import Counter

logger = logging.getLogger(__name__)

class MetricsLogger:
    """
    Centralized metrics logger for both RLHF and MinMax training
    Handles wandb integration and comprehensive metrics computation
    """
    
    def __init__(self, project_name: str, run_name: str, config: Dict[str, Any], 
                 trainer_type: str = "minmax", wandb_config: Optional[Dict] = None):
        """
        Initialize MetricsLogger with wandb
        
        Args:
            project_name: wandb project name
            run_name: wandb run name
            config: Training configuration dict
            trainer_type: "minmax" or "rlhf" 
            wandb_config: Additional wandb configuration
        """
        self.trainer_type = trainer_type
        self.wandb_config = wandb_config or {}

        self.global_train_step = 0
        
        # Initialize wandb
        wandb_init_config = {
            "project": project_name,
            "name": run_name,
            "config": config,
            **self.wandb_config
        }
        
        # Add trainer type tag
        if "tags" not in wandb_init_config:
            wandb_init_config["tags"] = []
        wandb_init_config["tags"].append(trainer_type)
        
        try:
            wandb.init(**wandb_init_config)
            logger.info(f"Wandb initialized for {trainer_type} training: {project_name}/{run_name}")
        except Exception as e:
            logger.warning(f"Failed to initialize wandb: {e}. Metrics will still be computed but not logged.")
            self.wandb_available = False
        else:
            self.wandb_available = True
    
    def compute_deception_metrics(self, judge_scores: torch.Tensor, 
                                policy_correctness: torch.Tensor) -> Dict[str, float]:
        """
        Compute deception analysis metrics for judge model
        
        Args:
            judge_scores: Judge model's truthfulness assessments (continuous or binary)
            policy_correctness: Ground truth correctness of policy answers (0/1)
            
        Returns:
            Dictionary with accuracy, deception_success_rate, over_penalization_rate, deception_tendency
        """
        if len(judge_scores) == 0:
            return {
                'accuracy': 0.0,
                'deception_success_rate': 0.0,
                'over_penalization_rate': 0.0,
                'deception_tendency': 0.0
            }
        
        # Convert to tensors if needed
        if not isinstance(judge_scores, torch.Tensor):
            judge_scores = torch.tensor(judge_scores, dtype=torch.float32)
        if not isinstance(policy_correctness, torch.Tensor):
            policy_correctness = torch.tensor(policy_correctness, dtype=torch.float32)
        
        # Convert continuous scores to binary judgments (threshold = 0.0 for judge scores)
        judge_binary = (judge_scores > 0.0).float()
        
        # Compute confusion matrix elements
        total = len(policy_correctness)
        correct_answers_mask = (policy_correctness == 1)
        incorrect_answers_mask = (policy_correctness == 0)
        
        # True positives: judge thinks correct answers are correct
        true_positives = ((judge_binary == 1) & correct_answers_mask).sum().float()
        # False positives: judge thinks incorrect answers are correct (DECEPTION SUCCESS)
        false_positives = ((judge_binary == 1) & incorrect_answers_mask).sum().float()
        # True negatives: judge thinks incorrect answers are incorrect  
        true_negatives = ((judge_binary == 0) & incorrect_answers_mask).sum().float()
        # False negatives: judge thinks correct answers are incorrect (OVER-PENALIZATION)
        false_negatives = ((judge_binary == 0) & correct_answers_mask).sum().float()
        
        total_correct_answers = correct_answers_mask.sum().float()
        total_incorrect_answers = incorrect_answers_mask.sum().float()
        
        # Compute metrics
        accuracy = (true_positives + true_negatives) / total if total > 0 else torch.tensor(-1.0, dtype=torch.float16)
        deception_success_rate = false_positives / total_incorrect_answers if total_incorrect_answers > 0 else torch.tensor(-1.0, dtype=torch.float16)
        over_penalization_rate = false_negatives / total_correct_answers if total_correct_answers > 0 else torch.tensor(-1.0, dtype=torch.float16)
        deception_tendency = judge_scores.mean() - policy_correctness.mean()
        
        return {
            'accuracy': accuracy.item(),
            'deception_success_rate': deception_success_rate.item(),
            'over_penalization_rate': over_penalization_rate.item(),
            'deception_tendency': deception_tendency.item()
        }
    
    def compute_answer_distribution(self, parsed_answers: List[str]) -> Dict[str, float]:
        """
        Compute answer distribution percentages
        
        Args:
            parsed_answers: List of parsed answers (A/B or 0/1)
            
        Returns:
            Dictionary with answer percentages
        """
        if not parsed_answers:
            return {'answer_A_pct': 0.0, 'answer_B_pct': 0.0, 'invalid_pct': 0.0}
        
        # Count answers
        answer_counts = Counter(parsed_answers)
        total = len(parsed_answers)
        
        # Handle different answer formats
        if any(ans in ['A', 'B'] for ans in parsed_answers):
            # A/B format
            a_count = answer_counts.get('A', 0)
            b_count = answer_counts.get('B', 0)
            invalid_count = total - a_count - b_count
            
            return {
                'answer_A_pct': (a_count / total) * 100,
                'answer_B_pct': (b_count / total) * 100,
                'invalid_pct': (invalid_count / total) * 100
            }
        else:
            # 0/1 format or other
            zero_count = answer_counts.get('0', 0)
            one_count = answer_counts.get('1', 0)
            invalid_count = total - zero_count - one_count
            
            return {
                'answer_0_pct': (zero_count / total) * 100,
                'answer_1_pct': (one_count / total) * 100,
                'invalid_pct': (invalid_count / total) * 100
            }
    
    def compute_comprehensive_metrics(self, 
        # Core data
        policy_correctness: torch.Tensor,
        parsed_answers: List[str],
        reward_scores: torch.Tensor,
        judge_scores: torch.Tensor,
        combined_scores: torch.Tensor,
        # Optional MinMax data
        monitor_judgments: Optional[List[str]] = None,
        monitor_correctness: Optional[torch.Tensor] = None,
        truthfulness_penalty: Optional[torch.Tensor] = None,
        constraint_lambda: Optional[float] = None,
        # Optional loss data
        policy_loss: Optional[float] = None,
        monitor_loss: Optional[float] = None) -> Dict[str, Any]:
        """
        Compute comprehensive metrics for a batch
        
        Args:
            policy_correctness: Ground truth correctness (0/1)
            parsed_answers: Policy model answers
            reward_scores: Reward model scores
            judge_scores: Judge model scores
            combined_scores: Combined reward scores
            monitor_judgments: Monitor model judgments (MinMax only)
            monitor_correctness: Monitor correctness tensor (MinMax only)
            truthfulness_penalty: Truthfulness constraint penalty (MinMax only)
            constraint_lambda: Constraint multiplier lambda (MinMax only)
            policy_loss: Policy training loss
            monitor_loss: Monitor training loss (MinMax only)
            
        Returns:
            Dictionary of computed metrics
        """
        metrics = {}
        
        # Core metrics (both RLHF and MinMax)
        metrics['policy_accuracy'] = policy_correctness.mean().item()
        metrics['avg_reward_score'] = reward_scores.mean().item()
        metrics['avg_judge_score'] = judge_scores.mean().item() 
        metrics['avg_combined_score'] = combined_scores.mean().item()
        
        # Judge deception analysis
        judge_metrics = self.compute_deception_metrics(judge_scores, policy_correctness)
        for key, value in judge_metrics.items():
            metrics[f'judge_{key}'] = value
        
        # Answer distribution
        answer_dist = self.compute_answer_distribution(parsed_answers)
        metrics.update(answer_dist)
        
        # Loss metrics
        if policy_loss is not None:
            metrics['policy_loss'] = policy_loss
        if monitor_loss is not None:
            metrics['monitor_loss'] = monitor_loss
            
        # MinMax-specific metrics
        if self.trainer_type == "minmax":
            if truthfulness_penalty is not None:
                metrics['avg_truthfulness_penalty'] = truthfulness_penalty.mean().item()
            if constraint_lambda is not None:
                metrics['constraint_lambda'] = constraint_lambda
                
            # Monitor deception analysis
            if monitor_judgments is not None and monitor_correctness is not None:
                monitor_metrics = self.compute_deception_metrics(monitor_correctness, policy_correctness)
                for key, value in monitor_metrics.items():
                    metrics[f'monitor_{key}'] = value
                    
                # Monitor answer distribution
                monitor_dist = self.compute_answer_distribution(monitor_judgments)
                for key, value in monitor_dist.items():
                    metrics[f'monitor_{key}'] = value
        
        return metrics
    
    def log_training_batch_metrics(self, metrics: Dict[str, Any]):
        """
        Log training metrics for a batch
        
        Args:
            metrics: Dictionary of metrics to log
            step: Training step number
        """
        if not self.wandb_available:
            return
            
        # Prefix with train/ for clarity
        train_metrics = {f"train/{key}": value for key, value in metrics.items()}
        
        try:
            wandb.log(train_metrics, step=self.global_train_step)
        except Exception as e:
            logger.warning(f"Failed to log training batch metrics: {e}")
    
    def log_epoch_metrics(self, metrics: Dict[str, Any], epoch: int):
        """
        Log aggregated metrics at epoch end
        
        Args:
            metrics: Dictionary of epoch metrics
            epoch: Epoch number
        """
        if not self.wandb_available:
            return
            
        # Prefix with epoch/ for clarity
        epoch_metrics = {f"epoch/{key}": value for key, value in metrics.items()}
        epoch_metrics["epoch"] = epoch
        
        try:
            wandb.log(epoch_metrics, step=self.global_train_step)
        except Exception as e:
            logger.warning(f"Failed to log epoch metrics: {e}")
    
    def log_evaluation_metrics(self, metrics: Dict[str, Any]):
        """
        Log evaluation metrics
        
        Args:
            metrics: Dictionary of evaluation metrics
            step: Training step number
        """
        if not self.wandb_available:
            return
            
        # Prefix with eval/ for clarity
        eval_metrics = {f"eval/{key}": value for key, value in metrics.items()}
        
        try:
            wandb.log(eval_metrics, step=self.global_train_step)
        except Exception as e:
            logger.warning(f"Failed to log evaluation metrics: {e}")
    
    def finish(self):
        """Finish wandb run"""
        if self.wandb_available:
            try:
                wandb.finish()
                logger.info("Wandb run finished successfully")
            except Exception as e:
                logger.warning(f"Failed to finish wandb run: {e}")

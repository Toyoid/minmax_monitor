"""
Centralized Metrics Logger for RLHF and MinMax Training
Handles multiple logging backends (wandb, tensorboard, console) and comprehensive metrics computation
"""
import torch
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Union
from collections import Counter

from .metrics_backends import WandbBackend, TensorboardBackend, ConsoleBackend, BaseMetricsBackend

logger = logging.getLogger(__name__)

class MetricsLogger:
    """
    Centralized metrics logger for both RLHF and MinMax training
    Supports multiple backends: wandb, tensorboard, console
    """
    
    def __init__(self, project_name: str, run_name: str, config: Dict[str, Any], 
                 trainer_type: str = "minmax", backends: Optional[List[str]] = None,
                 wandb_config: Optional[Dict] = None, tensorboard_config: Optional[Dict] = None):
        """
        Initialize MetricsLogger with multiple backends
        
        Args:
            project_name: Project name for logging
            run_name: Run name for this experiment
            config: Training configuration dict
            trainer_type: "minmax" or "rlhf" 
            backends: List of backends to use ["wandb", "tensorboard", "console"]
            wandb_config: Additional wandb configuration
            tensorboard_config: Additional tensorboard configuration
        """
        self.trainer_type = trainer_type
        self.project_name = project_name
        self.run_name = run_name
        self.config = config
        self.global_train_step = 0
        
        # Default to wandb if no backends specified
        if backends is None:
            backends = ["wandb"]

        logger.info(f"Initializing MetricsLogger with backends: {', '.join(backends)}")        
        # Initialize backends
        self.backends: List[BaseMetricsBackend] = []
        self._initialize_backends(backends, wandb_config, tensorboard_config)
        
        # Log configuration to all backends
        self._log_initial_config()
        
    def _initialize_backends(self, backend_names: List[str], wandb_config: Optional[Dict], 
                           tensorboard_config: Optional[Dict]):
        """Initialize requested backends"""
        
        # Enhance config with trainer type
        enhanced_config = {**self.config, "trainer_type": self.trainer_type}
        
        for backend_name in backend_names:
            try:
                if backend_name.lower() == "wandb":
                    backend = WandbBackend(
                        self.project_name, self.run_name, enhanced_config, 
                        wandb_config=wandb_config
                    )
                elif backend_name.lower() == "tensorboard":
                    tb_config = tensorboard_config or {}
                    backend = TensorboardBackend(
                        self.project_name, self.run_name, enhanced_config, 
                        **tb_config
                    )
                elif backend_name.lower() == "console":
                    backend = ConsoleBackend(
                        self.project_name, self.run_name, enhanced_config
                    )
                else:
                    logger.warning(f"Unknown backend: {backend_name}")
                    continue
                
                if backend.is_available:
                    self.backends.append(backend)
                    logger.info(f"Successfully initialized {backend_name} backend")
                else:
                    logger.warning(f"Backend {backend_name} is not available")
                    
            except Exception as e:
                logger.error(f"Failed to initialize {backend_name} backend: {e}")
        
        # Fallback to console if no backends are available
        if not self.backends:
            logger.warning("No backends available, falling back to console logging")
            self.backends.append(ConsoleBackend(self.project_name, self.run_name, enhanced_config))
    
    def _log_initial_config(self):
        """Log initial configuration to all backends"""
        config_to_log = {
            **self.config,
            "trainer_type": self.trainer_type,
            "project_name": self.project_name,
            "run_name": self.run_name
        }
        
        for backend in self.backends:
            try:
                backend.log_hyperparameters(config_to_log)
            except Exception as e:
                logger.warning(f"Failed to log config to {type(backend).__name__}: {e}")
    
    @property
    def has_available_backends(self) -> bool:
        """Check if any backends are available"""
        return len(self.backends) > 0 and any(backend.is_available for backend in self.backends)
    
    def compute_deception_metrics(self, judge_scores: torch.Tensor, 
                                policy_correctness: torch.Tensor) -> Dict[str, float]:
        """
        Compute deception analysis metrics for judgement on policy model
        
        Args:
            judge_scores: Judge model's truthfulness assessments (continuous or binary)
            policy_correctness: Ground truth correctness of policy answers (0/1)
            
        Returns:
            Dictionary with accuracy, deception_success_rate, over_penalization_rate
        """
        if len(judge_scores) == 0:
            return {
                'accuracy': 0.0,
                'deception_success_rate': 0.0,
                'over_penalization_rate': 0.0
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

        print(60 * "=")
        print(f"Judgements - Total: {total}, TP: {true_positives}, FP: {false_positives}, TN: {true_negatives}, FN: {false_negatives}")
        print(f"Ground Truth - Total: {total}, Correct: {total_correct_answers}, Incorrect: {total_incorrect_answers}")
        print(60 * "=")

        # Compute metrics
        accuracy = (true_positives + true_negatives) / total if total > 0 else torch.tensor(-1.0, dtype=torch.float16)
        deception_success_rate = false_positives / total_incorrect_answers if total_incorrect_answers > 0 else torch.tensor(-1.0, dtype=torch.float16)
        over_penalization_rate = false_negatives / total_correct_answers if total_correct_answers > 0 else torch.tensor(-1.0, dtype=torch.float16)

        return {
            'accuracy': accuracy.item(),
            'deception_success_rate': deception_success_rate.item(),
            'over_penalization_rate': over_penalization_rate.item(),
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
        """
        if not self.has_available_backends:
            return
            
        # Log to all backends with train prefix
        for backend in self.backends:
            try:
                backend.log_metrics(metrics, step=self.global_train_step, prefix="train")
            except Exception as e:
                logger.warning(f"Failed to log training batch metrics to {type(backend).__name__}: {e}")
    
    def log_epoch_metrics(self, metrics: Dict[str, Any], epoch: int):
        """
        Log aggregated metrics at epoch end
        
        Args:
            metrics: Dictionary of epoch metrics
            epoch: Epoch number
        """
        if not self.has_available_backends:
            return
            
        # Add epoch number to metrics
        epoch_metrics = {**metrics, "epoch": epoch}
        
        # Log to all backends with epoch prefix
        for backend in self.backends:
            try:
                backend.log_metrics(epoch_metrics, step=self.global_train_step, prefix="epoch")
            except Exception as e:
                logger.warning(f"Failed to log epoch metrics to {type(backend).__name__}: {e}")
    
    def log_evaluation_metrics(self, metrics: Dict[str, Any]):
        """
        Log evaluation metrics
        
        Args:
            metrics: Dictionary of evaluation metrics
        """
        if not self.has_available_backends:
            return
            
        # Log to all backends with eval prefix
        for backend in self.backends:
            try:
                backend.log_metrics(metrics, step=self.global_train_step, prefix="eval")
            except Exception as e:
                logger.warning(f"Failed to log evaluation metrics to {type(backend).__name__}: {e}")
    
    def log_custom_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None, prefix: str = ""):
        """
        Log custom metrics with optional prefix
        
        Args:
            metrics: Dictionary of metrics to log
            step: Optional step number (uses global_train_step if None)
            prefix: Optional prefix for metric names
        """
        if not self.has_available_backends:
            return
        
        step_to_use = step if step is not None else self.global_train_step
        
        # Log to all backends
        for backend in self.backends:
            try:
                backend.log_metrics(metrics, step=step_to_use, prefix=prefix)
            except Exception as e:
                logger.warning(f"Failed to log custom metrics to {type(backend).__name__}: {e}")
    
    def finish(self):
        """Finish all logging sessions"""
        for backend in self.backends:
            try:
                backend.finish()
            except Exception as e:
                logger.warning(f"Failed to finish {type(backend).__name__}: {e}")
        
        logger.info("All metrics logging sessions finished")

"""
Abstract Base Classes for Metrics Logging
Defines interfaces for different logging backends (wandb, tensorboard, etc.)
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)

class BaseMetricsBackend(ABC):
    """
    Abstract base class for metrics logging backends
    """
    
    def __init__(self, project_name: str, run_name: str, config: Dict[str, Any], **kwargs):
        """
        Initialize metrics backend
        
        Args:
            project_name: Project name for logging
            run_name: Run name for this experiment
            config: Training configuration dict
            **kwargs: Backend-specific configuration
        """
        self.project_name = project_name
        self.run_name = run_name
        self.config = config
        self.available = False
        
    @abstractmethod
    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None, prefix: str = "") -> None:
        """
        Log metrics to the backend
        
        Args:
            metrics: Dictionary of metrics to log
            step: Step number for the metrics
            prefix: Prefix to add to metric names
        """
        pass
    
    @abstractmethod
    def log_hyperparameters(self, hparams: Dict[str, Any]) -> None:
        """
        Log hyperparameters to the backend
        
        Args:
            hparams: Dictionary of hyperparameters
        """
        pass
    
    @abstractmethod
    def finish(self) -> None:
        """
        Finish the logging session
        """
        pass
    
    @property
    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if the backend is available and ready to log
        """
        pass


class WandbBackend(BaseMetricsBackend):
    """
    Weights & Biases logging backend
    """
    
    def __init__(self, project_name: str, run_name: str, config: Dict[str, Any], 
                 wandb_config: Optional[Dict] = None, **kwargs):
        super().__init__(project_name, run_name, config, **kwargs)
        self.wandb_config = wandb_config or {}
        self._initialize_wandb()
    
    def _initialize_wandb(self):
        """Initialize wandb session"""
        try:
            import wandb
            self.wandb = wandb
            
            wandb_init_config = {
                "project": self.project_name,
                "name": self.run_name,
                "config": self.config,
                **self.wandb_config
            }
            
            self.wandb.init(**wandb_init_config)
            self.available = True
            logger.info(f"Wandb initialized: {self.project_name}/{self.run_name}")
            
        except ImportError:
            logger.warning("wandb not available. Install with: pip install wandb")
            self.available = False
        except Exception as e:
            logger.warning(f"Failed to initialize wandb: {e}")
            self.available = False
    
    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None, prefix: str = "") -> None:
        """Log metrics to wandb"""
        if not self.available:
            return
            
        # Add prefix to metrics if provided
        if prefix:
            metrics = {f"{prefix}/{key}": value for key, value in metrics.items()}
        
        try:
            if step is not None:
                self.wandb.log(metrics, step=step)
            else:
                self.wandb.log(metrics)
        except Exception as e:
            logger.warning(f"Failed to log metrics to wandb: {e}")
    
    def log_hyperparameters(self, hparams: Dict[str, Any]) -> None:
        """Log hyperparameters to wandb"""
        if not self.available:
            return
            
        try:
            # Wandb logs hyperparameters via config during init
            # But we can update config if needed
            self.wandb.config.update(hparams)
        except Exception as e:
            logger.warning(f"Failed to log hyperparameters to wandb: {e}")
    
    def finish(self) -> None:
        """Finish wandb session"""
        if self.available:
            try:
                self.wandb.finish()
                logger.info("Wandb session finished")
            except Exception as e:
                logger.warning(f"Failed to finish wandb session: {e}")
    
    @property
    def is_available(self) -> bool:
        return self.available


class TensorboardBackend(BaseMetricsBackend):
    """
    TensorBoard logging backend
    """
    
    def __init__(self, project_name: str, run_name: str, config: Dict[str, Any], 
                 log_dir: Optional[str] = None, **kwargs):
        super().__init__(project_name, run_name, config, **kwargs)
        log_parent_dir = log_dir or "./tensorboard_logs"
        self.log_dir = f"{log_parent_dir}/{project_name}/{run_name}"
        self._initialize_tensorboard()
    
    def _initialize_tensorboard(self):
        """Initialize tensorboard writer"""
        try:
            from torch.utils.tensorboard import SummaryWriter
            import os
            
            # Create log directory
            os.makedirs(self.log_dir, exist_ok=True)
            
            self.writer = SummaryWriter(log_dir=self.log_dir)
            self.available = True
            logger.info(f"TensorBoard initialized: {self.log_dir}")
            
        except ImportError:
            logger.warning("torch.utils.tensorboard not available. Install PyTorch with tensorboard support.")
            self.available = False
        except Exception as e:
            logger.warning(f"Failed to initialize TensorBoard: {e}")
            self.available = False
    
    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None, prefix: str = "") -> None:
        """Log metrics to TensorBoard"""
        if not self.available:
            return
        
        # Default step to 0 if not provided
        if step is None:
            step = 0
            
        try:
            for key, value in metrics.items():
                # Add prefix if provided
                tag = f"{prefix}/{key}" if prefix else key
                
                # Handle different types of values
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(tag, value, step)
                elif hasattr(value, 'item'):  # PyTorch tensor
                    self.writer.add_scalar(tag, value.item(), step)
                else:
                    # Try to convert to float
                    try:
                        self.writer.add_scalar(tag, float(value), step)
                    except (ValueError, TypeError):
                        logger.warning(f"Cannot log metric {key} with value {value} - unsupported type")
            
            # Flush to ensure metrics are written
            self.writer.flush()
            
        except Exception as e:
            logger.warning(f"Failed to log metrics to TensorBoard: {e}")
    
    def log_hyperparameters(self, hparams: Dict[str, Any]) -> None:
        """Log hyperparameters to TensorBoard"""
        if not self.available:
            return
            
        try:
            # Convert all values to strings or numbers for TensorBoard
            clean_hparams = {}
            for key, value in hparams.items():
                if isinstance(value, (int, float, str, bool)):
                    clean_hparams[key] = value
                else:
                    clean_hparams[key] = str(value)
            
            self.writer.flush()
            
        except Exception as e:
            logger.warning(f"Failed to log hyperparameters to TensorBoard: {e}")
    
    def finish(self) -> None:
        """Finish TensorBoard session"""
        if self.available:
            try:
                self.writer.close()
                logger.info("TensorBoard session finished")
            except Exception as e:
                logger.warning(f"Failed to finish TensorBoard session: {e}")
    
    @property
    def is_available(self) -> bool:
        return self.available


class ConsoleBackend(BaseMetricsBackend):
    """
    Console logging backend for debugging/fallback
    """
    
    def __init__(self, project_name: str, run_name: str, config: Dict[str, Any], **kwargs):
        super().__init__(project_name, run_name, config, **kwargs)
        self.available = True  # Always available
        logger.info(f"Console logger initialized: {project_name}/{run_name}")
    
    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None, prefix: str = "") -> None:
        """Log metrics to console"""
        step_str = f"[Step {step}]" if step is not None else "[No Step]"
        prefix_str = f"[{prefix}]" if prefix else ""
        
        logger.info(f"METRICS {step_str} {prefix_str}")
        for key, value in metrics.items():
            if hasattr(value, 'item'):
                value = value.item()
            logger.info(f"  {key}: {value}")
    
    def log_hyperparameters(self, hparams: Dict[str, Any]) -> None:
        """Log hyperparameters to console"""
        logger.info("HYPERPARAMETERS:")
        for key, value in hparams.items():
            logger.info(f"  {key}: {value}")
    
    def finish(self) -> None:
        """Finish console session"""
        logger.info("Console logging finished")
    
    @property
    def is_available(self) -> bool:
        return True

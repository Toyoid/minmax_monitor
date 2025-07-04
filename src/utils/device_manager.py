"""
Simplified Device Manager for MinMax RLHF Training
Development strategy: Policy & Monitor on cuda:0, others on sequential GPUs
"""
import torch
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

class DeviceManager:
    """
    Simplified device manager for development phase
    
    Device allocation strategy:
    - Policy model: cuda:0 (for PPO compatibility)
    - Monitor model: cuda:0 (for PPO compatibility)
    - Reward model: Next available GPU (cuda:1, cuda:2, etc.)
    - Judge model: Next available GPU after reward
    """
    
    def __init__(self):
        """Initialize simplified device manager"""
        self.available_gpus = self._detect_available_gpus()
        self.device_allocation = self._allocate_devices()
        
        logger.info("Simplified Device Manager initialized")
        self._log_allocation()
    
    def _detect_available_gpus(self) -> List[str]:
        """Detect available GPU devices"""
        if not torch.cuda.is_available():
            logger.warning("CUDA not available, using CPU fallback")
            return []
        
        gpu_count = torch.cuda.device_count()
        available_gpus = []
        
        for i in range(gpu_count):
            gpu = f"cuda:{i}"
            try:
                # Test GPU accessibility
                test_tensor = torch.ones(10, device=gpu)
                del test_tensor
                torch.cuda.empty_cache()
                available_gpus.append(gpu)
            except Exception as e:
                logger.warning(f"GPU {gpu} not accessible: {e}")
        
        logger.info(f"Available GPUs: {available_gpus}")
        return available_gpus
    
    def _allocate_devices(self) -> Dict[str, str]:
        """Allocate devices with simplified strategy"""
        if len(self.available_gpus) == 0:
            # CPU fallback
            logger.warning("No GPUs available, using CPU for all models")
            return {
                "policy": "cpu",
                "monitor": "cpu",
                "reward": "cpu", 
                "judge": "cpu"
            }
        
        # Policy and Monitor always on cuda:0 for PPO compatibility
        allocation = {
            "policy": "cuda:0",
            "monitor": "cuda:0"
        }
        
        # Assign reward and judge to next available GPUs
        if len(self.available_gpus) >= 2:
            allocation["reward"] = "cuda:1"
        else:
            allocation["reward"] = "cuda:0"
            
        if len(self.available_gpus) >= 3:
            allocation["judge"] = "cuda:2"
        elif len(self.available_gpus) >= 2:
            allocation["judge"] = "cuda:1"
        else:
            allocation["judge"] = "cuda:0"
        
        return allocation
    
    def get_policy_device(self) -> str:
        """Get device for policy model"""
        return self.device_allocation["policy"]
    
    def get_monitor_device(self) -> str:
        """Get device for monitor model"""
        return self.device_allocation["monitor"]
    
    def get_reward_device(self) -> str:
        """Get device for reward model"""
        return self.device_allocation["reward"]
    
    def get_judge_device(self) -> str:
        """Get device for judge model"""
        return self.device_allocation["judge"]
    
    def get_ppo_device(self) -> str:
        """Get device for PPO training (same as policy)"""
        return self.get_policy_device()
    
    def move_to_device(self, tensor: torch.Tensor, target_device: str) -> torch.Tensor:
        """Move tensor to target device"""
        return tensor.to(target_device)
    
    def _log_allocation(self):
        """Log device allocation"""
        logger.info("=== Device Allocation ===")
        logger.info(f"Policy model: {self.get_policy_device()}")
        logger.info(f"Monitor model: {self.get_monitor_device()}")
        logger.info(f"Reward model: {self.get_reward_device()}")
        logger.info(f"Judge model: {self.get_judge_device()}")
        logger.info("========================")


def create_device_manager() -> DeviceManager:
    """Create device manager instance"""
    return DeviceManager()

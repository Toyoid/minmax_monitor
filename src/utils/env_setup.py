"""
Environment Setup and GPU Management Utilities
"""
import os
import logging
import torch
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

def setup_gpu_environment() -> int:
    """
    Setup and validate GPU environment
    
    Returns:
        Number of available GPUs
    """
    cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', None)
    
    if cuda_visible:
        logger.info(f"CUDA_VISIBLE_DEVICES set to: {cuda_visible}")
        try:
            gpu_list = [x.strip() for x in cuda_visible.split(',') if x.strip()]
            logger.info(f"Will use {len(gpu_list)} GPUs: {gpu_list}")
        except Exception as e:
            logger.error(f"Error parsing CUDA_VISIBLE_DEVICES: {e}")
            gpu_list = []
    else:
        logger.warning("CUDA_VISIBLE_DEVICES not set. Will use all available GPUs.")
        gpu_list = []
        
    # Validate CUDA availability
    if not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU mode")
        return 0
        
    available_gpus = torch.cuda.device_count()
    logger.info(f"PyTorch sees {available_gpus} GPUs")
    
    # Additional environment optimizations
    setup_pytorch_optimizations()
    
    return available_gpus

def setup_pytorch_optimizations():
    """Setup PyTorch optimizations for multi-GPU training"""
    
    # Memory management optimizations
    if not os.environ.get('PYTORCH_CUDA_ALLOC_CONF'):
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512,expandable_segments:True'
        logger.info("Set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512,expandable_segments:True")
    
    # Disable some debugging for performance
    if not os.environ.get('CUDA_LAUNCH_BLOCKING'):
        os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
    
    # Enable optimized attention if available
    try:
        torch.backends.cuda.enable_flash_sdp(True)
        logger.info("Enabled Flash Attention optimization")
    except:
        logger.debug("Flash Attention not available")
    
    # Set multiprocessing start method for CUDA
    try:
        torch.multiprocessing.set_start_method('spawn', force=True)
        logger.debug("Set multiprocessing start method to 'spawn'")
    except RuntimeError:
        logger.debug("Multiprocessing start method already set")

def print_gpu_usage():
    """Print current GPU memory usage"""
    if not torch.cuda.is_available():
        logger.info("CUDA not available - CPU mode")
        return
        
    logger.info("GPU Memory Usage:")
    logger.info("-" * 60)
    
    for i in range(torch.cuda.device_count()):
        try:
            # Get device properties
            props = torch.cuda.get_device_properties(i)
            
            # Get memory usage
            memory_allocated = torch.cuda.memory_allocated(i) / 1024**3
            memory_cached = torch.cuda.memory_reserved(i) / 1024**3
            memory_total = props.total_memory / 1024**3
            
            usage_pct = (memory_allocated / memory_total * 100) if memory_total > 0 else 0
            
            logger.info(f"GPU {i} ({props.name}): "
                       f"{memory_allocated:.1f}GB/{memory_total:.1f}GB ({usage_pct:.1f}%) allocated, "
                       f"{memory_cached:.1f}GB cached")
        except Exception as e:
            logger.warning(f"Could not get memory info for GPU {i}: {e}")
    
    logger.info("-" * 60)

def print_environment_info():
    """Print comprehensive environment information"""
    logger.info("=" * 60)
    logger.info("Environment Information")
    logger.info("=" * 60)
    
    # CUDA info
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"cuDNN version: {torch.backends.cudnn.version()}")
        logger.info(f"Number of GPUs: {torch.cuda.device_count()}")
        
        # List all GPUs
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            logger.info(f"GPU {i}: {props.name} ({props.total_memory // 1024**3}GB)")
    
    # PyTorch info
    logger.info(f"PyTorch version: {torch.__version__}")
    
    # Environment variables
    important_env_vars = [
        'CUDA_VISIBLE_DEVICES',
        'PYTORCH_CUDA_ALLOC_CONF',
        'CUDA_LAUNCH_BLOCKING',
        'OMP_NUM_THREADS',
        'MKL_NUM_THREADS'
    ]
    
    logger.info("Environment Variables:")
    for var in important_env_vars:
        value = os.environ.get(var, 'Not set')
        logger.info(f"  {var}: {value}")
    
    logger.info("=" * 60)

def validate_device_allocation(device_allocation: Dict, available_gpus: List[str]) -> bool:
    """
    Validate that device allocation is feasible with available GPUs
    
    Args:
        device_allocation: Device allocation dictionary
        available_gpus: List of available GPU device strings
        
    Returns:
        True if allocation is valid, False otherwise
    """
    if "cpu" in available_gpus:
        logger.info("CPU mode - device allocation validation skipped")
        return True
    
    try:
        # Get maximum device index used
        all_device_indices = []
        all_device_indices.append(device_allocation["policy_device"])
        all_device_indices.extend(device_allocation["reward_devices"])
        all_device_indices.extend(device_allocation["judge_devices"])
        all_device_indices.append(device_allocation["data_device"])
        
        max_device_idx = max(all_device_indices)
        num_available = len(available_gpus)
        
        if max_device_idx >= num_available:
            logger.error(f"Device allocation requires device index {max_device_idx} "
                        f"but only {num_available} GPUs available")
            return False
            
        logger.info(f"Device allocation validated: max device index {max_device_idx} "
                   f"within available range [0, {num_available-1}]")
        return True
        
    except Exception as e:
        logger.error(f"Error validating device allocation: {e}")
        return False

def clear_gpu_cache():
    """Clear GPU cache to free memory"""
    if torch.cuda.is_available():
        logger.info("Clearing GPU cache...")
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i):
                torch.cuda.empty_cache()
        logger.info("GPU cache cleared")

def set_deterministic_mode(seed: int = 42):
    """Set deterministic mode for reproducible results"""
    import random
    import numpy as np
    
    logger.info(f"Setting deterministic mode with seed {seed}")
    
    # Python random
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # Additional deterministic settings
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    logger.info("Deterministic mode enabled")

def get_optimal_num_workers() -> int:
    """Get optimal number of dataloader workers based on system"""
    import multiprocessing
    
    num_cores = multiprocessing.cpu_count()
    
    # Conservative estimate: use half the cores, but at least 2 and at most 8
    optimal_workers = min(max(num_cores // 2, 2), 8)
    
    logger.info(f"Detected {num_cores} CPU cores, using {optimal_workers} dataloader workers")
    return optimal_workers


# Test functions
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    print("Testing environment setup...")
    
    # Test environment setup
    num_gpus = setup_gpu_environment()
    print(f"Available GPUs: {num_gpus}")
    
    # Print environment info
    print_environment_info()
    
    # Print GPU usage
    print_gpu_usage()
    
    # Test optimal workers
    workers = get_optimal_num_workers()
    print(f"Optimal workers: {workers}")
    
    print("Environment setup test completed!")

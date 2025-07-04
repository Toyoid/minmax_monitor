# MinMax RLHF Training Pipeline Architecture

## Overview

This project implements a **MinMax adversarial RLHF training pipeline** where a Policy model and Monitor model compete against each other using Two-Timescale Update Rule (TTUR) dynamics. The system is designed for development and debugging with simplified device management and clean separation of concerns.

## Core Purpose

**MinMax Training Objective:**
- **Policy Model**: Learns to generate responses that maximize combined reward (R_policy + Judge evaluation)  
- **Monitor Model**: Learns to critique policy outputs to minimize combined reward while respecting truthfulness constraints
- **TTUR Dynamics**: Monitor updates more frequently (inner loop) than Policy (outer loop) for stable adversarial training

## Architecture Overview

### Key Components

1. **Device Management** - Simplified multi-GPU allocation (`src/utils/device_manager.py`)
2. **Model Classes** - Policy, Monitor, Reward, Judge with LoRA support (`src/models/`)
3. **MinMax Pipeline** - Orchestrates adversarial forward pass (`src/pipelines/minmax_pipeline.py`)
4. **Dual PPO Optimizer** - Implements TTUR training dynamics (`src/train/dual_ppo_optimizer.py`)
5. **Training Coordinators** - Main training loops and evaluation (`src/train/`)

## Device Allocation Strategy

**Development-Optimized Setup:**
- **Policy & Monitor Models**: `cuda:0` (for PPO compatibility)
- **Reward Model**: `cuda:1` (sequential assignment with fallback)
- **Judge Model**: `cuda:2` (sequential assignment with fallback)
- **Automatic Fallback**: Fewer GPUs → shared devices with memory management

```python
# Device Manager (simplified)
DeviceManager:
├── Policy: cuda:0     # PPO training
├── Monitor: cuda:0    # PPO training  
├── Reward: cuda:1     # Inference only
└── Judge: cuda:2      # Inference only
```

## Project Structure

```
src/
├── config/
│   ├── dataset_config.py     # Dataset-specific configurations
│   └── model_config.py       # Model configurations & MinMax settings
├── data/
│   ├── dataset_processor.py  # Dataset-agnostic processing
│   └── qa_processor.py       # QA Simple dataset loader
├── models/
│   ├── policy_model.py       # Policy model (cuda:0, LoRA)
│   ├── monitor_model.py      # Monitor model (cuda:0, LoRA) 
│   ├── reward_model.py       # Reward model (cuda:1)
│   └── judge_model.py        # Judge model (cuda:2)
├── pipelines/
│   ├── minmax_pipeline.py    # MinMax adversarial pipeline
│   └── rlhf_pipeline.py      # Traditional RLHF pipeline
├── train/
│   ├── minmax_trainer.py     # Main MinMax training coordinator
│   ├── dual_ppo_optimizer.py # TTUR PPO optimizer
│   └── rlhf_trainer.py       # Traditional RLHF trainer
└── utils/
    ├── device_manager.py     # Simplified device allocation
    └── env_setup.py          # Environment setup
```

## Configuration Files

```
config/
├── minmax_config.json        # MinMax training configuration
└── rlhf_config.json         # Traditional RLHF configuration
```

## Core Components

### 1. MinMax Pipeline (`src/pipelines/minmax_pipeline.py`)

**Adversarial Forward Pass:**
```python
def forward_pass(batch_data) -> MinMaxOutput:
    # 1. Policy Generation: S → O
    policy_outputs = policy_model.generate(stories, questions)
    
    # 2. Monitor Critique: (S, O) → O'
    monitor_critiques = monitor_model.critique(stories, questions, policy_outputs)
    
    # 3. Reward Evaluation: O → R_reward
    reward_scores = reward_model.score(policy_outputs)
    
    # 4. Judge Evaluation: (O, O') → R_judge  
    judge_scores = judge_model.evaluate(policy_outputs, monitor_critiques)
    
    # 5. Combined Reward: R(O, O') = w1*R_reward + w2*R_judge
    combined_scores = combine_rewards(reward_scores, judge_scores)
    
    # 6. Truthfulness Constraint: F(π_M) = alignment(monitor_verdict, ground_truth)
    truthfulness_penalty = compute_constraint(monitor_critiques, ground_truth)
    
    return MinMaxOutput(...)
```

### 2. Dual PPO Optimizer (`src/train/dual_ppo_optimizer.py`)

**TTUR Implementation:**
```python
def step(minmax_output) -> metrics:
    # Inner Loop: Monitor updates (5x more frequent)
    for i in range(monitor_updates_per_policy):
        monitor_rewards = -combined_scores - λ * truthfulness_penalty
        monitor_ppo_trainer.step(monitor_queries, monitor_responses, monitor_rewards)
    
    # Outer Loop: Policy update (1x)
    policy_rewards = combined_scores
    policy_ppo_trainer.step(policy_queries, policy_responses, policy_rewards)
    
    # Dual Ascent: Update constraint multiplier λ
    if avg_violation > threshold:
        λ *= (1 + step_size)
    else:
        λ *= (1 - step_size)
```

### 3. Model Classes

**Policy Model** (`src/models/policy_model.py`):
- Generates answers for questions
- LoRA fine-tuning enabled
- Device: `cuda:0` for PPO compatibility

**Monitor Model** (`src/models/monitor_model.py`):  
- Generates critiques of policy outputs
- LoRA fine-tuning enabled
- Device: `cuda:0` for PPO compatibility

**Reward Model** (`src/models/reward_model.py`):
- Scores policy responses for helpfulness/quality
- Device: `cuda:1` for parallel inference

**Judge Model** (`src/models/judge_model.py`):
- Evaluates truthfulness of policy + monitor pairs
- Device: `cuda:2` for parallel inference

### 4. Device Manager (`src/utils/device_manager.py`)

**Simplified Allocation:**
```python
class DeviceManager:
    def __init__(self):
        self.device_allocation = {
            "policy": "cuda:0",   # PPO compatibility
            "monitor": "cuda:0",  # PPO compatibility
            "reward": "cuda:1",   # Sequential assignment
            "judge": "cuda:2"     # Sequential assignment
        }
    
    def get_policy_device(self) -> str: return "cuda:0"
    def get_monitor_device(self) -> str: return "cuda:0"
    def get_reward_device(self) -> str: return "cuda:1"
    def get_judge_device(self) -> str: return "cuda:2"
```

## Training Flow

### MinMax Training Loop

1. **Load Configuration**: `config/minmax_config.json`
2. **Initialize Models**: Load all models on assigned devices
3. **Setup Pipeline**: Create MinMax adversarial pipeline
4. **Create Optimizer**: Setup dual PPO with TTUR
5. **Training Loop**:
   ```python
   for epoch in range(num_epochs):
       for batch in train_dataloader:
           # Forward pass through MinMax pipeline
           minmax_output = pipeline.forward_pass(batch)
           
           # TTUR step (monitor inner, policy outer)
           metrics = optimizer.step(minmax_output)
           
           # Log progress
           log_metrics(metrics)
   ```

### Data Flow

```
Input Batch (stories, questions, choices)
    ↓
Policy Model (cuda:0) → Generated Answers
    ↓
Monitor Model (cuda:0) → Critiques of Answers
    ↓
Reward Model (cuda:1) → Reward Scores
    ↓
Judge Model (cuda:2) → Judge Scores  
    ↓
Combined Rewards + Truthfulness Constraints
    ↓
Dual PPO Optimizer (TTUR dynamics)
    ↓
Model Updates
```

## Configuration

### MinMax Configuration (`config/minmax_config.json`)

```json
{
  "policy_model": {
    "model_name": "gpt2",
    "load_in_8bit": true,
    "lora_r": 16,
    "temperature": 0.7,
    "max_new_tokens": 25
  },
  "monitor_model": {
    "model_name": "gpt2", 
    "use_lora": true,
    "lora_r": 16,
    "temperature": 0.8,
    "max_new_tokens": 25
  },
  "training": {
    "policy_learning_rate": 1e-5,
    "monitor_learning_rate": 3e-4,
    "monitor_updates_per_policy": 5,
    "constraint_penalty_lambda": 0.1,
    "batch_size": 2,
    "num_epochs": 8
  }
}
```

## Entry Points

### 1. MinMax Training
```bash
# Main MinMax training
python -m src.train.minmax_trainer --config config/minmax_config.json

# Evaluation only
python -m src.train.minmax_trainer --config config/minmax_config.json --eval-only

# Custom epochs
python -m src.train.minmax_trainer --config config/minmax_config.json --epochs 5
```

### 2. Traditional RLHF Training  
```bash
# Traditional RLHF (no adversarial training)
python -m src.train.rlhf_trainer --config config/rlhf_config.json

# Evaluation only
python -m src.train.rlhf_trainer --config config/rlhf_config.json --eval-only
```

## Key Features

### ✅ Adversarial Training
- **Policy vs Monitor**: Competitive dynamics via TTUR
- **Constraint Optimization**: Dual ascent for truthfulness
- **Stable Training**: Inner/outer loop separation

### ✅ Simplified Device Management
- **PPO Compatible**: Both trainable models on `cuda:0`
- **Multi-GPU Utilization**: Reward/Judge on separate GPUs
- **Development Focused**: Easy debugging and modification
- **Automatic Fallback**: Graceful degradation with fewer GPUs

### ✅ Efficient Implementation
- **LoRA Training**: Memory-efficient fine-tuning for Policy/Monitor
- **Batch Processing**: Full pipeline batch support
- **Clean Separation**: Models, pipeline, optimizer decoupled
- **Memory Management**: Aggressive GPU memory clearing

### ✅ Robust Configuration
- **JSON Configuration**: Easy hyperparameter tuning via config files
- **Model Flexibility**: Support for different architectures
- **Training Control**: Configurable TTUR dynamics
- **Device Mapping**: Automatic device allocation with fallbacks

## Training Flow

### MinMax Training Loop

1. **Load Configuration**: `config/minmax_config.json`
2. **Initialize Device Manager**: Allocate GPUs for models
3. **Load Models**: Policy/Monitor on cuda:0, Reward/Judge on other GPUs
4. **Setup Pipeline**: Create MinMax adversarial pipeline
5. **Create Optimizer**: Setup dual PPO with TTUR
6. **Training Loop**:
   ```python
   for epoch in range(num_epochs):
       for batch in train_dataloader:
           # Forward pass through MinMax pipeline
           minmax_output = pipeline.forward_pass(batch)
           
           # TTUR step (monitor inner, policy outer)
           metrics = optimizer.step(minmax_output)
           
           # Log progress and save checkpoints
           log_metrics(metrics)
   ```

### Data Flow

```
Input Batch (stories, questions, choices)
    ↓
Policy Model (cuda:0) → Generated Answers
    ↓
Monitor Model (cuda:0) → Critiques of Answers
    ↓
Reward Model (cuda:1) → Reward Scores
    ↓
Judge Model (cuda:2) → Judge Scores  
    ↓
Combined Rewards + Truthfulness Constraints
    ↓
Dual PPO Optimizer (TTUR dynamics)
    ↓
Model Updates (Policy & Monitor)
```

## System Requirements

- **Minimum**: 2 GPUs (basic functionality, some sharing)
- **Recommended**: 3+ GPUs (optimal performance with full separation)
- **Memory**: 8-16 GB per GPU depending on model size
- **Dependencies**: PyTorch, Transformers, TRL, PEFT

This architecture provides a clean, debuggable foundation for MinMax adversarial RLHF training while maintaining efficiency and extensibility.

# MinMax RLHF Training Framework: Monitor-vs-LLM

## Overview

This project implements a **MinMax adversarial RLHF training framework** where a **Policy model (π_L)** and **Monitor model (π_M)** compete against each other in a game-theoretic setting. The system follows the theoretical formulation of the Monitor-vs-LLM game with **Two-Timescale Update Rule (TTUR)** dynamics for stable adversarial training.

## Core Theoretical Formulation

The framework implements the following mathematical formulation:

### Game-Theoretic Setup

| Symbol | Description | Implementation |
|--------|-------------|----------------|
| **S** | Global state (prompt, context) | `(story, question, answer_choices)` |
| **π_L** | LLM Policy: S → O | Policy model generating answers |
| **π_M** | Monitor Policy: S → O' | Monitor model generating critiques |
| **R(O,O')** | Human reward function | Combined reward + judge evaluation |

### Training Objective

**Payoff Function:**
```
U(π_L, π_M) = E_S∼D[ R(O = π_L(S), O' = π_M(S)) ]
```

**Fidelity Constraint:**
```
F(π_M) = E_S[ d(O', Truth(S)) ] ≤ δ
```

**Saddle Point Optimization:**
```
min_π_M max_π_L  U(π_L, π_M) + λ F(π_M)
```

The system seeks **ε-approximate stationary points** where both `‖∇_θ_L U‖ ≤ ε` and `‖∇_θ_M U‖ ≤ ε`.

### TTUR Implementation

- **Monitor (Inner Loop)**: Higher learning rate, updated 5× per policy update
- **Policy (Outer Loop)**: Lower learning rate, updated once per cycle  
- **Dual Ascent**: Constraint multiplier λ updated based on violation magnitude

## Architecture Overview

### Key Components

1. **Device Management** - Multi-GPU allocation with PPO compatibility (`src/utils/device_manager.py`)
2. **Model Classes** - Policy, Monitor, Reward, Judge with LoRA fine-tuning (`src/models/`)
3. **MinMax Pipeline** - Orchestrates adversarial forward pass (`src/pipelines/minmax_pipeline.py`)
4. **Dual PPO Optimizer** - Implements TTUR dynamics (`src/train/dual_ppo_optimizer.py`)
5. **Training Coordinators** - Main training loops and evaluation (`src/train/`)
6. **Metrics System** - Multi-backend logging (WandB, TensorBoard, Console) (`src/utils/`)

## Device Allocation Strategy

**Development-Optimized Setup:**
- **Policy & Monitor Models**: `cuda:0` (for PPO training compatibility)
- **Reward Model**: `cuda:1` (independent inference)
- **Judge Model**: `cuda:2` (independent inference)
- **Automatic Fallback**: Graceful degradation for fewer GPUs

```python
# Device Manager Allocation
DeviceManager:
├── Policy Model: cuda:0      # LoRA + PPO training
├── Monitor Model: cuda:0     # LoRA + PPO training
├── Reward Model: cuda:1      # Inference only
└── Judge Model: cuda:2       # Inference only
```

## Project Structure

```
src/
├── config/
│   ├── dataset_config.py     # Dataset-specific configurations (QA Simple)
│   └── model_config.py       # Model configs & MinMax training parameters
├── data/
│   ├── dataset_processor.py  # Dataset-agnostic processing + data structures
│   └── qa_processor.py       # QA Simple dataset loader and utilities
├── models/
│   ├── policy_model.py       # Policy model (π_L) with LoRA fine-tuning
│   ├── monitor_model.py      # Monitor model (π_M) with LoRA fine-tuning
│   ├── reward_model.py       # Reward model for scoring outputs
│   ├── judge_model.py        # Judge model for evaluating correctness
│   └── reward_plugins/       # Extensible reward model plugins
├── pipelines/
│   ├── minmax_pipeline.py    # MinMax adversarial pipeline (main)
│   └── rlhf_pipeline.py      # Traditional RLHF pipeline (baseline)
├── train/
│   ├── minmax_trainer.py     # Main MinMax training coordinator
│   ├── dual_ppo_optimizer.py # TTUR PPO optimizer implementation
│   └── rlhf_trainer.py       # Traditional RLHF trainer (baseline)
└── utils/
    ├── device_manager.py     # Multi-GPU device allocation
    ├── metrics_logger.py     # Multi-backend metrics logging
    ├── metrics_backends.py   # WandB, TensorBoard, Console backends
    └── env_setup.py          # Environment setup utilities
```

## Configuration Files

```
config/
├── minmax_config.json        # MinMax training configuration
└── rlhf_config.json         # Traditional RLHF configuration (baseline)

dataset/
├── qa_simple/               # QA Simple dataset for development
│   └── data/
│       ├── train.json       # Training data
│       └── val.json         # Validation data
└── quality/                 # QuALITY dataset for advanced testing
```

## Core Implementation

### 1. MinMax Pipeline (`src/pipelines/minmax_pipeline.py`)

**Adversarial Forward Pass Implementation:**

```python
def forward_pass(batch_data) -> MinMaxOutput:
    # 1. Policy Generation: S → O
    policy_outputs = policy_model.generate(stories, questions, answer_choices)
    parsed_answers = extract_answers(policy_outputs)
    
    # 2. Monitor Critique: (S, O) → O'
    monitor_critiques = monitor_model.critique(stories, questions, policy_outputs)
    
    # 3. Reward Evaluation: O → R_reward
    reward_scores = reward_model.score(policy_outputs)
    
    # 4. Monitor Reward Evaluation: O' → R_monitor
    monitor_reward_scores = reward_model.score(monitor_critiques)
    
    # 5. Judge Evaluation: (O, O') → R_judge
    judge_scores = judge_model.evaluate(policy_outputs, monitor_critiques)
    
    # 6. Combined Reward: R(O, O') = w1*R_reward + w2*R_judge
    combined_scores = combine_rewards(reward_scores, judge_scores)
    
    # 7. Truthfulness Constraint: F(π_M) = α*verdict_penalty + β*reward_penalty
    truthfulness_penalty = compute_constraint(
        monitor_critiques, ground_truth, monitor_reward_scores,
        monitor_verdict_weight=0.7, monitor_reward_weight=0.3
    )
    
    return MinMaxOutput(...)
```

### 2. Dual PPO Optimizer (`src/train/dual_ppo_optimizer.py`)

**TTUR Implementation:**

```python
def step(minmax_output) -> Dict[str, float]:
    # Extract training data
    policy_queries, policy_responses = extract_policy_ppo_data(minmax_output)
    monitor_queries, monitor_responses = extract_monitor_ppo_data(minmax_output)
    
    # Update constraint multiplier (dual ascent)
    avg_violation = truthfulness_penalties.mean()
    λ += step_size * (avg_violation - threshold)
    
    # Inner Loop: Monitor updates (5× more frequent)
    for i in range(monitor_updates_per_policy):
        # Monitor maximizes: -R(O,O') - λ*F(π_M)
        monitor_rewards = -judge_scores - λ * truthfulness_penalties
        monitor_ppo_trainer.step(monitor_queries, monitor_responses, monitor_rewards)
    
    # Outer Loop: Policy update (1×)
    # Policy maximizes: R(O,O')
    policy_rewards = combined_scores
    policy_ppo_trainer.step(policy_queries, policy_responses, policy_rewards)
    
    return training_metrics
```

### 3. Model Classes

**Policy Model** (`src/models/policy_model.py`):
- **Purpose**: Generate answers to questions (π_L: S → O)
- **Architecture**: Mistral-7B-Instruct with LoRA fine-tuning
- **Device**: `cuda:0` for PPO training compatibility
- **Output**: Structured answers with justifications

**Monitor Model** (`src/models/monitor_model.py`):
- **Purpose**: Generate critiques of policy outputs (π_M: S,O → O')
- **Architecture**: Llama-3.1-8B-Instruct with LoRA fine-tuning
- **Device**: `cuda:0` for PPO training compatibility
- **Output**: Critical analysis and alternative suggestions

**Reward Model** (`src/models/reward_model.py`):
- **Purpose**: Score outputs for helpfulness/quality
- **Architecture**: Skywork-Reward-V2-Llama-3.1-8B
- **Device**: `cuda:1` for parallel inference
- **Output**: Scalar reward scores

**Judge Model** (`src/models/judge_model.py`):
- **Purpose**: Evaluate truthfulness and correctness
- **Architecture**: Mistral-7B-Instruct
- **Device**: `cuda:2` for parallel inference
- **Output**: Binary correctness judgments

### 4. Device Manager (`src/utils/device_manager.py`)

**Simplified Multi-GPU Allocation:**

```python
class DeviceManager:
    def __init__(self):
        self.device_allocation = {
            "policy": "cuda:0",    # PPO training required
            "monitor": "cuda:0",   # PPO training required
            "reward": "cuda:1",    # Inference only
            "judge": "cuda:2"      # Inference only
        }
    
    # Automatic fallback for fewer GPUs
    def _allocate_devices(self):
        if len(available_gpus) < 3:
            # Share devices intelligently
            ...
```

### 5. Metrics System (`src/utils/metrics_logger.py`)

**Multi-Backend Logging:**

```python
class MetricsLogger:
    def __init__(self, backends=["wandb", "tensorboard", "console"]):
        self.backends = [
            WandbBackend(),      # Experiment tracking
            TensorboardBackend(),  # Local visualization  
            ConsoleBackend()     # Real-time monitoring
        ]
    
    def log_training_batch_metrics(self, metrics):
        # Log deception metrics, training losses, rewards
        
    def log_evaluation_metrics(self, metrics):
        # Log accuracy, answer distributions, model performance
```

## Training Flow

### MinMax Training Loop

1. **Configuration Loading**: Load `config/minmax_config.json` with model and training parameters
2. **Device Initialization**: Initialize `DeviceManager` for multi-GPU allocation
3. **Model Loading**: 
   - Policy/Monitor models → `cuda:0` with LoRA fine-tuning
   - Reward/Judge models → separate GPUs for parallel inference
4. **Pipeline Setup**: Create `MinMaxPipeline` with all models
5. **Optimizer Initialization**: Setup `DualPPOMinMaxOptimizer` with TTUR configuration
6. **Training Loop**:

```python
for epoch in range(num_epochs):
    for batch in train_dataloader:
        # Forward pass through MinMax pipeline
        minmax_output = pipeline.forward_pass(batch)
        
        # TTUR step (monitor inner, policy outer)
        metrics = optimizer.step(minmax_output)
        
        # Log comprehensive metrics
        log_training_metrics(metrics)
        
    # Periodic evaluation and checkpointing
    if epoch % eval_frequency == 0:
        eval_metrics = evaluate()
        log_evaluation_metrics(eval_metrics)
```

### Data Flow Architecture

```
Input Batch: (stories, questions, answer_choices, correct_answers)
    ↓
[Policy Model - cuda:0] → Generated Answers + Justifications
    ↓
[Monitor Model - cuda:0] → Critiques of Policy Outputs
    ↓
[Reward Model - cuda:1] → Reward Scores (Policy + Monitor outputs)
    ↓
[Judge Model - cuda:2] → Correctness Evaluations
    ↓
Combined Rewards + Truthfulness Constraints
    ↓
[Dual PPO Optimizer] → TTUR Updates (Monitor 5×, Policy 1×)
    ↓
Model Parameter Updates
```

### RLHF Training (Baseline)

For comparison, the system also implements traditional RLHF:

**Forward Pass:**
1. Policy generates answers
2. Reward model scores outputs
3. Judge evaluates correctness

**Backward Pass:**
- Combined reward (reward + judge) → PPO update of policy only

## Configuration

### MinMax Configuration (`config/minmax_config.json`)

```json
{
  "policy_model": {
    "model_name": "mistralai/Mistral-7B-Instruct-v0.3",
    "use_lora": true,
    "lora_r": 16,
    "temperature": 1.0,
    "max_new_tokens": 200
  },
  "monitor_model": {
    "model_name": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "use_lora": true,
    "lora_r": 16,
    "temperature": 1.0,
    "max_new_tokens": 200
  },
  "training": {
    "policy_learning_rate": 1e-6,
    "monitor_learning_rate": 4e-6,
    "monitor_updates_per_policy": 5,
    "constraint_penalty_lambda": 10.0,
    "constraint_threshold": -0.2,
    "dual_ascent_step_size": 0.25,
    "batch_size": 16,
    "num_epochs": 10
  },
  "logging": {
    "backends": ["wandb", "tensorboard", "console"],
    "wandb_config": {
      "project": "minmax_rlhf"
    }
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

# Custom epochs and dataset
python -m src.train.minmax_trainer --config config/minmax_config.json --epochs 5 --dataset qa_simple
```

### 2. Traditional RLHF Training (Baseline)
```bash
# Traditional RLHF (no adversarial training)
python -m src.train.rlhf_trainer --config config/rlhf_config.json

# Evaluation only
python -m src.train.rlhf_trainer --config config/rlhf_config.json --eval-only
```

### 2. Test Scripts
```bash
# Test environment setup
bash scripts/test_minmax.sh
bash scripts/test_rlhf.sh

# Install environment
bash install_env.sh
```
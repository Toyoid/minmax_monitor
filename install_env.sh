#!/bin/bash
# One-click environment setup script - setup_environment.sh
# Compatible with Linux/macOS systems

set -e  # Exit immediately on error

echo "🚀 Starting minmax-monitor environment setup..."

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "❌ Error: conda not found, please install Anaconda/Miniconda first"
    exit 1
fi

# Environment name
ENV_NAME="minmax-monitor"

# Check if environment already exists
echo "🔍 Checking for existing environment..."
if conda info --envs | grep -q "^${ENV_NAME}"; then
    echo "⚠️  Environment '${ENV_NAME}' already exists!"
    echo "📝 Please remove the existing environment first using:"
    echo "   conda env remove -n ${ENV_NAME}"
    echo "   Then run this script again."
    exit 1
fi

# Create base environment with Python and essential tools
echo "📦 Creating base Python environment..."
conda create -n ${ENV_NAME} python=3.11 -y

# Activate environment
echo "🔄 Activating environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ${ENV_NAME}

# Install PyTorch
echo "🔥 Installing PyTorch 2.6.0..."
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

# Upgrade pip and setuptools
echo "⬆️ Upgrading pip and base tools..."
pip install --upgrade pip setuptools wheel

# Install core ML packages
echo "🤖 Installing core machine learning packages..."
pip install transformers>=4.30.0
pip install datasets tokenizers accelerate
pip install sentencepiece

# Install RLHF training packages
echo "🎯 Installing RLHF training packages..."
pip install trl==0.11.3
pip install peft

# Install optimization tools
echo "⚡ Installing optimization tools..."

# Install CUDA Toolkit within the conda environment for DeepSpeed/bitsandbytes
echo "🔧 Installing CUDA Toolkit 12.4 inside the environment..."
conda install -c "nvidia/label/cuda-12.4.0" cuda-toolkit -y

echo "🔧 Installing DeepSpeed and bitsandbytes..."
pip install deepspeed
pip install bitsandbytes

# echo "🔧 Installing FlashAttention..."
# pip install flash-attn --no-build-isolation

# Install additional dependencies
echo "📊 Installing additional dependencies..."
pip install -r requirements_additional.txt

echo "Environment setup completed successfully"
echo ""
echo "Usage:"
echo "  conda activate ${ENV_NAME}"
echo ""

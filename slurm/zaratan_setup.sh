#!/bin/bash
# Setup script for Zaratan GPU cluster
# Run this once after copying files to Zaratan

set -e

echo "=========================================="
echo "ICL Drone Racing - Zaratan Setup"
echo "=========================================="

# Load modules
echo "Loading modules..."
module load python/3.10
module load cuda/11.8
module load cudnn/8.6.0

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv_zaratan
source venv_zaratan/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch with CUDA support
echo "Installing PyTorch with CUDA 11.8..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
echo "Installing dependencies..."
pip install h5py numpy tqdm matplotlib seaborn

# Verify CUDA
echo ""
echo "=========================================="
echo "Verifying CUDA availability..."
echo "=========================================="
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU count: {torch.cuda.device_count()}'); print(f'GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Activate environment: source venv_zaratan/bin/activate"
echo "2. Submit training job: sbatch zaratan_train.slurm"

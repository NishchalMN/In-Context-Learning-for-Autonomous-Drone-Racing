#!/bin/bash
#SBATCH --job-name=icl_drone_continue
#SBATCH --output=logs/train_continue_%j.out
#SBATCH --error=logs/train_continue_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80GB
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=your_email@umd.edu

# Continue ICL Drone Racing Training from Epoch 100
# With learning rate scheduler for improved convergence

echo "=========================================="
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "=========================================="

# Load modules
module load python/3.10
module load cuda/11.8
module load cudnn/8.6.0

# Activate virtual environment
source venv_zaratan/bin/activate

# Print environment info
echo ""
echo "Python version:"
python3 --version
echo ""
echo "PyTorch info:"
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
echo ""

# Create directories
mkdir -p checkpoints/ratm_icl_expanded
mkdir -p logs

# Continue training from epoch 100
echo "=========================================="
echo "Continuing training from epoch 100..."
echo "Dataset: data/ratm_racing_dataset_expanded.h5"
echo "Resume from: checkpoints/ratm_icl_expanded/checkpoint_epoch_100.pt"
echo "Target epochs: 150"
echo "GPU: H100 (80GB)"
echo "Batch size: 256"
echo "Sequence length: 512"
echo "Learning rate scheduler: ReduceLROnPlateau (enabled)"
echo "=========================================="
echo ""

cd drone_icl

python3 train.py \
    --data ../data/ratm_racing_dataset_expanded.h5 \
    --epochs 150 \
    --resume ../checkpoints/ratm_icl_expanded/checkpoint_epoch_100.pt \
    --batch-size 256 \
    --max-seq-len 512 \
    --num-demos 3 \
    --device cuda \
    --checkpoint-dir ../checkpoints/ratm_icl_expanded \
    --log-dir ../logs \
    --augment \
    --augment-prob 0.5 \
    --lr 1e-4 \
    --weight-decay 1e-5 \
    --num-workers 4 \
    --save-freq 5

echo ""
echo "=========================================="
echo "Training completed at: $(date)"
echo "=========================================="

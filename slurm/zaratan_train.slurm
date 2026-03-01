#!/bin/bash
#SBATCH --job-name=icl_drone_train
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80GB
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=your_email@umd.edu

# ICL Drone Racing Training on Zaratan H100 GPU
# Optimized config: batch_size=256, max_seq_len=512
# Estimated training time: 4-8 hours for 100 epochs

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

# Train with expanded dataset
echo "=========================================="
echo "Starting training with expanded dataset..."
echo "Dataset: data/ratm_racing_dataset_expanded.h5"
echo "Episodes: 19,796"
echo "GPU: H100 (80GB)"
echo "Batch size: 256"
echo "Sequence length: 512"
echo "Split: Episode-level (no data leakage)"
echo "Augmentation: ENABLED"
echo "=========================================="
echo ""

cd drone_icl

python3 train.py \
    --data ../data/ratm_racing_dataset_expanded.h5 \
    --epochs 100 \
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

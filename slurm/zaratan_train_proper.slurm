#!/bin/bash
#SBATCH --job-name=icl_drone_proper
#SBATCH --output=logs/train_proper_%j.out
#SBATCH --error=logs/train_proper_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80GB
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=your_email@umd.edu

# ICL Drone Racing Training - Proper Data (No Leakage)
# Training from scratch with correctly prepared train/val split

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
mkdir -p checkpoints/ratm_icl_proper
mkdir -p logs

# Training configuration
echo "=========================================="
echo "TRAINING WITH PROPER DATA (NO LEAKAGE)"
echo "=========================================="
echo ""
echo "Train set: data/train_final.h5"
echo "  - Base episodes: 28"
echo "  - Augmented episodes: 262"
echo "  - Windows: 137,967"
echo ""
echo "Val set: data/val_final.h5"
echo "  - Base episodes: 8 (NO augmentation)"
echo "  - Windows: 4,331"
echo "  - Tracks: trackRATM (3), lemniscate (3), ellipse (2)"
echo ""
echo "Key improvements:"
echo "  1. Data: Split THEN augment train only → no leakage"
echo "  2. Gate awareness: Model sees gate positions, directions, distances"
echo "     - Enables generalization to NEW track layouts!"
echo "  3. Validation episodes are TRULY UNSEEN"
echo ""
echo "Model features:"
echo "  - State dim: 26D (19 base + 7 gate features)"
echo "  - Gate direction (3D), distance (1D), relative pos (3D)"
echo "  - Should generalize to new tracks/gate configurations"
echo ""
echo "Expect HIGHER val loss (0.15-0.20) - this is CORRECT!"
echo "Previous val loss 0.118 was artificially low due to leakage."
echo "=========================================="
echo ""

cd drone_icl

python3 train.py \
    --data ../data/train_final.h5 \
    --val-data ../data/val_final.h5 \
    --epochs 150 \
    --batch-size 256 \
    --max-seq-len 512 \
    --num-demos 3 \
    --device cuda \
    --checkpoint-dir ../checkpoints/ratm_icl_proper \
    --log-dir ../logs \
    --lr 1e-4 \
    --weight-decay 1e-5 \
    --num-workers 4 \
    --save-freq 5 \
    --embed-dim 256 \
    --num-heads 8 \
    --num-decoder-layers 6 \
    --dropout 0.1 \
    --use-gate-info \
    2>&1 | tee ../logs/train_proper_${SLURM_JOB_ID}.log

echo ""
echo "=========================================="
echo "Training completed at: $(date)"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Download checkpoints from Zaratan"
echo "2. Run comprehensive evaluation:"
echo "   python scripts/comprehensive_evaluation.py \\"
echo "       --checkpoint checkpoints/ratm_icl_proper/checkpoint_best.pt \\"
echo "       --data data/val_final.h5"
echo ""
echo "3. Compare results to old training (with leakage)"
echo "   Expected: Higher val loss, but more trustworthy"
echo "=========================================="

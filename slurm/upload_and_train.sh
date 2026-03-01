#!/bin/bash
# Quick script to upload data and start training on Zaratan

set -e  # Exit on error

ZARATAN_USER="your_username"  # UPDATE THIS
ZARATAN_HOST="zaratan.umd.edu"
ZARATAN_DIR="~/Project"

echo "========================================"
echo "Upload Data & Start Training on Zaratan"
echo "========================================"
echo ""

# Check if datasets exist
if [ ! -f "data/train_final.h5" ]; then
    echo "❌ Error: data/train_final.h5 not found!"
    echo "   Run data preparation pipeline first."
    exit 1
fi

if [ ! -f "data/val_final.h5" ]; then
    echo "❌ Error: data/val_final.h5 not found!"
    echo "   Run data preparation pipeline first."
    exit 1
fi

echo "✅ Datasets found:"
echo "   - data/train_final.h5 ($(ls -lh data/train_final.h5 | awk '{print $5}'))"
echo "   - data/val_final.h5 ($(ls -lh data/val_final.h5 | awk '{print $5}'))"
echo ""

# Upload datasets
echo "📤 Uploading datasets to Zaratan..."
scp data/train_final.h5 ${ZARATAN_USER}@${ZARATAN_HOST}:${ZARATAN_DIR}/data/
scp data/val_final.h5 ${ZARATAN_USER}@${ZARATAN_HOST}:${ZARATAN_DIR}/data/
echo "✅ Datasets uploaded"
echo ""

# Upload updated training script
echo "📤 Uploading updated training script..."
scp drone_icl/train.py ${ZARATAN_USER}@${ZARATAN_HOST}:${ZARATAN_DIR}/drone_icl/
echo "✅ Training script uploaded"
echo ""

# Upload SLURM job script
echo "📤 Uploading SLURM job script..."
scp zaratan_train_proper.slurm ${ZARATAN_USER}@${ZARATAN_HOST}:${ZARATAN_DIR}/
echo "✅ SLURM script uploaded"
echo ""

# Submit job
echo "🚀 Submitting training job..."
ssh ${ZARATAN_USER}@${ZARATAN_HOST} "cd ${ZARATAN_DIR} && sbatch zaratan_train_proper.slurm"
echo ""

echo "========================================"
echo "✅ Upload complete! Training job submitted."
echo "========================================"
echo ""
echo "Monitor training progress:"
echo "  ssh ${ZARATAN_USER}@${ZARATAN_HOST}"
echo "  squeue -u ${ZARATAN_USER}"
echo "  tail -f ~/Project/logs/train_proper_<JOB_ID>.out"
echo ""
echo "Expected training time: ~15-18 hours on H100"
echo "Expected val loss: 0.15-0.20 (higher than before, but honest!)"
echo ""

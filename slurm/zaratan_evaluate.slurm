#!/bin/bash
#SBATCH --job-name=icl_drone_eval
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err
#SBATCH --time=4:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB

# ICL Drone Racing Evaluation on Zaratan GPU
# Run this after training completes

echo "=========================================="
echo "Evaluation started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "=========================================="

# Load modules
module load python/3.10
module load cuda/11.8
module load cudnn/8.6.0

# Activate virtual environment
source venv_zaratan/bin/activate

# Create directories
mkdir -p results/expanded
mkdir -p results/trajectories_expanded
mkdir -p results/rollouts_expanded

# Print environment info
echo ""
echo "Python version:"
python3 --version
echo ""
echo "PyTorch info:"
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
echo ""

# Evaluate model on expanded dataset
echo "=========================================="
echo "Evaluating model on expanded dataset..."
echo "=========================================="
echo ""

python3 scripts/evaluate_icl.py \
    --checkpoint checkpoints/ratm_icl_expanded/best_model.pt \
    --data data/ratm_racing_dataset_expanded.h5 \
    --output-dir results/expanded \
    --num-demos 1 3 \
    --device cuda \
    --max-seq-len 256 \
    --subsample-factor 10

echo ""
echo "=========================================="
echo "Evaluation completed at: $(date)"
echo "=========================================="

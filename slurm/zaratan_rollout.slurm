#!/bin/bash
#SBATCH --job-name=icl_drone_rollout
#SBATCH --output=logs/rollout_%j.out
#SBATCH --error=logs/rollout_%j.err
#SBATCH --time=3:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB

# ICL Drone Racing Autoregressive Rollout on Zaratan GPU
# Run this after training completes

echo "=========================================="
echo "Rollout inference started at: $(date)"
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
mkdir -p results/rollouts_expanded
mkdir -p checkpoints/dynamics_expanded

# Print environment info
echo ""
echo "Python version:"
python3 --version
echo ""
echo "PyTorch info:"
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
echo ""

# Train dynamics model first (needed for autoregressive rollout)
echo "=========================================="
echo "Training dynamics model..."
echo "=========================================="
echo ""

python3 scripts/train_dynamics.py \
    --data data/ratm_racing_dataset_expanded.h5 \
    --checkpoint-dir checkpoints/dynamics_expanded \
    --epochs 50 \
    --batch-size 64 \
    --device cuda

echo ""
echo "=========================================="
echo "Running autoregressive rollouts..."
echo "=========================================="
echo ""

# Run rollouts on diverse episodes from each track
# Episode 0: ellipse track
python3 scripts/rollout_trajectory.py \
    --checkpoint checkpoints/ratm_icl_expanded/best_model.pt \
    --data data/ratm_racing_dataset_expanded.h5 \
    --output-dir results/rollouts_expanded/episode_0 \
    --episode-idx 0 \
    --num-demos 3 \
    --max-steps 200 \
    --train-dynamics \
    --dynamics-path checkpoints/dynamics_expanded/best_model.pt \
    --device cuda \
    --max-seq-len 256 \
    --subsample-factor 10

# Episode 1000: lemniscate track
python3 scripts/rollout_trajectory.py \
    --checkpoint checkpoints/ratm_icl_expanded/best_model.pt \
    --data data/ratm_racing_dataset_expanded.h5 \
    --output-dir results/rollouts_expanded/episode_1000 \
    --episode-idx 1000 \
    --num-demos 3 \
    --max-steps 200 \
    --train-dynamics \
    --dynamics-path checkpoints/dynamics_expanded/best_model.pt \
    --device cuda \
    --max-seq-len 256 \
    --subsample-factor 10

# Episode 5000: trackRATM
python3 scripts/rollout_trajectory.py \
    --checkpoint checkpoints/ratm_icl_expanded/best_model.pt \
    --data data/ratm_racing_dataset_expanded.h5 \
    --output-dir results/rollouts_expanded/episode_5000 \
    --episode-idx 5000 \
    --num-demos 3 \
    --max-steps 200 \
    --train-dynamics \
    --dynamics-path checkpoints/dynamics_expanded/best_model.pt \
    --device cuda \
    --max-seq-len 256 \
    --subsample-factor 10

echo ""
echo "=========================================="
echo "Rollout inference completed at: $(date)"
echo "=========================================="

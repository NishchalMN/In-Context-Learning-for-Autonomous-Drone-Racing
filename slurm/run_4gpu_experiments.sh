#!/bin/bash
# Run 4 parallel training experiments on 4 H100 80GB GPUs

# Kill existing experiments
echo "Killing existing tmux sessions..."
tmux kill-session -t exp1 2>/dev/null || true
tmux kill-session -t exp2 2>/dev/null || true
tmux kill-session -t exp3 2>/dev/null || true
tmux kill-session -t exp4 2>/dev/null || true

# Kill any remaining python processes
pkill -f "drone_icl/train.py" || true

echo "Waiting 20 seconds for GPU memory to clear..."
sleep 20

# Setup
mkdir -p checkpoints/{exp1,exp2,exp3,exp4} logs

# Experiment 1: Baseline (embed=256, batch=256, seq=400, demos=3)
tmux new-session -d -s exp1 "CUDA_VISIBLE_DEVICES=0 python3 drone_icl/train.py \
    --data data/train_final.h5 \
    --val-data data/val_final.h5 \
    --epochs 150 --batch-size 256 --max-seq-len 400 --num-demos 3 \
    --checkpoint-dir checkpoints/exp1 --log-dir logs \
    --lr 1e-4 --weight-decay 1e-5 --num-workers 20 --save-freq 5 \
    --embed-dim 256 --device cuda \
    2>&1 | tee logs/exp1.log"

# Experiment 2: More demos (embed=256, batch=192, seq=400, demos=4)
tmux new-session -d -s exp2 "CUDA_VISIBLE_DEVICES=1 python3 drone_icl/train.py \
    --data data/train_final.h5 \
    --val-data data/val_final.h5 \
    --epochs 150 --batch-size 192 --max-seq-len 400 --num-demos 4 \
    --checkpoint-dir checkpoints/exp2 --log-dir logs \
    --lr 1e-4 --weight-decay 1e-5 --num-workers 20 --save-freq 5 \
    --embed-dim 256 --device cuda \
    2>&1 | tee logs/exp2.log"

# Experiment 3: Large model (embed=384, batch=128, seq=400, demos=3)
tmux new-session -d -s exp3 "CUDA_VISIBLE_DEVICES=2 python3 drone_icl/train.py \
    --data data/train_final.h5 \
    --val-data data/val_final.h5 \
    --epochs 150 --batch-size 128 --max-seq-len 400 --num-demos 3 \
    --checkpoint-dir checkpoints/exp3 --log-dir logs \
    --lr 5e-5 --weight-decay 1e-5 --num-workers 20 --save-freq 5 \
    --embed-dim 384 --device cuda \
    2>&1 | tee logs/exp3.log"

# Experiment 4: Higher regularization (embed=256, batch=256, seq=400, demos=3, wd=1e-4)
tmux new-session -d -s exp4 "CUDA_VISIBLE_DEVICES=3 python3 drone_icl/train.py \
    --data data/train_final.h5 \
    --val-data data/val_final.h5 \
    --epochs 150 --batch-size 256 --max-seq-len 400 --num-demos 3 \
    --checkpoint-dir checkpoints/exp4 --log-dir logs \
    --lr 5e-5 --weight-decay 1e-4 --num-workers 20 --save-freq 5 \
    --embed-dim 256 --device cuda \
    2>&1 | tee logs/exp4.log"

echo "Started 4 experiments in tmux sessions"
echo "List sessions: tmux ls"
echo "Attach to session: tmux attach -t exp1  (or exp2/3/4)"
echo "Monitor logs: tail -f logs/exp*.log"

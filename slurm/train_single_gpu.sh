#!/bin/bash
# Single A100 40GB training

mkdir -p checkpoints/baseline logs

CUDA_VISIBLE_DEVICES=0 python3 drone_icl/train.py \
    --data data/train_final.h5 \
    --val-data data/val_final.h5 \
    --epochs 150 \
    --batch-size 256 \
    --max-seq-len 512 \
    --num-demos 3 \
    --checkpoint-dir checkpoints/baseline \
    --log-dir logs \
    --lr 1e-4 \
    --weight-decay 1e-5 \
    --num-workers 4 \
    --save-freq 5 \
    --embed-dim 256 \
    --device cuda \
    2>&1 | tee logs/train_baseline.log

#!/bin/bash

python drone_icl/train.py \
    --data data/train_final.h5 \
    --val-data data/val_final.h5 \
    --epochs 100 \
    --batch-size 256 \
    --max-seq-len 512 \
    --num-demos 3 \
    --checkpoint-dir checkpoints \
    --log-dir logs \
    --lr 1e-4 \
    --weight-decay 1e-5 \
    --num-workers 4 \
    --embed-dim 256 \
    --device cuda

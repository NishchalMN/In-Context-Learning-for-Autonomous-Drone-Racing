#!/bin/bash

if [ -z "$1" ]; then
    DEVICE=$(python -c "import torch; print('cuda' if torch.cuda.is_available() else 'cpu')" 2>/dev/null || echo "cpu")
else
    DEVICE=$1
fi

if [ "$2" == "eval" ]; then
    python scripts/evaluate.py \
        --checkpoint checkpoints/best_model.pt \
        --data data/val_final.h5 \
        --output-dir results \
        --num-demos 1 3 \
        --device $DEVICE
else
    python scripts/create_demo_video.py \
        --checkpoint checkpoints/best_model.pt \
        --data data/val_final.h5 \
        --track ellipse \
        --num-demos 1 \
        --output-dir results/videos \
        --device $DEVICE
fi

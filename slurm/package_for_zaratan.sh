#!/bin/bash
# Package files for Zaratan training

echo "Creating zaratan_training.tar.gz..."

tar -czf zaratan_training.tar.gz \
  drone_icl/*.py \
  data/train_final_gates.h5 \
  data/val_final_gates.h5 \
  --exclude="__pycache__" \
  --exclude="*.pyc"

echo "Done! Transfer with:"
echo "scp zaratan_training.tar.gz zaratan:~/"
ls -lh zaratan_training.tar.gz

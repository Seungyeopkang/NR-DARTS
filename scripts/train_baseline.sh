#!/bin/bash
set -e

echo "========================================================"
echo "INFO: Starting Baseline Model Training..."
echo "========================================================"

SEED=42
EPOCHS=100
BATCH_SIZE=64
LR=0.025
DROP_PROB=0.3
SAVE_PATH="exp/baseline_seed${SEED}"


mkdir -p $SAVE_PATH



python train/train_baseline.py \
    --save $SAVE_PATH \
    --seed $SEED \
    --epochs $EPOCHS

echo "========================================================"
echo "Baseline Training Finished!"
echo "Best model and logs saved in: ${SAVE_PATH}"
echo "========================================================"
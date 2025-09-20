#!/bin/bash
set -e

echo "========================================================"
echo "INFO: Starting Node Importance Search (Baseline Params)"
echo "========================================================"

SEED=42
EPOCHS=120
BATCH_SIZE=64
LR=0.025
DROP_PROB=0.3
SAVE_PATH="exp/search_seed${SEED}"


mkdir -p $SAVE_PATH


python search/train_search.py \
    --save $SAVE_PATH \
    --seed $SEED \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate $LR \
    --drop_path_prob $DROP_PROB \
    --cutout

echo "========================================================"
echo "Node Importance Search Finished!"
echo "Best model saved in: ${SAVE_PATH}"
echo "========================================================"
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

DATA=/teamspace/studios/this_studio/figshare_37ec464af8e81ae6ebbf
EPOCHS=150

run_train () {
  local NAME="$1"; shift
  local DIR="outputs/ablation_${NAME}"
  mkdir -p "$DIR"
  echo "[$(date +%H:%M:%S)] === train ${NAME} ==="
  python -u train.py \
    --dataset isbi2015 \
    --dataset-root "$DATA" \
    --backbone convnextv2_tiny \
    --image-size 1024 --batch-size 2 \
    --epochs "$EPOCHS" --lr 2e-4 \
    --num-workers 2 \
    --pixel-size-mm 0.1 \
    --output-dir "$DIR" \
    "$@" \
    > "$DIR/train.log" 2>&1
}

run_eval () {
  local NAME="$1"; shift
  local DIR="outputs/ablation_${NAME}"
  for SPLIT in test1 test2; do
    echo "[$(date +%H:%M:%S)] === eval ${NAME} on ${SPLIT} ==="
    python -u infer.py \
      --dataset isbi2015 \
      --dataset-root "$DATA" \
      --split "$SPLIT" \
      --backbone convnextv2_tiny --image-size 1024 --batch-size 1 \
      --pixel-size-mm 0.1 \
      --checkpoint "$DIR/best_model.pt" \
      --output "$DIR/${SPLIT}_predictions.json" \
      "$@" \
      2>&1 | tee -a "$DIR/eval.log"
  done
}

run_train no_heatmap   --no-heatmap
run_eval  no_heatmap

run_train no_finetune  --no-finetune
run_eval  no_finetune  --no-finetune

run_train no_rle       --no-rle
run_eval  no_rle

run_train m1           --num-finetune-layers 1
run_eval  m1           --num-finetune-layers 1

echo "[$(date +%H:%M:%S)] all ablations done"

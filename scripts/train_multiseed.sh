#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/train_multiseed.sh \
#     /path/1536/data.jsonl /path/1536/images \
#     /path/new/data.jsonl /path/new/image \
#     /path/artifacts

if [[ $# -lt 5 ]]; then
  echo "Usage: $0 <train_jsonl_1536> <train_img_1536> <train_jsonl_new> <train_img_new> <artifact_root>"
  exit 1
fi

TRAIN_JSONL_1536="$1"
TRAIN_IMG_1536="$2"
TRAIN_JSONL_NEW="$3"
TRAIN_IMG_NEW="$4"
ARTIFACT_ROOT="$5"

SEEDS=(42 2024 3407)

for s in "${SEEDS[@]}"; do
  echo "[Run] seed=${s}"
  python -u scripts/train.py \
    --dataset train1536:"${TRAIN_JSONL_1536}":"${TRAIN_IMG_1536}" \
    --dataset validNew:"${TRAIN_JSONL_NEW}":"${TRAIN_IMG_NEW}" \
    --artifact-dir "${ARTIFACT_ROOT}/seed_${s}" \
    --n-splits 5 \
    --seed "${s}" \
    --use-handcrafted 1 \
    --use-clip 1 \
    --use-pyiqa 0 \
    --trusted-source train1536 \
    --noisy-source validNew \
    --denoise-action flip \
    --denoise-threshold 0.85 \
    --trusted-weight 1.0 \
    --noisy-weight 0.6 \
    --flipped-weight 0.8 \
    --use-lr 1 \
    --use-hgb 1 \
    --use-xgboost 1 \
    --device cuda \
    --verbose 1
  echo "[Done] seed=${s}"
done

echo "[OK] all seeds done"

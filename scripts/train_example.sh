#!/usr/bin/env bash
set -euo pipefail

python scripts/train.py \
  --dataset train1536:/Users/comefly/Desktop/compeitions/others/IQA/1536/data.jsonl:/Users/comefly/Desktop/compeitions/others/IQA/1536/images \
  --dataset validNew:/Users/comefly/Desktop/compeitions/others/IQA/new/data.jsonl:/Users/comefly/Desktop/compeitions/others/IQA/new/image \
  --artifact-dir artifacts/run_clip_pairwise \
  --n-splits 5 \
  --seed 42 \
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
  --stratify-by-source 1 \
  --use-lr 1 \
  --use-hgb 1 \
  --use-xgboost 1 \
  --verbose 1

#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <TEST_JSONL> <TEST_IMAGE_ROOT> [OUTPUT_JSONL]"
  exit 1
fi

TEST_JSONL="$1"
TEST_IMG_ROOT="$2"
OUT="${3:-predictions.jsonl}"

python scripts/predict.py \
  --model-bundle artifacts/run_clip_pairwise/model_bundle.joblib \
  --dataset test:"${TEST_JSONL}":"${TEST_IMG_ROOT}" \
  --output "${OUT}" \
  --output-style competition \
  --with-thinking 1 \
  --include-prob 0

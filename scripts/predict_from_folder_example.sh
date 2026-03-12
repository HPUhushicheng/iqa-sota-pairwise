#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/predict_from_folder_example.sh <MODEL_BUNDLE> <TEST_DIR> [OUT_JSONL]

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <MODEL_BUNDLE> <TEST_DIR> [OUT_JSONL]"
  exit 1
fi

MODEL_BUNDLE="$1"
TEST_DIR="$2"
OUT_JSONL="${3:-predictions_from_folder.jsonl}"

python -u scripts/predict_from_folder.py \
  --model-bundle "${MODEL_BUNDLE}" \
  --test-dir "${TEST_DIR}" \
  --output "${OUT_JSONL}" \
  --output-style answer \
  --swap-tta 1 \
  --include-prob 0 \
  --strict 1

echo "[OK] wrote ${OUT_JSONL}"

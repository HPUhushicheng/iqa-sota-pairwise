#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/predict_ensemble_from_folder_example.sh <TEST_DIR> [OUT_JSONL]

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <TEST_DIR> [OUT_JSONL]"
  exit 1
fi

TEST_DIR="$1"
OUT_JSONL="${2:-predictions_ensemble_from_folder.jsonl}"

python -u scripts/predict_ensemble_from_folder.py \
  --model-bundle /root/autodl-tmp/artifacts/run_clip_pairwise/model_bundle.joblib \
  --model-bundle /root/autodl-tmp/artifacts/run_clip_pairwise_v2/model_bundle.joblib \
  --model-bundle /root/autodl-tmp/artifacts/sweep_denoise/flip_t065/model_bundle.joblib \
  --test-dir "${TEST_DIR}" \
  --output "${OUT_JSONL}" \
  --output-style answer \
  --swap-tta 1 \
  --include-prob 0 \
  --strict 1 \
  --device cuda

echo "[OK] wrote ${OUT_JSONL}"

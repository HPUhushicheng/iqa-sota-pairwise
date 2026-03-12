#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/predict_ensemble_example.sh \
#     /path/test/data.jsonl /path/test/images /path/out.jsonl [artifact_root]

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <test_jsonl> <test_image_root> <output_jsonl> [artifact_root]"
  exit 1
fi

TEST_JSONL="$1"
TEST_IMG_ROOT="$2"
OUT_JSONL="$3"
ARTIFACT_ROOT="${4:-artifacts/multiseed}"

python -u scripts/predict_ensemble.py \
  --model-bundle "${ARTIFACT_ROOT}/seed_42/model_bundle.joblib" \
  --model-bundle "${ARTIFACT_ROOT}/seed_2024/model_bundle.joblib" \
  --model-bundle "${ARTIFACT_ROOT}/seed_3407/model_bundle.joblib" \
  --dataset test:"${TEST_JSONL}":"${TEST_IMG_ROOT}" \
  --output "${OUT_JSONL}" \
  --output-style competition \
  --with-thinking 1 \
  --include-prob 0

echo "[OK] wrote ${OUT_JSONL}"

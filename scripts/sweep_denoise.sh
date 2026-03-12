#!/usr/bin/env bash
set -u -o pipefail

# Sweep denoise strategy for noisy-source training.
#
# Usage:
#   bash scripts/sweep_denoise.sh \
#     <train1536_jsonl> <train1536_images> \
#     <validNew_jsonl> <validNew_images> \
#     <artifact_root>
#
# Optional env vars:
#   DEVICE=cuda
#   SEED=42
#   N_SPLITS=5
#   USE_XGBOOST=1
#   CLIP_MODEL=vit_base_patch32_clip_224.openai

if [[ $# -lt 5 ]]; then
  echo "Usage: $0 <train1536_jsonl> <train1536_images> <validNew_jsonl> <validNew_images> <artifact_root>"
  exit 1
fi

TRAIN1536_JSONL="$1"
TRAIN1536_IMG="$2"
VALIDNEW_JSONL="$3"
VALIDNEW_IMG="$4"
ARTIFACT_ROOT="$5"

DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-42}"
N_SPLITS="${N_SPLITS:-5}"
USE_XGBOOST="${USE_XGBOOST:-1}"
CLIP_MODEL="${CLIP_MODEL:-vit_base_patch32_clip_224.openai}"

mkdir -p "${ARTIFACT_ROOT}"
LEADERBOARD_CSV="${ARTIFACT_ROOT}/leaderboard.csv"

cat > "${LEADERBOARD_CSV}" <<CSV
config,action,denoise_threshold,noisy_weight,flipped_weight,bal_acc,acc,method,model_threshold,status,artifact_dir
CSV

# config_name|action|denoise_threshold|noisy_weight|flipped_weight
CONFIGS=(
  "baseline_none|none|0.85|1.0|1.0"
  "drop_t065|drop|0.65|1.0|1.0"
  "drop_t070|drop|0.70|1.0|1.0"
  "drop_t075|drop|0.75|1.0|1.0"
  "flip_t065|flip|0.65|0.9|1.0"
  "flip_t070|flip|0.70|0.9|1.0"
  "flip_t075|flip|0.75|0.9|1.0"
)

echo "[Sweep] total configs: ${#CONFIGS[@]}"
echo "[Sweep] artifact root: ${ARTIFACT_ROOT}"

for cfg in "${CONFIGS[@]}"; do
  IFS='|' read -r NAME ACTION DENOISE_T NOISY_W FLIPPED_W <<< "${cfg}"
  RUN_DIR="${ARTIFACT_ROOT}/${NAME}"
  LOG_PATH="${RUN_DIR}/train.log"

  mkdir -p "${RUN_DIR}"
  echo ""
  echo "[Run] ${NAME} | action=${ACTION} denoise_t=${DENOISE_T} noisy_w=${NOISY_W} flipped_w=${FLIPPED_W}"

  set +e
  python -u scripts/train.py \
    --dataset train1536:"${TRAIN1536_JSONL}":"${TRAIN1536_IMG}" \
    --dataset validNew:"${VALIDNEW_JSONL}":"${VALIDNEW_IMG}" \
    --artifact-dir "${RUN_DIR}" \
    --n-splits "${N_SPLITS}" \
    --seed "${SEED}" \
    --use-handcrafted 1 \
    --use-clip 1 \
    --clip-model "${CLIP_MODEL}" \
    --use-pyiqa 0 \
    --trusted-source train1536 \
    --noisy-source validNew \
    --denoise-action "${ACTION}" \
    --denoise-threshold "${DENOISE_T}" \
    --trusted-weight 1.0 \
    --noisy-weight "${NOISY_W}" \
    --flipped-weight "${FLIPPED_W}" \
    --stratify-by-source 1 \
    --use-lr 1 \
    --use-hgb 1 \
    --use-xgboost "${USE_XGBOOST}" \
    --device "${DEVICE}" \
    --verbose 1 \
    2>&1 | tee "${LOG_PATH}"
  RC=$?
  set -e

  REPORT_JSON="${RUN_DIR}/cv_report.json"
  if [[ ${RC} -ne 0 || ! -f "${REPORT_JSON}" ]]; then
    echo "[Run] ${NAME} failed (rc=${RC})"
    echo "${NAME},${ACTION},${DENOISE_T},${NOISY_W},${FLIPPED_W},,,,,failed,${RUN_DIR}" >> "${LEADERBOARD_CSV}"
    continue
  fi

  METRICS=$(python - <<PY
import json
path = "${REPORT_JSON}"
obj = json.load(open(path, "r", encoding="utf-8"))
b = obj.get("blended_oof", {})
print(",".join([
  str(b.get("bal_acc", "")),
  str(b.get("acc", "")),
  str(b.get("method", "")),
  str(b.get("threshold", "")),
]))
PY
)

  BAL_ACC="$(echo "${METRICS}" | cut -d',' -f1)"
  ACC="$(echo "${METRICS}" | cut -d',' -f2)"
  METHOD="$(echo "${METRICS}" | cut -d',' -f3)"
  MODEL_TH="$(echo "${METRICS}" | cut -d',' -f4)"

  echo "[Run] ${NAME} done | bal_acc=${BAL_ACC} acc=${ACC} method=${METHOD}"
  echo "${NAME},${ACTION},${DENOISE_T},${NOISY_W},${FLIPPED_W},${BAL_ACC},${ACC},${METHOD},${MODEL_TH},ok,${RUN_DIR}" >> "${LEADERBOARD_CSV}"
done

echo ""
echo "[Sweep] leaderboard: ${LEADERBOARD_CSV}"

python - <<PY
import csv
path = "${LEADERBOARD_CSV}"
rows = list(csv.DictReader(open(path, "r", encoding="utf-8")))
ok = [r for r in rows if r.get("status") == "ok" and r.get("bal_acc") not in ("", None)]
if not ok:
    print("[Sweep] no successful run")
    raise SystemExit(0)

ok.sort(key=lambda r: float(r["bal_acc"]), reverse=True)
print("[Sweep] Top-3:")
for r in ok[:3]:
    print(
        f"  - {r['config']}: bal_acc={float(r['bal_acc']):.4f}, "
        f"acc={float(r['acc']):.4f}, method={r['method']}, "
        f"action={r['action']}, denoise_t={r['denoise_threshold']}"
    )

best = ok[0]
print("[Sweep] Best config:")
print(
    f"  {best['config']} | action={best['action']} | denoise_t={best['denoise_threshold']} "
    f"| noisy_w={best['noisy_weight']} | flipped_w={best['flipped_weight']} | "
    f"bal_acc={float(best['bal_acc']):.4f}"
)
print(f"[Sweep] Best artifacts: {best['artifact_dir']}")
PY

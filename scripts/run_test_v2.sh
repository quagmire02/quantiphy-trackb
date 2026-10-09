#!/usr/bin/env bash
# Test submission from cached parse/ground/track: re-solve (CPU), VLM answers for unsolved items (GPU,
# resumes), build the upload file.
# usage: bash scripts/run_test_v2.sh [solutions_name] [submission_name]
# Model / results tree follow QUANTIPHY_VLM and QUANTIPHY_RESULTS (see pipeline/vlm.py, pipeline/scoring.py).
set -euo pipefail
cd "$(dirname "$0")/.."
PY="$HOME/.venvs/quantiphy/bin/python"
RES="${QUANTIPHY_RESULTS:-results}"
SOL="${1:-measure_v3}"
NAME="${2:-submission_v2}"
FB="$("$PY" -c 'from pipeline.vlm import vlm_short_name; print(vlm_short_name())')_video_f16_px448"
mkdir -p "$RES/logs"
LOG="$RES/logs/test_run_${NAME}.log"
"$PY" experiments/measure_2d.py --split test --categories S2,D2,S3,D3 --stages solve --name "$SOL" \
  --fallback none 2>&1 | tee "$LOG"
PYTHONUNBUFFERED=1 "$PY" experiments/baseline_vlm.py --split test --mode video --max-frames 16 --longest 448 \
  --unsolved-from "$RES/cache/test/${SOL}_solutions.csv" 2>&1 \
  | stdbuf -oL tr '\r' '\n' | grep --line-buffered -v -E "it/s\]$|s/it\]$|Loading weights" | tee -a "$LOG"
"$PY" experiments/make_submission.py --split test --solutions "$RES/cache/test/${SOL}_solutions.csv" \
  --fallback "$FB" --name "$NAME" 2>&1 | tee -a "$LOG"
echo "DONE" | tee -a "$LOG"

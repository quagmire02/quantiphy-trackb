#!/usr/bin/env bash
# v2 test submission from the cached parse/ground/track of the v1 run:
# re-solve with the fixed solver + guards (CPU), VLM for newly unsolved items (GPU, resumes), build file.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="$HOME/.venvs/quantiphy/bin/python"
LOG="results/logs/test_run_${2:-submission_v2}.log"
# usage: bash scripts/run_test_v2.sh [solutions_name] [submission_name]
SOL="${1:-measure_v3}"
NAME="${2:-submission_v2}"
"$PY" experiments/measure_2d.py --split test --categories S2,D2,S3,D3 --stages solve --name "$SOL" \
  --fallback none 2>&1 | tee "$LOG"
PYTHONUNBUFFERED=1 "$PY" experiments/baseline_vlm.py --split test --mode video --max-frames 16 --longest 448 \
  --unsolved-from "results/cache/test/${SOL}_solutions.csv" 2>&1 \
  | stdbuf -oL tr '\r' '\n' | grep --line-buffered -v -E "it/s\]$|s/it\]$|Loading weights" | tee -a "$LOG"
"$PY" experiments/make_submission.py --split test --solutions "results/cache/test/${SOL}_solutions.csv" \
  --fallback Qwen3-VL-4B_video_f16_px448 --name "$NAME" 2>&1 | tee -a "$LOG"
echo "DONE" | tee -a "$LOG"

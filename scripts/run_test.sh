#!/usr/bin/env bash
# Full test-set run (resumable: rerunning skips cached parse/ground/track and finished VLM items).
#   1. measurement pipeline on all categories -> results/cache/test/measure_solutions.csv
#   2. VLM answers only for items the pipeline could not solve
#   3. merged + validated submission -> results/submissions/<name>.csv
# usage: bash scripts/run_test.sh [vlm_longest_px] [submission_name]
set -uo pipefail
cd "$(dirname "$0")/.."
PX="${1:-448}"
NAME="${2:-submission_v1}"
PY="$HOME/.venvs/quantiphy/bin/python"
LOG=results/logs/test_run.log
mkdir -p results/logs
stamp() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

stamp "stage 1: measurement pipeline (parse, ground, track, solve)"
bash scripts/run_measure.sh test_measure --split test --categories S2,D2,S3,D3 --name measure --fallback none
tail -n 12 results/logs/test_measure.log | tee -a "$LOG"

stamp "stage 2: VLM fallback (16 frames, ${PX}px) on unsolved items"
PYTHONUNBUFFERED=1 "$PY" experiments/baseline_vlm.py --split test --mode video --max-frames 16 --longest "$PX" \
  --unsolved-from results/cache/test/measure_solutions.csv 2>&1 \
  | stdbuf -oL tr '\r' '\n' | grep --line-buffered -v -E "it/s\]$|s/it\]$|Loading weights" >> "$LOG"

stamp "stage 3: submission"
"$PY" experiments/make_submission.py --split test --solutions results/cache/test/measure_solutions.csv \
  --fallback "Qwen3-VL-4B_video_f16_px${PX}" --name "$NAME" 2>&1 | tee -a "$LOG"
stamp "DONE"

#!/usr/bin/env bash
# Complete pipeline from scratch into its own results tree (for a different VLM, e.g. on the lab PC).
#
#   QUANTIPHY_VLM=Qwen/Qwen3-VL-8B-Instruct QUANTIPHY_RESULTS=results_8b bash scripts/run_full.sh submission_8b
#   (6 GB GPU: also set QUANTIPHY_VLM_QUANT_VISION=1; big GPU: QUANTIPHY_VLM_4BIT=0 for full precision)
#
# Stages: parse, ground (VLM) -> track (SAM 2.1) -> findmissing (tiled grounding) -> anchors (VLM + SAM, 3D)
# -> refix (VLM + SAM, motion priors) -> refine (native-resolution sizes) -> geom (MoGe-2, 3D) -> mogecal
# (label-free MoGe focal-length correction) -> solve (CPU) on validation and test; VLM fallback for unsolved
# items; validation score; test upload file. Use a Qwen3-VL model: grounding reads Qwen3-VL's 0-1000 box
# coordinates (Qwen2.5-VL answers in pixels of its resized input, which this code would misplace).
# Resumable: every stage caches per video; rerun the same command after an interruption.
set -uo pipefail
cd "$(dirname "$0")/.."
PY="$HOME/.venvs/quantiphy/bin/python"
NAME="${1:?usage: run_full.sh <submission_name>}"
export QUANTIPHY_RESULTS="${QUANTIPHY_RESULTS:-results}"
RES="$QUANTIPHY_RESULTS"
mkdir -p "$RES/logs" "$RES/test_predictions" "$RES/validation_predictions"
# Last-resort answers (per-unit median) are model independent: reuse the main tree's.
for s in test validation; do
  for f in results/${s}_predictions/floor_unit_median*.csv; do
    [ -e "$f" ] && [ ! -e "$RES/${s}_predictions/$(basename "$f")" ] && cp "$f" "$RES/${s}_predictions/"
  done
done
LOG="$RES/logs/run_full_${NAME}.log"
FB="$("$PY" -c 'from pipeline.vlm import vlm_short_name; print(vlm_short_name())')_video_f16_px448"
filter() { stdbuf -oL tr '\r' '\n' | grep --line-buffered -v -E "it/s\]$|s/it\]$|Loading weights|Warning"; }

for s in validation test; do
  PYTHONUNBUFFERED=1 "$PY" experiments/measure_2d.py --split "$s" --categories S2,D2,S3,D3 \
    --stages parse,ground,track,findmissing,anchors,refix,refine,geom --fallback none 2>&1 | filter | tee -a "$LOG"
done
"$PY" experiments/measure_2d.py --split test --categories S3,D3 --stages mogecal --fallback none 2>&1 | filter | tee -a "$LOG"
for s in validation test; do
  "$PY" experiments/measure_2d.py --split "$s" --categories S2,D2,S3,D3 --stages solve --name measure \
    --fallback none 2>&1 | filter | tee -a "$LOG"
  PYTHONUNBUFFERED=1 "$PY" experiments/baseline_vlm.py --split "$s" --mode video --max-frames 16 --longest 448 \
    --unsolved-from "$RES/cache/$s/measure_solutions.csv" 2>&1 | filter | tee -a "$LOG"
done
"$PY" experiments/make_submission.py --split validation --solutions "$RES/cache/validation/measure_solutions.csv" \
  --fallback "$FB" --name "val_${NAME}" 2>&1 | filter | tee -a "$LOG"
"$PY" experiments/make_submission.py --split test --solutions "$RES/cache/test/measure_solutions.csv" \
  --fallback "$FB" --name "$NAME" 2>&1 | filter | tee -a "$LOG"
echo "DONE" | tee -a "$LOG"

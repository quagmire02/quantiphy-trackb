#!/usr/bin/env bash
# Run both official zero-shot settings of the local VLM baseline on validation.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="$HOME/.venvs/quantiphy/bin/python"
rm -f results/validation_raw/smoke.jsonl results/validation_predictions/smoke.csv
"$PY" experiments/baseline_vlm.py --split validation --mode video 2>&1 | grep -v -E "Loading weights|videos:"
"$PY" experiments/baseline_vlm.py --split validation --mode no_video 2>&1 | grep -v -E "Loading weights|videos:"
echo "ALL DONE"

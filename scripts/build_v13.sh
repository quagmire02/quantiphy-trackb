#!/usr/bin/env bash
# v13: v12 + question-meaning rules (horizontal/vertical motion component in 2D, wingspan = long axis).
set -uo pipefail
cd "$(dirname "$0")/.."
PY="$HOME/.venvs/quantiphy/bin/python"
"$PY" experiments/measure_2d.py --split validation --categories S2,D2,S3,D3 --stages solve --name measure_v22 --fallback none > /dev/null 2>&1 || echo "VAL SOLVE FAILED"
"$PY" experiments/make_submission.py --split validation --solutions results/cache/validation/measure_v22_solutions.csv \
  --fallback Qwen3-VL-4B_video_f16_px448 --name val_measure_v22 2>&1 | grep -E "val_measure_v(21|22) |model "
bash scripts/run_test_v2.sh measure_v22 submission_v13 2>&1 | grep -E "solved|wrote|All |NOT WRITTEN|Error|Traceback|to run|DONE"
"$PY" scripts/diff_subs.py submission_v12 submission_v13

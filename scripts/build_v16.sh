#!/usr/bin/env bash
# v16: v15 + 2026-10-09 dataset (reworded questions, category fixes) + lab camera prior (84 deg).
set -uo pipefail
cd "$(dirname "$0")/.."
PY="$HOME/.venvs/quantiphy/bin/python"
"$PY" experiments/measure_2d.py --split validation --categories S2,D2,S3,D3 --stages solve --name measure_v24 --fallback none > /dev/null 2>&1 || echo "VAL SOLVE FAILED"
"$PY" experiments/make_submission.py --split validation --solutions results/cache/validation/measure_v24_solutions.csv \
  --fallback Qwen3-VL-4B_video_f16_px448 --name val_measure_v24 2>&1 | grep -E "val_measure_v2[345] |model "
bash scripts/run_test_v2.sh measure_v24 submission_v16 2>&1 | grep -E "solved|wrote|All |NOT WRITTEN|Error|Traceback|to run|DONE"
"$PY" scripts/diff_subs.py submission_v15 submission_v16

#!/usr/bin/env bash
# v12: v11 + MoGe-2 focal length bias-corrected per footage group (results/cache/moge_f_correction.json).
set -uo pipefail
cd "$(dirname "$0")/.."
PY="$HOME/.venvs/quantiphy/bin/python"
"$PY" experiments/measure_2d.py --split validation --categories S2,D2,S3,D3 --stages solve --name measure_v21 --fallback none > /dev/null 2>&1 || echo "VAL SOLVE FAILED"
"$PY" experiments/make_submission.py --split validation --solutions results/cache/validation/measure_v21_solutions.csv \
  --fallback Qwen3-VL-4B_video_f16_px448 --name val_measure_v21 2>&1 | grep -E "val_measure_v(20|21) |model "
bash scripts/run_test_v2.sh measure_v21 submission_v12 2>&1 | grep -E "solved|wrote|All |NOT WRITTEN|Error|Traceback|to run|DONE"
"$PY" scripts/diff_subs.py submission_v11 submission_v12

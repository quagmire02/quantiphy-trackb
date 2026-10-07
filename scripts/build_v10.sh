#!/usr/bin/env bash
# v10: v9 + 3D target accelerations over the constant-acceleration phase. Validation score, then the test submission.
set -uo pipefail
cd "$(dirname "$0")/.."
PY="$HOME/.venvs/quantiphy/bin/python"
"$PY" experiments/measure_2d.py --split validation --categories S2,D2,S3,D3 --stages solve --name measure_v19 --fallback none > /dev/null 2>&1 || echo "VAL SOLVE FAILED"
"$PY" experiments/make_submission.py --split validation --solutions results/cache/validation/measure_v19_solutions.csv \
  --fallback Qwen3-VL-4B_video_f16_px448 --name val_measure_v19 2>&1 | grep -E "val_measure_v(18|19) |model "
bash scripts/run_test_v2.sh measure_v19 submission_v10 2>&1 | grep -E "solved|wrote|All |NOT WRITTEN|Error|Traceback|to run|DONE"
"$PY" scripts/diff_subs.py submission_v9 submission_v10

#!/usr/bin/env bash
# v8: 3D MoGe-2 focal-length fallback. Validation score, then the test submission.
set -uo pipefail
cd "$(dirname "$0")/.."
PY="$HOME/.venvs/quantiphy/bin/python"
"$PY" experiments/measure_2d.py --split validation --categories S2,D2,S3,D3 --stages solve --name measure_v16 --fallback none > /dev/null 2>&1 || echo "VAL SOLVE FAILED"
"$PY" experiments/make_submission.py --split validation --solutions results/cache/validation/measure_v16_solutions.csv \
  --fallback Qwen3-VL-4B_video_f16_px448 --name val_measure_v16 2>&1 | grep -E "val_measure_v1[56] |model "
bash scripts/run_test_v2.sh measure_v16 submission_v8 2>&1 | grep -E "solved|wrote|All |NOT WRITTEN|Error|Traceback|to run|DONE"

#!/usr/bin/env bash
# v14: v13 + label-free last resort (median of our own answers per unit, replaces the validation-label
# median) + score-optimal answer correction, only if it wins leave-one-video-out validation.
set -uo pipefail
cd "$(dirname "$0")/.."
PY="$HOME/.venvs/quantiphy/bin/python"
FB=Qwen3-VL-4B_video_f16_px448
echo "== validation, new last resort, no correction"
"$PY" experiments/make_submission.py --split validation --solutions results/cache/validation/measure_v22_solutions.csv \
  --fallback "$FB" --name val_measure_v23 2>&1 | grep -E "val_measure_v2[23] |model |median"
echo "== how far from the answers + corrections"
"$PY" experiments/calibrate_answers.py val_measure_v23 2>&1 | grep -v -i warning
echo "== validation with the correction (in-sample, for reference only)"
"$PY" experiments/make_submission.py --split validation --solutions results/cache/validation/measure_v22_solutions.csv \
  --fallback "$FB" --name val_measure_v24 --correct 2>&1 | grep -E "val_measure_v2[234] |model |applied"
echo "== test submission v14"
"$PY" experiments/make_submission.py --split test --solutions results/cache/test/measure_v22_solutions.csv \
  --fallback "$FB" --name submission_v14 --correct 2>&1 | grep -E "wrote|All |NOT WRITTEN|applied|Error|Traceback"
"$PY" scripts/diff_subs.py submission_v13 submission_v14 2>&1 | head -12

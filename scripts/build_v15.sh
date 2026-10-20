#!/usr/bin/env bash
# v15: v14 + equal-depth fallback (3D) + tiled grounding of missing objects + native-resolution size
# refinement of small objects. Validation score, test stages, test submission.
set -uo pipefail
cd "$(dirname "$0")/.."
PY="$HOME/.venvs/quantiphy/bin/python"
FB=Qwen3-VL-4B_video_f16_px448
"$PY" experiments/measure_2d.py --split validation --categories S2,D2,S3,D3 --stages solve --name measure_v23 --fallback none > /dev/null 2>&1 || echo "VAL SOLVE FAILED"
"$PY" experiments/make_submission.py --split validation --solutions results/cache/validation/measure_v23_solutions.csv \
  --fallback "$FB" --name val_measure_v25 2>&1 | grep -E "val_measure_v2[35] |model "
bash scripts/tighten_run.sh test
bash scripts/run_test_v2.sh measure_v23 submission_v15 2>&1 | grep -E "solved|wrote|All |NOT WRITTEN|Error|Traceback|to run|DONE"
"$PY" scripts/diff_subs.py submission_v14 submission_v15 2>&1 | head -30

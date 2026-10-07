#!/usr/bin/env bash
# v9: compound-noun distance matching + ground/track stated-distance objects for MoGe transfer (3D).
set -uo pipefail
cd "$(dirname "$0")/.."
PY="$HOME/.venvs/quantiphy/bin/python"
LOG=results/logs/build_v9.log
stamp=$(date +%Y%m%d_%H%M%S)
for s in validation test; do
  tar -czf "results/cache/${s}_anchors_backup_${stamp}.tgz" -C "results/cache/$s" ground tracks moge 2>/dev/null
  PYTHONUNBUFFERED=1 "$PY" experiments/measure_2d.py --split "$s" --categories S2,D2,S3,D3 \
    --stages anchors,geom --fallback none 2>&1 | stdbuf -oL tr '\r' '\n' \
    | grep --line-buffered -E "anchors |ERROR|Error|Traceback" | tee -a "$LOG"
done
"$PY" experiments/measure_2d.py --split validation --categories S2,D2,S3,D3 --stages solve --name measure_v18 --fallback none > /dev/null 2>&1 || echo "VAL SOLVE FAILED"
"$PY" experiments/make_submission.py --split validation --solutions results/cache/validation/measure_v18_solutions.csv \
  --fallback Qwen3-VL-4B_video_f16_px448 --name val_measure_v18 2>&1 | grep -E "val_measure_v1[68] |model " | tee -a "$LOG"
bash scripts/run_test_v2.sh measure_v18 submission_v9 2>&1 | grep -E "solved|wrote|All |NOT WRITTEN|Error|Traceback|to run|DONE" | tee -a "$LOG"
"$PY" scripts/diff_subs.py submission_v8 submission_v9 | tee -a "$LOG"

#!/usr/bin/env bash
# Re-ground clips with conflicts / missing objects under the current grounding rules, then re-track,
# re-run MoGe and re-solve. Backs up the affected caches first.
# usage: bash scripts/reground_run.sh <solutions_name>
set -uo pipefail
cd "$(dirname "$0")/.."
PY="$HOME/.venvs/quantiphy/bin/python"
NAME="$1"
LOG=results/logs/reground_${NAME}.log
stamp=$(date +%Y%m%d_%H%M%S)
for s in validation test; do
  tar -czf "results/cache/${s}_ground_backup_${stamp}.tgz" -C "results/cache/$s" ground tracks moge 2>/dev/null
  "$PY" scripts/invalidate_ground.py "$s" --apply | tee -a "$LOG"
  PYTHONUNBUFFERED=1 "$PY" experiments/measure_2d.py --split "$s" --categories S2,D2,S3,D3 \
    --stages ground,track,geom --fallback none 2>&1 | stdbuf -oL tr '\r' '\n' \
    | grep --line-buffered -E "frames,|ERROR|Error|Traceback" | tee -a "$LOG"
done
bash scripts/check_iteration.sh "$NAME" D2 measure_v12 2>&1 | tee -a "$LOG"
echo "DONE" | tee -a "$LOG"

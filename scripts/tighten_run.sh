#!/usr/bin/env bash
# This round on one split: back up caches, find missing objects (tiled grounding), refine small
# objects' sizes at native resolution, MoGe for re-tracked 3D clips, then evaluate.
# usage: bash scripts/tighten_run.sh <split>
set -uo pipefail
cd "$(dirname "$0")/.."
PY="$HOME/.venvs/quantiphy/bin/python"
S="${1:-validation}"
LOG="results/logs/tighten_${S}.log"
stamp=$(date +%Y%m%d_%H%M%S)
tar -czf "results/cache/${S}_tighten_backup_${stamp}.tgz" -C "results/cache/$S" ground tracks moge 2>/dev/null
PYTHONUNBUFFERED=1 "$PY" experiments/measure_2d.py --split "$S" --categories S2,D2,S3,D3 \
  --stages findmissing,refine,geom --fallback none 2>&1 | stdbuf -oL tr '\r' '\n' \
  | grep --line-buffered -E "findmissing |REFINE ERROR|GEOM ERROR|Error|Traceback" | tee -a "$LOG"
echo "refined clips: $(ls results/cache/$S/refine | wc -l)" | tee -a "$LOG"
"$PY" experiments/tighten_eval.py "$S" 2>&1 | grep -v -i warning | tee -a "$LOG"

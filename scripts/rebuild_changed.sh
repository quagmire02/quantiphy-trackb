#!/usr/bin/env bash
# Recompute every stage for clips whose caches were cleared (scripts/invalidate_changed.py); cached
# clips are skipped by each stage. Validation first, then test.
set -uo pipefail
cd "$(dirname "$0")/.."
PY="$HOME/.venvs/quantiphy/bin/python"
LOG=results/logs/rebuild_changed.log
for s in validation test; do
  PYTHONUNBUFFERED=1 "$PY" experiments/measure_2d.py --split "$s" --categories S2,D2,S3,D3 \
    --stages parse,ground,track,findmissing,anchors,refix,refine,geom --fallback none 2>&1 \
    | stdbuf -oL tr '\r' '\n' | grep --line-buffered -E "frames,|findmissing |anchors |refix |ERROR|Error|Traceback" | tee -a "$LOG"
  echo "== $s done" | tee -a "$LOG"
done

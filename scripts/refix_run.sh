#!/usr/bin/env bash
# Motion-guided re-grounding of static references (both splits), re-track, MoGe, then check + submission.
# usage: bash scripts/refix_run.sh <solutions_name> <submission_name>
set -uo pipefail
cd "$(dirname "$0")/.."
PY="$HOME/.venvs/quantiphy/bin/python"
NAME="$1"; SUB="$2"
LOG=results/logs/refix_${NAME}.log
for s in validation test; do
  PYTHONUNBUFFERED=1 "$PY" experiments/measure_2d.py --split "$s" --categories S2,D2,S3,D3 \
    --stages refix,geom --fallback none 2>&1 | stdbuf -oL tr '\r' '\n' \
    | grep --line-buffered -E "refix |ERROR|Error|Traceback" | tee -a "$LOG"
done
bash scripts/check_iteration.sh "$NAME" D2 measure_v13 2>&1 | tee -a "$LOG"
bash scripts/run_test_v2.sh "$NAME" "$SUB" 2>&1 | grep -E "wrote|All  |NOT WRITTEN|Error" | tee -a "$LOG"
echo "DONE" | tee -a "$LOG"

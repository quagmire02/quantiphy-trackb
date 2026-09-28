#!/usr/bin/env bash
# Run experiments/measure_2d.py with unbuffered output into results/logs/<name>.log
# usage: bash scripts/run_measure.sh <logname> [measure_2d.py args...]
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p results/logs
log="results/logs/$1.log"; shift
PYTHONUNBUFFERED=1 "$HOME/.venvs/quantiphy/bin/python" experiments/measure_2d.py "$@" 2>&1 \
  | stdbuf -oL tr '\r' '\n' | grep --line-buffered -v -E "it/s\]$|s/it\]$|Loading weights|propagate in video" > "$log"
echo "EXIT ${PIPESTATUS[0]}" >> "$log"

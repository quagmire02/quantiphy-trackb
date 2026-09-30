#!/usr/bin/env bash
# One test-and-fix iteration (CPU only): re-solve validation + test with the current code as
# solutions <name>, then print the validation score, the label-free free-fall physics check and the
# failure inventory of one category on test.
# usage: bash scripts/check_iteration.sh <name> [category] [baseline_name]
set -uo pipefail
cd "$(dirname "$0")/.."
PY="$HOME/.venvs/quantiphy/bin/python"
NAME="$1"; CAT="${2:-D2}"; BASE="${3:-measure_v8}"
for s in validation test; do
  "$PY" experiments/measure_2d.py --split "$s" --categories S2,D2,S3,D3 --stages solve --name "$NAME" --fallback none \
    > /dev/null 2>&1 || echo "SOLVE FAILED on $s"
done
"$PY" experiments/make_submission.py --split validation --solutions "results/cache/validation/${NAME}_solutions.csv" \
  --fallback Qwen3-VL-4B_video_f16_px448 --name "val_${NAME}" 2>&1 | grep -E "val_(${BASE}|${NAME}) |model "
"$PY" experiments/freefall_check.py --split test --solutions "$BASE" "$NAME"
"$PY" experiments/category_inventory.py --split test --category "$CAT" --solutions "$NAME" | sed -n '1,16p'

#!/usr/bin/env bash
# Lab PC (A6000): synthetic data -> LoRA adapter for Qwen3-VL-8B -> validation check of the backup answers,
# base model vs adapter (validation answers used for evaluation only, never for training).
#   bash scripts/synth_train.sh [n_clips] [model]
set -uo pipefail
cd "$(dirname "$0")/.."
PY="$HOME/.venvs/quantiphy/bin/python"
N="${1:-3000}"
MODEL="${2:-Qwen/Qwen3-VL-8B-Instruct}"
TAG="synth_$(basename "$MODEL" | tr 'A-Z' 'a-z' | sed 's/-instruct//')"
mkdir -p results/logs
LOG="results/logs/${TAG}.log"
"$PY" -m pip install -q peft 2>&1 | tail -1
[ -f data_synth/items.jsonl ] || "$PY" experiments/synth/generate.py --out data_synth --n-clips "$N" --seed 0 \
  --workers "$(nproc)" 2>&1 | tee -a "$LOG"
"$PY" experiments/synth/train_lora.py --model "$MODEL" --data data_synth --out "adapters/$TAG" 2>&1 \
  | grep -v -i warning | tee -a "$LOG"
# backup answers on validation, full precision: base model, then base + adapter
for adapter in "" "adapters/$TAG"; do
  QUANTIPHY_VLM_4BIT=0 QUANTIPHY_VLM_ADAPTER="$adapter" PYTHONUNBUFFERED=1 "$PY" experiments/baseline_vlm.py \
    --split validation --mode video --max-frames 16 --longest 448 --model "$MODEL" 2>&1 \
    | grep -v -i -E "warning|it/s\]|s/it\]|Loading weights" | tail -8 | tee -a "$LOG"
done
echo "DONE" | tee -a "$LOG"

#!/usr/bin/env bash
# Download Qwen3-VL-8B-Instruct (~17 GB) and run the smoke test with the 6 GB settings.
set -uo pipefail
cd "$(dirname "$0")/.."
PY="$HOME/.venvs/quantiphy/bin/python"
"$HOME/.venvs/quantiphy/bin/hf" download Qwen/Qwen3-VL-8B-Instruct 2>&1 | tail -2
QUANTIPHY_VLM=Qwen/Qwen3-VL-8B-Instruct QUANTIPHY_VLM_QUANT_VISION=1 "$PY" scripts/smoke_vlm.py 2>&1 \
  | grep -v -i -E "warning|Loading weights"

#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# Use the existing project environment if already active; otherwise create an isolated one.
if [ -z "${VIRTUAL_ENV:-}" ]; then
  python3 -m venv final_try/.venv
  source final_try/.venv/bin/activate
fi
python -m pip install --upgrade pip
python -m pip install 'torch==2.11.0' 'torchvision==0.26.0' --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r final_try/requirements.txt
python final_try/test_final_try.py
printf '%s\n' 'Ready. Activate final_try/.venv/bin/activate in future shells.'

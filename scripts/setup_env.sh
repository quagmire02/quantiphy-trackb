#!/usr/bin/env bash
# One-time WSL2 environment setup for QuantiPhy experiments.
set -euo pipefail

if ! command -v uv >/dev/null 2>&1 && [ ! -x "$HOME/.local/bin/uv" ]; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

VENV="$HOME/.venvs/quantiphy"
[ -d "$VENV" ] || uv venv --python 3.11 "$VENV"
source "$VENV/bin/activate"

uv pip install numpy pandas scipy tqdm matplotlib opencv-python-headless huggingface_hub pyarrow

python -c 'import cv2, pandas, numpy; print("cv2", cv2.__version__, "pandas", pandas.__version__, "numpy", numpy.__version__)'

# GPU / VLM stack (RTX 2060 = Turing sm75: fp16 only, no bf16 / FlashAttention-2)
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
uv pip install "transformers>=4.57" accelerate bitsandbytes "qwen-vl-utils>=0.0.14" av pillow

python - <<'EOF'
import torch, transformers, bitsandbytes
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
print("gpu", torch.cuda.get_device_name(0), "capability", torch.cuda.get_device_capability(0))
print("transformers", transformers.__version__, "bitsandbytes", bitsandbytes.__version__)
x = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
print("fp16 matmul ok", (x @ x).float().abs().mean().item() > 0)
EOF

#!/usr/bin/env bash
# Install MoGe (monocular geometry + camera FoV estimation, https://github.com/microsoft/MoGe) into the venv.
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
source "$HOME/.venvs/quantiphy/bin/activate"
uv pip install "git+https://github.com/microsoft/MoGe.git"
python - <<'EOF'
import inspect
from moge.model.v2 import MoGeModel
print("moge ok; infer signature:", inspect.signature(MoGeModel.infer))
EOF

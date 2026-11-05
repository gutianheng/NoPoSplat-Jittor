#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ "$(uname -s)" != Linux ]]; then
  echo 'This server script targets Linux. It does not change your Windows environments.' >&2
  exit 1
fi
"${NPS_PYTHON_BIN:-python3.10}" -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip freeze > environment_actual.txt
printf '%s\n' 'Environment prepared. Next: bash scripts/build_rasterizer.sh (set NPS_CUDA_ARCH first).'


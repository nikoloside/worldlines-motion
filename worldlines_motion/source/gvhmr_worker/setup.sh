#!/usr/bin/env bash
# Linux + NVIDIA GPU; needs uv and ffmpeg on PATH. Mirrors setup.ps1.
set -euo pipefail
cd "$(dirname "$0")"
uv venv --python 3.11 .venv
export VIRTUAL_ENV="$PWD/.venv"
uv pip install "torch==2.7.1" "torchvision==0.22.1" --index-url https://download.pytorch.org/whl/cu126
uv pip install pip setuptools wheel "numpy>=1.26"
uv pip install --no-build-isolation "chumpy==0.70"
uv pip install -r requirements.txt
uv pip install --no-deps "git+https://github.com/warmshao/WiLoR-mini"
export GVHMR_CHECKPOINTS="$PWD/weights" GVHMR_BODY_MODELS="$PWD/body_models"
.venv/bin/gvhmr download demo
mkdir -p body_models/smplx
echo "Done. Add body_models/smplx/SMPLX_NEUTRAL.npz (https://smpl-x.is.tue.mpg.de/, v1.1)."

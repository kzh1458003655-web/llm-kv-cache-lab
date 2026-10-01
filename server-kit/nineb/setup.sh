#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
python3 -c 'import sys; assert sys.version_info[:2] == (3, 12), "Python 3.12 required"'
nvidia-smi
python3 -m venv .venv-9b
.venv-9b/bin/python -m pip install --upgrade pip
.venv-9b/bin/python -m pip install --only-binary=:all: vllm==0.30.0 torch==2.13.0 transformers==5.17.0
.venv-9b/bin/python -c 'import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))'

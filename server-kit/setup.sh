#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
nvidia-smi
python3 -m venv .venv-server
.venv-server/bin/python -m pip install --upgrade pip
.venv-server/bin/python -m pip install 'vllm==0.30.0' 'torch==2.13.0' 'transformers==5.17.0'
.venv-server/bin/python -c 'import torch,vllm,transformers; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0),vllm.__version__,torch.__version__,transformers.__version__)'
echo 'Setup complete. Start the bounded run from server-kit/README.md.'

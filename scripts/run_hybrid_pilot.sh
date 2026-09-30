#!/usr/bin/env bash
# Run inside the existing WSL vLLM environment. Weights are not redistributed.
set -euo pipefail
: "${HYBRID_MODEL_DIR:?Set HYBRID_MODEL_DIR to the downloaded Qwen3.5-0.8B snapshot}"
exec python -m vllm.entrypoints.openai.api_server \
  --model "$HYBRID_MODEL_DIR" \
  --served-model-name hybrid-pilot --host 127.0.0.1 --port 8011 \
  --language-model-only --dtype bfloat16 --enforce-eager \
  --max-model-len 4096 --max-num-seqs 2 --max-num-batched-tokens 2048 \
  --gpu-memory-utilization 0.60 --kv-cache-memory-bytes 268435456 \
  --enable-prefix-caching --mamba-cache-mode align --enable-prompt-tokens-details

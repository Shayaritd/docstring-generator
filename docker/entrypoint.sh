#!/usr/bin/env bash
set -euo pipefail

echo "=== Docstring Generator container starting ==="
echo "BASE_MODEL=${BASE_MODEL:-<unset>}"
echo "ADAPTER_PATH=${ADAPTER_PATH:-<none, serving base model only>}"
echo "HF_HOME=${HF_HOME:-<unset>}"

if command -v nvidia-smi &> /dev/null; then
    echo "--- GPU status ---"
    nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader
else
    echo "WARNING: nvidia-smi not found inside container."
fi

if [ -d "${HF_HOME:-/model_cache}" ]; then
    cache_size=$(du -sh "${HF_HOME:-/model_cache}" 2>/dev/null | cut -f1 || echo "unknown")
    echo "Model cache at ${HF_HOME:-/model_cache}: ${cache_size}"
fi

export LOG_LEVEL="${LOG_LEVEL:-info}"

echo "=== Starting server ==="
exec "$@"
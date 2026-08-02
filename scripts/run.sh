#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

WITH_LOGGING=false
DETACH=false
for arg in "$@"; do
    case "$arg" in
        --with-logging) WITH_LOGGING=true ;;
        --detach|-d) DETACH=true ;;
        *) echo "Unknown argument: $arg" >&2; exit 1 ;;
    esac
done

echo "=== Pre-flight checks ==="

if [ ! -f ".env" ]; then
    echo "ERROR: .env not found. Copy .env.example to .env and fill in real values first:" >&2
    echo "  cp .env.example .env" >&2
    exit 1
fi

if $WITH_LOGGING && ! grep -q "^POSTGRES_PASSWORD=.\+" .env; then
    echo "ERROR: --with-logging requires POSTGRES_PASSWORD to be set in .env" >&2
    exit 1
fi

if command -v nvidia-smi &> /dev/null; then
    echo "GPU detected on host:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "WARNING: nvidia-smi not found on host."
fi

mkdir -p adapters
if [ -z "$(ls -A adapters 2>/dev/null)" ]; then
    echo "NOTE: ./adapters is empty — serving base model with no fine-tuning"
fi

echo ""
echo "=== Starting services ==="

COMPOSE_ARGS=(up)
$DETACH && COMPOSE_ARGS+=(-d)

if $WITH_LOGGING; then
    docker compose --profile logging "${COMPOSE_ARGS[@]}"
else
    docker compose "${COMPOSE_ARGS[@]}"
fi

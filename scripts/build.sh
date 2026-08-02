#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

TAG="${1:-latest}"
IMAGE_NAME="docstring-generator"

echo "=== Building ${IMAGE_NAME}:${TAG} ==="

if [ ! -f "docker/Dockerfile" ]; then
    echo "ERROR: docker/Dockerfile not found." >&2
    exit 1
fi

if ! command -v docker &> /dev/null; then
    echo "ERROR: docker command not found." >&2
    exit 1
fi

export DOCKER_BUILDKIT=1

docker build \
    -f docker/Dockerfile \
    -t "${IMAGE_NAME}:${TAG}" \
    -t "${IMAGE_NAME}:latest" \
    --progress=plain \
    .

echo ""
echo "=== Build complete: ${IMAGE_NAME}:${TAG} ==="
docker images "${IMAGE_NAME}" --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}"

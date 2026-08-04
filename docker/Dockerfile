# syntax=docker/dockerfile:1
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv curl \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3

FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3-pip \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3

WORKDIR /build
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY api/requirements_api.txt .
RUN pip install --upgrade pip && \
    pip install torch --index-url https://download.pytorch.org/whl/cu124 && \
    pip install -r requirements_api.txt

FROM base AS runtime

RUN useradd --create-home --shell /bin/bash appuser

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app
COPY api/app ./app
COPY training/load_model.py training/training_config.py ./training/
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENV HF_HOME=/model_cache \
    TRANSFORMERS_CACHE=/model_cache \
    BASE_MODEL=Qwen/Qwen2.5-Coder-1.5B \
    GENERATION_TIMEOUT_SECONDS=120 \
    LOG_LEVEL=info

RUN mkdir -p /model_cache /adapters && chown -R appuser:appuser /app /model_cache /adapters

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

ENTRYPOINT ["/entrypoint.sh"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
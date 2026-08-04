# Docstring Generator API — Reference

Interactive docs (Swagger UI): GET `/docs`
Alternative docs (ReDoc): GET `/redoc`
Raw OpenAPI schema: GET `/openapi.json`

---

## GET `/health`

```bash
curl http://localhost:8000/health
```

```json
{"status": "ok", "model_loaded": true, "device": "cuda:0", "base_model": "Qwen/Qwen2.5-Coder-1.5B"}
```

If the model failed to load at startup, status is "degraded" and model_loaded is false.

## POST `/generate`

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"function_code": "def add(a, b):\n    return a + b", "max_length": 150, "temperature": 0.0}'
```

```json
{
  "docstring": "Add two numbers together.\n\nArgs:\n    a (int): The first number.\n    b (int): The second number.\n\nReturns:\n    int: The sum of the two numbers.",
  "model": "docstring-generator-v1",
  "latency_ms": 342.7
}
```

Error: invalid function code (422)

```json
{"error": "validation_error", "detail": "function_code is not valid Python: invalid syntax (line 1)"}
```

Error: model not loaded (503)

```json
{"error": "model_error", "detail": "Model is not loaded. Check server startup logs..."}
```

Error: generation timeout (504)

```json
{"detail": "Generation exceeded 120.0s timeout. Try a shorter function or lower max_length."}
```

## POST `/generate/batch`

```bash
curl -X POST http://localhost:8000/generate/batch \
  -H "Content-Type: application/json" \
  -d '{"functions": ["def add(a, b):\n    return a + b", "not valid python :("], "max_length": 150}'
```

```json
{
  "results": [
    {"index": 0, "docstring": "Add two numbers together...", "error": null},
    {"index": 1, "docstring": null, "error": "function_code is not valid Python: invalid syntax (line 1)"}
  ],
  "model": "docstring-generator-v1",
  "total_latency_ms": 410.2
}
```

## POST `/generate/stream`

```bash
curl -N -X POST http://localhost:8000/generate/stream \
  -H "Content-Type: application/json" \
  -d '{"function_code": "def add(a, b):\n    return a + b"}'
```

Streams plain-text chunks as they're generated (`-N` disables curl's output buffering).

## Configuration (environment variables)

| Variable | Default | Purpose |
| --- | --- | --- |
| BASE_MODEL | Qwen/Qwen2.5-Coder-1.5B | Base model to load |
| ADAPTER_PATH | unset | Path to LoRA adapter; omit to serve base model only |
| GENERATION_TIMEOUT_SECONDS | 120 | Per-request generation timeout |
| MODEL_LABEL | docstring-generator-v1 | Label returned in the model field of responses |

## Running

```bash
pip install -r requirements_api.txt
export ADAPTER_PATH=../training/checkpoints_best/final_model
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

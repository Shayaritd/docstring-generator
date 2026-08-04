"""
FastAPI inference service for the docstring generator.

Run with:
    uvicorn app.main:app --host 0.0.0.0 --port 8000

Docs available at /docs (Swagger UI) and /redoc once running.
"""

import asyncio
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError

from .model_manager import ModelManager, ModelLoadError
from .schemas import (
    GenerateRequest, GenerateResponse,
    BatchGenerateRequest, BatchGenerateResponse, BatchGenerateResult,
    HealthResponse, ErrorResponse,
    AdapterReloadRequest, AdapterReloadResponse,
)
from .logging_config import setup_logging

logger = setup_logging()

BASE_MODEL = os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-Coder-1.5B")
ADAPTER_PATH = os.environ.get("ADAPTER_PATH")  # None = base model only, unset LoRA
GENERATION_TIMEOUT_SECONDS = float(os.environ.get("GENERATION_TIMEOUT_SECONDS", "300"))
MODEL_LABEL = os.environ.get("MODEL_LABEL", "docstring-generator-v1")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Loads the model once at startup (not per-request) and cleans up at shutdown.
    This is the current recommended pattern, replacing the deprecated
    @app.on_event("startup") decorator.
    """
    manager = ModelManager(BASE_MODEL, ADAPTER_PATH)
    try:
        start = time.perf_counter()
        manager.load()
        logger.info(
            "Model loaded successfully",
            extra={"base_model": BASE_MODEL, "adapter_path": ADAPTER_PATH or "none",
                   "load_time_s": round(time.perf_counter() - start, 1)},
        )
    except ModelLoadError as e:
        # Don't crash the whole process - let it start so /health reports the
        # failure clearly, rather than the container just exiting with no explanation.
        logger.warning("Model failed to load at startup", extra={"error": str(e)})
    app.state.model_manager = manager
    yield
    app.state.model_manager = None  # release reference for cleanup


app = FastAPI(
    title="Docstring Generator API",
    description="Generates Google-style Python docstrings from function source code, "
                 "using a QLoRA-fine-tuned model.",
    version="1.0.0",
    lifespan=lifespan,
)


# --- Error handlers: convert internal exceptions into consistent ErrorResponse JSON ---

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Pydantic validation failures -> 422 with a consistent error shape."""
    first_error = exc.errors()[0] if exc.errors() else {}
    detail = first_error.get("msg", "Invalid request")
    return JSONResponse(
        status_code=422,
        content=ErrorResponse(error="validation_error", detail=detail).model_dump(),
    )


@app.exception_handler(RuntimeError)
async def runtime_error_handler(request: Request, exc: RuntimeError):
    """Model-not-loaded or generation-failure errors -> 503, not a raw 500."""
    return JSONResponse(
        status_code=503,
        content=ErrorResponse(error="model_error", detail=str(exc)).model_dump(),
    )


def get_model_manager(request: Request) -> ModelManager:
    manager: ModelManager = request.app.state.model_manager
    if manager is None or not manager.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Check server startup logs - likely a GPU or model download failure.",
        )
    return manager


@app.get("/health", response_model=HealthResponse, tags=["monitoring"])
async def health(request: Request):
    """Health check. Returns 200 whether or not the model loaded successfully -
    check `model_loaded` in the response body, don't rely on HTTP status alone,
    since load balancers may need 200 to keep the pod routable for debugging.
    """
    manager: ModelManager = request.app.state.model_manager
    return HealthResponse(
        status="ok" if (manager and manager.is_loaded) else "degraded",
        model_loaded=bool(manager and manager.is_loaded),
        device=manager.device if manager else None,
        base_model=manager.base_model_name if manager else None,
    )


@app.post(
    "/generate",
    response_model=GenerateResponse,
    tags=["inference"],
    responses={
        422: {"model": ErrorResponse, "description": "Invalid function_code or out-of-range parameters"},
        503: {"model": ErrorResponse, "description": "Model not loaded"},
        504: {"model": ErrorResponse, "description": "Generation timed out"},
    },
)
async def generate(req: GenerateRequest, request: Request):
    """Generate a Google-style docstring for a single Python function.

    Example request:
        {"function_code": "def add(a, b):\n    return a + b", "max_length": 150, "temperature": 0.0}
    """
    manager = get_model_manager(request)
    start = time.perf_counter()

    try:
        docstring = await asyncio.wait_for(
            manager.generate_async(req.function_code, req.max_length, req.temperature),
            timeout=GENERATION_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=504,
            detail=f"Generation exceeded {GENERATION_TIMEOUT_SECONDS}s timeout. Try a shorter function or lower max_length.",
        )

    latency_ms = (time.perf_counter() - start) * 1000
    return GenerateResponse(docstring=docstring, model=MODEL_LABEL, latency_ms=round(latency_ms, 2))


@app.post(
    "/generate/batch",
    response_model=BatchGenerateResponse,
    tags=["inference"],
    responses={422: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
)
async def generate_batch(req: BatchGenerateRequest, request: Request):
    """Generate docstrings for multiple functions in one batched forward pass
    (more GPU-efficient than calling /generate N times).

    Per-item errors (e.g. one malformed function in the batch) are reported in
    that item's result rather than failing the whole batch.
    """
    manager = get_model_manager(request)
    start = time.perf_counter()

    # Validate each function individually so one bad item doesn't 422 the whole batch
    from .schemas import validate_python_function_code
    valid_indices, valid_codes, results = [], [], [None] * len(req.functions)
    for i, code in enumerate(req.functions):
        try:
            validate_python_function_code(code)
            valid_indices.append(i)
            valid_codes.append(code)
        except ValueError as e:
            results[i] = BatchGenerateResult(index=i, error=str(e))

    if valid_codes:
        try:
            generated = await asyncio.wait_for(
                asyncio.to_thread(manager.generate_batch, valid_codes, req.max_length, req.temperature),
                timeout=GENERATION_TIMEOUT_SECONDS * max(1, len(valid_codes) // 4),  # scale timeout with batch size
            )
        except asyncio.TimeoutError:
            raise HTTPException(status_code=504, detail="Batch generation timed out.")

        for idx, doc in zip(valid_indices, generated):
            results[idx] = BatchGenerateResult(index=idx, docstring=doc)

    total_latency_ms = (time.perf_counter() - start) * 1000
    return BatchGenerateResponse(results=results, model=MODEL_LABEL, total_latency_ms=round(total_latency_ms, 2))


@app.post("/generate/stream", tags=["inference"])
async def generate_stream(req: GenerateRequest, request: Request):
    """Stream the generated docstring token-by-token as plain text chunks.

    Note: streaming runs the blocking HF generate() call in a background
    thread (via the model_manager's internal Thread), so it doesn't block
    the event loop, but it does hold a GPU generation slot for its duration
    same as a normal request - the timeout wrapper does NOT apply to
    streaming responses (the client is expected to read at its own pace).
    """
    manager = get_model_manager(request)

    async def token_stream():
        loop = asyncio.get_event_loop()
        # generate_stream() is itself a blocking generator; iterate it in a thread
        # and forward chunks via a queue so the event loop isn't blocked.
        queue: asyncio.Queue = asyncio.Queue()
        SENTINEL = object()

        def producer():
            try:
                for chunk in manager.generate_stream(req.function_code, req.max_length, req.temperature):
                    loop.call_soon_threadsafe(queue.put_nowait, chunk)
            except Exception as e:
                loop.call_soon_threadsafe(queue.put_nowait, e)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, SENTINEL)

        import threading
        threading.Thread(target=producer, daemon=True).start()

        while True:
            item = await queue.get()
            if item is SENTINEL:
                break
            if isinstance(item, Exception):
                yield f"\n[ERROR: {item}]"
                break
            yield item

    return StreamingResponse(token_stream(), media_type="text/plain")


@app.post(
    "/admin/reload-adapter",
    response_model=AdapterReloadResponse,
    tags=["admin"],
    responses={
        400: {"model": ErrorResponse, "description": "Adapter path invalid or failed to load"},
        503: {"model": ErrorResponse, "description": "Base model not loaded - nothing to attach the adapter to"},
    },
)
async def reload_adapter(req: AdapterReloadRequest, request: Request):
    """Hot-swap the LoRA adapter without restarting the container.

    Blocks new /generate requests from starting until the swap completes
    (see ModelManager.reload_adapter's locking), so no request is ever
    served by a half-swapped model. In-flight requests at the moment this
    is called are allowed to finish on the old adapter first.

    Not authenticated - if this API is exposed beyond a trusted internal
    network, put this route behind an auth check or a separate internal-only
    port before deploying, since anyone who can reach it can swap the model
    serving all traffic.
    """
    manager: ModelManager = request.app.state.model_manager
    if manager is None or manager.model is None:
        raise HTTPException(status_code=503, detail="Base model is not loaded - cannot attach an adapter yet.")

    try:
        reload_time = await manager.reload_adapter(req.adapter_path)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))

    logger.info("Adapter reloaded", extra={"adapter_path": req.adapter_path, "reload_time_s": round(reload_time, 2)})
    return AdapterReloadResponse(status="ok", adapter_path=req.adapter_path, reload_time_s=round(reload_time, 2))

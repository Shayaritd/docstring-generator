import ast
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator, ConfigDict


def validate_python_function_code(code: str) -> None:
    if not code or not code.strip():
        raise ValueError("function_code cannot be empty")

    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise ValueError(f"function_code is not valid Python: {e}") from e

    has_function = any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) for n in ast.walk(tree))
    if not has_function:
        raise ValueError("function_code must contain at least one function definition (def or async def)")


class GenerateRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "function_code": "def add(a, b):\n    return a + b",
                    "max_length": 150,
                    "temperature": 0.0,
                    "style": "Google Style",
                    "enable_self_correction": False,
                    "enable_schema_aware": False,
                }
            ]
        }
    )

    function_code: str = Field(..., description="Python source containing the function to document", min_length=1)
    max_length: int = Field(150, ge=16, le=512, description="Maximum tokens to generate")
    temperature: float = Field(0.0, ge=0.0, le=2.0, description="Sampling temperature. 0.0 = greedy/deterministic")
    style: str = Field("Google Style", description="Docstring style preset: Google Style, NumPy Style, Concise Internal")
    enable_self_correction: bool = Field(False, description="Enable self-correction loop for missing parameters")
    enable_schema_aware: bool = Field(False, description="Incorporate parsed function signature metadata into the prompt")

    @field_validator("function_code")
    @classmethod
    def check_function_code(cls, v: str) -> str:
        validate_python_function_code(v)
        return v


class GenerateResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "docstring": "Add two numbers together.\n\nArgs:\n    a (int): The first number.\n    b (int): The second number.\n\nReturns:\n    int: The sum of the two numbers.",
                    "model": "docstring-generator-v1 (Qwen2.5-Coder-1.5B + LoRA)",
                    "latency_ms": 342.7,
                    "quality": {"accuracy": 5.0, "completeness": 5.0, "clarity": 5.0, "conciseness": 5.0},
                    "confidence": 100,
                    "hallucinations": [],
                    "corrected": False,
                }
            ]
        }
    )

    docstring: str
    model: str
    latency_ms: float
    quality: Optional[dict] = None
    confidence: Optional[int] = None
    hallucinations: Optional[List[str]] = None
    corrected: bool = False


class BatchGenerateRequest(BaseModel):
    functions: List[str] = Field(..., min_length=1, max_length=50, description="List of function source strings")
    max_length: int = Field(150, ge=16, le=512)
    temperature: float = Field(0.0, ge=0.0, le=2.0)

    @field_validator("functions")
    @classmethod
    def check_all_functions(cls, v: List[str]) -> List[str]:
        if not v:
            raise ValueError("functions list cannot be empty")
        return v


class BatchGenerateResult(BaseModel):
    index: int
    docstring: Optional[str] = None
    error: Optional[str] = None


class BatchGenerateResponse(BaseModel):
    results: List[BatchGenerateResult]
    model: str
    total_latency_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: Optional[str] = None
    base_model: Optional[str] = None


class ErrorResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {"error": "validation_error", "detail": "function_code is not valid Python: invalid syntax (line 1)"}
            ]
        }
    )
    error: str
    detail: str


class AdapterReloadRequest(BaseModel):
    adapter_path: str = Field(..., description="Path to the new LoRA adapter directory to hot-load")


class AdapterReloadResponse(BaseModel):
    status: str
    adapter_path: str
    reload_time_s: float


class VersionResponse(BaseModel):
    version: str
    model: str
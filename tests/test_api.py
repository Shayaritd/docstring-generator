import pytest
from unittest.mock import MagicMock, AsyncMock

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

def test_validate_python_function_code_accepts_valid_function():
    from api.app.schemas import validate_python_function_code
    validate_python_function_code("def add(a, b):\n    return a + b")

def test_validate_python_function_code_rejects_empty():
    from api.app.schemas import validate_python_function_code
    with pytest.raises(ValueError, match="cannot be empty"):
        validate_python_function_code("")

def test_validate_python_function_code_rejects_syntax_error():
    from api.app.schemas import validate_python_function_code
    with pytest.raises(ValueError, match="not valid Python"):
        validate_python_function_code("def add(a, b:\n    return a + b")

def test_validate_python_function_code_rejects_no_function():
    from api.app.schemas import validate_python_function_code
    with pytest.raises(ValueError, match="function definition"):
        validate_python_function_code("x = 1 + 2")

def test_validate_python_function_code_accepts_async():
    from api.app.schemas import validate_python_function_code
    validate_python_function_code("async def fetch(url):\n    pass")

def test_generate_request_rejects_out_of_range_temperature():
    from api.app.schemas import GenerateRequest
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        GenerateRequest(function_code="def f(): pass", temperature=3.0)

def test_generate_request_rejects_out_of_range_max_length():
    from api.app.schemas import GenerateRequest
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        GenerateRequest(function_code="def f(): pass", max_length=5)

def test_batch_request_rejects_empty_list():
    from api.app.schemas import BatchGenerateRequest
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        BatchGenerateRequest(functions=[])

def test_generation_kwargs_greedy_omits_temperature():
    from api.app.model_manager import ModelManager
    mgr = ModelManager("fake/model")
    mgr.tokenizer = MagicMock(pad_token_id=0)
    kwargs = mgr._generation_kwargs(150, 0.0)
    assert kwargs["do_sample"] is False
    assert "temperature" not in kwargs

def test_generation_kwargs_sampling_includes_temperature():
    from api.app.model_manager import ModelManager
    mgr = ModelManager("fake/model")
    mgr.tokenizer = MagicMock(pad_token_id=0)
    kwargs = mgr._generation_kwargs(150, 0.7)
    assert kwargs["do_sample"] is True
    assert kwargs["temperature"] == 0.7

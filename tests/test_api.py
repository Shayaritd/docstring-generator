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


def test_scoring_good_add():
    from ui.ui_helpers import score_quality, calculate_confidence
    code = "def add(a, b):\n    return a + b"
    docstring = """Add two numbers.

    Args:
        a: First number.
        b: Second number.

    Returns:
        The sum of a and b.
    """
    scores = score_quality(docstring, code)
    avg_score = sum(scores.values()) / len(scores)
    assert 4.0 <= avg_score <= 4.5
    conf = calculate_confidence(docstring, code)
    assert conf > 50


def test_scoring_lower_calculate():
    from ui.ui_helpers import score_quality, calculate_confidence
    code = "def calculate(a, b, c):\n    return a + b + c"
    docstring = """Calculate the sum.

    Args:
        a: First number.
        b: Second number.

    Returns:
        The sum of the numbers.
    """
    scores = score_quality(docstring, code)
    avg_score = sum(scores.values()) / len(scores)
    assert 3.0 <= avg_score <= 4.0
    conf = calculate_confidence(docstring, code)
    assert conf < 90


def test_scoring_best_divide():
    from ui.ui_helpers import score_quality, calculate_confidence
    code = "def divide(a, b):\n    if b == 0:\n        raise ValueError('Cannot divide by zero')\n    return a / b"
    docstring = """Divide two numbers.

    Args:
        a (float): The numerator.
        b (float): The denominator.

    Returns:
        float: The result of division.

    Raises:
        ValueError: If the denominator is zero.
    """
    scores = score_quality(docstring, code)
    avg_score = sum(scores.values()) / len(scores)
    assert 4.5 <= avg_score <= 5.0
    conf = calculate_confidence(docstring, code)
    assert conf >= 90


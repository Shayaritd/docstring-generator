import pytest
import requests

BASE_URL = "http://localhost:8000"

def test_health():
    r = requests.get(f"{BASE_URL}/health")
    assert r.status_code == 200
    assert "model_loaded" in r.json()

def test_generate():
    r = requests.post(
        f"{BASE_URL}/generate",
        json={"function_code": "def add(a, b):\n    return a + b"}
    )
    assert r.status_code == 200
    assert "docstring" in r.json()

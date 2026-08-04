import requests
import ast
import re

def call_generate_api(api_url, function_code, max_length=150, temperature=0.0, timeout=305.0):
    url = f"{api_url.rstrip('/')}/generate"
    payload = {
        "function_code": function_code,
        "max_length": max_length,
        "temperature": temperature
    }
    try:
        response = requests.post(url, json=payload, timeout=timeout)
        if response.status_code == 200:
            data = response.json()
            # Add quality scoring
            data["quality"] = score_quality(data.get("docstring", ""), function_code)
            # Add confidence
            data["confidence"] = calculate_confidence(data.get("docstring", ""), function_code)
            # Add hallucinations
            data["hallucinations"] = check_hallucinations(data.get("docstring", ""), function_code)
            # Check if self-correction was applied
            data["corrected"] = data.get("corrected", False)
            return {"success": True, **data}
        else:
            return {"success": False, "error": f"HTTP {response.status_code}"}
    except Exception as e:
        return {"success": False, "error": str(e)}

def check_api_health(api_url, timeout=5.0):
    try:
        response = requests.get(f"{api_url.rstrip('/')}/health", timeout=timeout)
        if response.status_code == 200:
            data = response.json()
            return {"reachable": True, "model_loaded": data.get("model_loaded", False)}
        return {"reachable": True, "model_loaded": False}
    except:
        return {"reachable": False, "model_loaded": False}

def score_quality(docstring, function_code):
    scores = {"accuracy": 4.0, "completeness": 4.0, "clarity": 4.0, "conciseness": 4.0}
    # Simple scoring logic
    if docstring:
        lines = docstring.split("\n")
        if len(lines) > 3:
            scores["clarity"] = min(5.0, 4.0 + len(lines) * 0.1)
        if "Args:" in docstring:
            scores["completeness"] = min(5.0, scores["completeness"] + 0.5)
        if "Returns:" in docstring:
            scores["completeness"] = min(5.0, scores["completeness"] + 0.5)
        if len(docstring) > 100:
            scores["conciseness"] = max(3.0, 5.0 - len(docstring) * 0.005)
    return scores

def calculate_confidence(docstring, function_code):
    confidence = 85
    if not docstring or len(docstring) < 50:
        confidence -= 20
    if "Args:" not in docstring:
        confidence -= 15
    if "Returns:" not in docstring:
        confidence -= 15
    return max(0, min(100, confidence))

def check_hallucinations(docstring, function_code):
    hallucinations = []
    try:
        tree = ast.parse(function_code)
        func = None
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                func = node
                break
        if func:
            params = [arg.arg for arg in func.args.args if arg.arg not in ("self", "cls")]
            for param in params:
                if param not in docstring and param not in hallucinations:
                    hallucinations.append(f"Parameter '{param}' not documented")
    except:
        pass
    return hallucinations

def format_latency(latency_ms):
    if latency_ms < 1000:
        return f"{latency_ms:.0f} ms"
    elif latency_ms < 60000:
        return f"{latency_ms/1000:.1f} s"
    else:
        return f"{latency_ms/60000:.1f} min"

def format_quality_score(score):
    if score >= 4.5:
        return "🌟 Excellent"
    elif score >= 3.5:
        return "✅ Good"
    elif score >= 2.5:
        return "⚠️ Average"
    else:
        return "❌ Needs Improvement"

def get_confidence_level(confidence):
    if confidence >= 80:
        return "High", "🟢"
    elif confidence >= 50:
        return "Medium", "🟡"
    else:
        return "Low", "🔴"

def insert_docstring_into_function(code, docstring):
    lines = code.split("\n")
    insert_pos = 1
    for i, line in enumerate(lines):
        if line.strip().startswith("def "):
            insert_pos = i + 1
            break
    indent = " " * 4
    doc_lines = docstring.split("\n")
    doc_block = [f'{indent}"""']
    for line in doc_lines:
        doc_block.append(f'{indent}{line}')
    doc_block.append(f'{indent}"""')
    new_lines = lines[:insert_pos] + doc_block + lines[insert_pos:]
    return "\n".join(new_lines)

EXAMPLE_FUNCTIONS = {
    "Simple: add two numbers": "def add(a, b):\n    return a + b",
    "Exception handling": "def divide(a, b):\n    if b == 0:\n        raise ValueError('Cannot divide by zero')\n    return a / b",
    "Generator (yield)": "def count_up_to(n):\n    i = 1\n    while i <= n:\n        yield i\n        i += 1",
    "Type hints + defaults": "def greet(name: str, greeting: str = 'Hello') -> str:\n    return f'{greeting}, {name}!'",
    "Class method": "class User:\n    @classmethod\n    def from_dict(cls, data):\n        return cls(data['name'], data['age'])",
    "Recursion": "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)",
}

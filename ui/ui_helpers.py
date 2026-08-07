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
            data["quality"] = score_quality(data.get("docstring", ""), function_code)
            data["confidence"] = calculate_confidence(data.get("docstring", ""), function_code)
            data["hallucinations"] = check_hallucinations(data.get("docstring", ""), function_code)
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

def validate_code_syntax(code):
    if not code or not code.strip():
        return {"valid": False, "error": "Code cannot be empty"}
    try:
        ast.parse(code)
        return {"valid": True}
    except SyntaxError as e:
        return {"valid": False, "error": f"Syntax Error: {e.msg} at line {e.lineno}, column {e.offset}"}

def score_quality(docstring, function_code):
    """REAL quality scoring based on docstring quality."""
    scores = {"accuracy": 0.0, "completeness": 0.0, "clarity": 0.0, "conciseness": 0.0}
    
    if not docstring or not docstring.strip():
        return scores
    
    # Parse function signature
    params = []
    has_raise = False
    has_return = False
    try:
        tree = ast.parse(function_code)
        func = None
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                func = node
                break
        if func:
            params = [arg.arg for arg in func.args.args if arg.arg not in ("self", "cls")]
            has_raise = any(isinstance(n, ast.Raise) for n in ast.walk(func))
            has_return = any(isinstance(n, ast.Return) for n in ast.walk(func))
    except:
        pass
    
    # Check sections
    has_args_section = any(x in docstring for x in ["Args:", "Parameters:", "Arguments:"])
    has_returns_section = any(x in docstring for x in ["Returns:", "Yields:"])
    has_raises_section = "Raises:" in docstring
    
    # 1. ACCURACY: Parameter coverage and type hints
    if params:
        documented = sum(1 for p in params if re.search(r'\b' + re.escape(p) + r'\b', docstring))
        accuracy = (documented / len(params)) * 3.0
        if documented == len(params):
            accuracy = 4.5
            
            # Penalize missing type info in params (e.g. param (type):)
            missing_types = 0
            for p in params:
                match = re.search(r'^\s*' + re.escape(p) + r'\s*(\([^)]+\)|:[a-zA-Z_0-9]+)\s*:', docstring, re.MULTILINE)
                if not match:
                    missing_types += 1
            accuracy -= missing_types * 0.5
            
        scores["accuracy"] = max(1.0, min(5.0, accuracy))
    else:
        scores["accuracy"] = 4.0
    
    # 2. COMPLETENESS: Required sections and missing params
    completeness = 0.0
    if has_args_section: completeness += 1.5
    if has_returns_section: completeness += 1.5
    if has_raises_section: completeness += 1.0
    
    # Penalize missing sections/parameters
    if params and not has_args_section:
        completeness -= 1.0
    
    if params:
        documented = sum(1 for p in params if re.search(r'\b' + re.escape(p) + r'\b', docstring))
        missing_count = len(params) - documented
        completeness -= missing_count * 1.0
        
    scores["completeness"] = max(1.0, min(5.0, completeness + 0.5))
    
    # 3. CLARITY: Structure, formatting, and description quality
    clarity = 0.0
    lines = [l.strip() for l in docstring.split("\n") if l.strip()]
    if lines:
        summary = lines[0].replace('"""', '').strip()
        if summary and summary[0].isupper():
            clarity += 0.5
        if summary and summary[-1] in (".", "?", "!"):
            clarity += 0.5
            
    if len(lines) >= 5:
        clarity += 0.5
    if len(lines) >= 8:
        clarity += 0.5
    if has_args_section and has_returns_section:
        clarity += 0.5
    if has_raises_section:
        clarity += 0.5
        
    # Penalize unclear parameter/return descriptions (1 or fewer words)
    unclear_penalties = 0.0
    for line in lines:
        match = re.match(r'^\s*([a-zA-Z_0-9]+)\s*(\([^)]+\))?\s*[:-]\s*(.*)$', line)
        if match:
            param_name = match.group(1)
            if param_name in params:
                description = match.group(3).strip()
                words = description.split()
                if len(words) < 2:
                    unclear_penalties += 0.25
                
    scores["clarity"] = max(1.0, min(5.0, clarity + 2.0 - unclear_penalties))
    
    # 4. CONCISENESS: Appropriate length relative to function complexity
    word_count = len(docstring.split())
    if 12 <= word_count <= 150:
        scores["conciseness"] = 5.0
    elif 8 <= word_count < 12:
        scores["conciseness"] = 4.0
    elif 150 < word_count <= 250:
        scores["conciseness"] = 3.5
    else:
        scores["conciseness"] = 2.0
    
    return scores

def calculate_confidence(docstring, function_code):
    """REAL confidence based on docstring quality."""
    # Start at 50%
    confidence = 50
    
    if not docstring or not docstring.strip():
        return 10
    
    # Parse function
    params = []
    has_raise = False
    has_return = False
    try:
        tree = ast.parse(function_code)
        func = None
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                func = node
                break
        if func:
            params = [arg.arg for arg in func.args.args if arg.arg not in ("self", "cls")]
            has_raise = any(isinstance(n, ast.Raise) for n in ast.walk(func))
            has_return = any(isinstance(n, ast.Return) for n in ast.walk(func))
    except:
        pass
    
    # Check sections
    has_args = any(x in docstring for x in ["Args:", "Parameters:", "Arguments:"])
    has_returns = any(x in docstring for x in ["Returns:", "Yields:"])
    has_raises = "Raises:" in docstring
    
    # Args section
    if params:
        documented = sum(1 for p in params if re.search(r'\b' + re.escape(p) + r'\b', docstring))
        if documented == len(params) and has_args:
            confidence += 15
        else:
            missing = len(params) - documented
            confidence -= missing * 15
    else:
        if not has_args:
            confidence += 15
            
    # Returns section
    if has_return:
        if has_returns:
            confidence += 15
        else:
            confidence -= 10
    else:
        if not has_returns:
            confidence += 15
            
    # Raises section
    if has_raise:
        if has_raises:
            confidence += 10
        else:
            confidence -= 10
    else:
        if not has_raises:
            confidence += 10
            
    # Length check
    if len(docstring) < 40:
        confidence -= 15
    elif len(docstring) > 600:
        confidence -= 10
    
    return min(100, max(0, confidence))

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
                if param not in docstring:
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

import requests
import ast
import re

def call_generate_api(api_url, function_code, max_length=150, temperature=0.0, style="Google Style", enable_self_correction=False, enable_schema_aware=False, timeout=305.0):
    url = f"{api_url.rstrip('/')}/generate"
    payload = {
        "function_code": function_code,
        "max_length": max_length,
        "temperature": temperature,
        "style": style,
        "enable_self_correction": enable_self_correction,
        "enable_schema_aware": enable_schema_aware
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
            return {"success": False, "error": f"HTTP {response.status_code}: {response.text}"}
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

def parse_function_signature(function_code):
    try:
        tree = ast.parse(function_code)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                params = [arg.arg for arg in node.args.args if arg.arg not in ("self", "cls")]
                
                # Check for return statement
                has_return = False
                for subnode in ast.walk(node):
                    if isinstance(subnode, ast.Return) and subnode.value is not None:
                        has_return = True
                        break
                
                # Check for yield
                has_yield = any(isinstance(sn, (ast.Yield, ast.YieldFrom)) for sn in ast.walk(node))
                
                # Check for raises
                raises = []
                for subnode in ast.walk(node):
                    if isinstance(subnode, ast.Raise):
                        if subnode.exc:
                            if isinstance(subnode.exc, ast.Name):
                                raises.append(subnode.exc.id)
                            elif isinstance(subnode.exc, ast.Call) and isinstance(subnode.exc.func, ast.Name):
                                raises.append(subnode.exc.func.id)
                
                return {
                    "name": node.name,
                    "params": params,
                    "has_return": has_return or has_yield,
                    "raises": list(set(raises))
                }
    except:
        pass
    return None

def find_docstring_params(docstring):
    params = []
    lines = docstring.split("\n")
    in_args_section = False
    for line in lines:
        line_stripped = line.strip()
        if any(marker in line_stripped for marker in ["Args:", "Parameters:", "Parameters", "Arguments:"]):
            in_args_section = True
            continue
        elif in_args_section and any(marker in line_stripped for marker in ["Returns:", "Raises:", "Yields:", "Returns"]):
            in_args_section = False
        
        if in_args_section:
            match = re.match(r'^([a-zA-Z_][a-zA-Z0-9_]*)\s*(\([^)]+\))?\s*[:-]', line_stripped)
            if match:
                params.append(match.group(1))
    return list(set(params))

def get_missing_parameters(docstring, function_code):
    try:
        tree = ast.parse(function_code)
        func = None
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                func = node
                break
        if func:
            params = [arg.arg for arg in func.args.args if arg.arg not in ("self", "cls")]
            docstring_lower = docstring.lower()
            missing = []
            for param in params:
                if not re.search(r'\b' + re.escape(param.lower()) + r'\b', docstring_lower):
                    missing.append(param)
            return missing
    except:
        pass
    return []

def score_quality(docstring, function_code):
    scores = {
        "accuracy": 5.0,
        "completeness": 5.0,
        "clarity": 5.0,
        "conciseness": 5.0
    }
    
    if not docstring or not docstring.strip():
        return {k: 0.0 for k in scores}
        
    sig = parse_function_signature(function_code)
    docstring_lower = docstring.lower()
    
    # 1. Accuracy
    if sig:
        if sig["params"]:
            documented_count = sum(1 for p in sig["params"] if re.search(r'\b' + re.escape(p.lower()) + r'\b', docstring_lower))
            coverage = documented_count / len(sig["params"])
            scores["accuracy"] = coverage * 5.0
        
        # Deduct for hallucinations (documented but not in signature)
        hallucinations = check_hallucinations(docstring, function_code)
        if hallucinations:
            scores["accuracy"] = max(0.0, scores["accuracy"] - len(hallucinations) * 1.0)
            
    # 2. Completeness
    if sig:
        comp_deductions = 0.0
        
        # Section presence checks
        has_args = any(x in docstring for x in ["Args:", "Parameters:", "Arguments:"])
        has_returns = any(x in docstring for x in ["Returns:", "Yields:"])
        has_raises = any(x in docstring for x in ["Raises:"])
        
        if sig["params"] and not has_args:
            comp_deductions += 2.0
        if sig["has_return"] and not has_returns:
            comp_deductions += 2.0
        if sig["raises"] and not has_raises:
            comp_deductions += 1.0
            
        # Parameter completeness
        missing_params = get_missing_parameters(docstring, function_code)
        comp_deductions += len(missing_params) * 1.0
            
        # Deduct if no type hints are present
        has_type_hints = bool(re.search(r'[a-zA-Z0-9_]+\s*\([^)]+\)\s*:', docstring))
        if not has_type_hints:
            comp_deductions += 1.0
            
        scores["completeness"] = max(0.0, 5.0 - comp_deductions)
        
    # 3. Clarity
    clarity_deductions = 0.0
    lines = [l.strip() for l in docstring.split("\n") if l.strip()]
    if lines:
        summary = lines[0]
        if not (summary and summary[0].isupper()):
            clarity_deductions += 0.5
        if not (summary and summary[-1] in (".", "!", "?")):
            clarity_deductions += 0.5
            
    # Check blank line after summary line
    raw_lines = docstring.split("\n")
    summary_idx = -1
    for idx, l in enumerate(raw_lines):
        if l.strip():
            summary_idx = idx
            break
    if summary_idx != -1 and summary_idx + 1 < len(raw_lines):
        if raw_lines[summary_idx + 1].strip() != "":
            clarity_deductions += 0.5
        
    # Deduct if no type hints are present
    has_type_hints = bool(re.search(r'[a-zA-Z0-9_]+\s*\([^)]+\)\s*:', docstring))
    if not has_type_hints:
        clarity_deductions += 1.5
        
    scores["clarity"] = max(0.0, 5.0 - clarity_deductions)
    
    # 4. Conciseness
    char_len = len(docstring)
    if char_len < 30:
        scores["conciseness"] = 2.0
    elif char_len < 100:
        scores["conciseness"] = 4.5
    elif char_len > 400:
        scores["conciseness"] = max(1.0, 5.0 - (char_len - 400) * 0.005)
        
    # Round all scores to 1 decimal place
    for k in scores:
        scores[k] = round(scores[k], 1)
        
    return scores

def calculate_confidence(docstring, function_code):
    confidence = 100
    
    if not docstring or not docstring.strip():
        return 0
        
    sig = parse_function_signature(function_code)
    if sig:
        # 1. Parameter coverage
        if sig["params"]:
            missing_params = get_missing_parameters(docstring, function_code)
            coverage = (len(sig["params"]) - len(missing_params)) / len(sig["params"])
            confidence -= (1.0 - coverage) * 40
            
        # 2. Section presence
        has_args = any(x in docstring for x in ["Args:", "Parameters:"])
        if sig["params"] and not has_args:
            confidence -= 20
            
        has_returns = any(x in docstring for x in ["Returns:", "Yields:"])
        if sig["has_return"] and not has_returns:
            confidence -= 20
            
        has_raises = any(x in docstring for x in ["Raises:"])
        if sig["raises"] and not has_raises:
            confidence -= 10

    # 3. Docstring length
    word_count = len(docstring.split())
    if word_count < 10:
        confidence -= 30
    elif word_count < 25:
        confidence -= 10
    elif word_count > 150:
        confidence -= min(30, (word_count - 150) * 0.2)
        
    return max(0, min(100, int(confidence)))

def check_hallucinations(docstring, function_code):
    try:
        tree = ast.parse(function_code)
        func = None
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                func = node
                break
        if func:
            actual_params = [arg.arg for arg in func.args.args if arg.arg not in ("self", "cls")]
            doc_params = find_docstring_params(docstring)
            hallucinations = []
            for dp in doc_params:
                if dp not in actual_params:
                    hallucinations.append(f"Parameter '{dp}' documented but not in function signature")
            return hallucinations
    except:
        pass
    return []

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

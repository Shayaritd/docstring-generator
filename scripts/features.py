"""
Extracts structural features from a (code, docstring) pair.
"""
import ast
from dataclasses import dataclass, asdict

@dataclass
class FunctionFeatures:
    is_async: bool = False
    is_method: bool = False
    num_params: int = 0
    has_star_args: bool = False
    has_star_kwargs: bool = False
    has_defaults: bool = False
    has_type_hints: bool = False
    has_decorator: bool = False
    decorator_names: tuple = ()
    has_return_value: bool = False
    has_yield: bool = False
    has_raise: bool = False
    is_nested: bool = False
    num_lines: int = 0
    num_branches: int = 0
    complexity_score: int = 0
    complexity_bucket: str = "simple"
    category: str = "basic"

    def to_dict(self) -> dict:
        return asdict(self)

def _find_primary_function(tree: ast.Module):
    candidates = []
    def collect(body):
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                candidates.append(node)
            elif isinstance(node, ast.ClassDef):
                collect(node.body)
    collect(tree.body)
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    non_init = [c for c in candidates if c.name != "__init__"]
    pool = non_init if non_init else candidates
    decorated = [c for c in pool if c.decorator_list]
    if decorated:
        return decorated[-1]
    return pool[-1]

def extract_features(code: str, docstring: str = "") -> FunctionFeatures:
    features = FunctionFeatures()
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return features
    func = _find_primary_function(tree)
    if func is None:
        return features
    features.is_async = isinstance(func, ast.AsyncFunctionDef)
    features.num_lines = len(code.strip().splitlines())
    args = func.args
    all_args = args.posonlyargs + args.args + args.kwonlyargs
    features.num_params = len(all_args)
    features.has_star_args = args.vararg is not None
    features.has_star_kwargs = args.kwarg is not None
    features.has_defaults = bool(args.defaults or args.kw_defaults)
    features.has_type_hints = any(a.annotation is not None for a in all_args) or (func.returns is not None)
    if all_args and all_args[0].arg in ("self", "cls"):
        features.is_method = True
    features.has_decorator = bool(func.decorator_list)
    features.decorator_names = tuple(
        d.id if isinstance(d, ast.Name) else getattr(d, "attr", "unknown")
        for d in func.decorator_list
    )
    for node in ast.walk(func):
        if isinstance(node, ast.Return) and node.value is not None:
            features.has_return_value = True
        if isinstance(node, (ast.Yield, ast.YieldFrom)):
            features.has_yield = True
        if isinstance(node, ast.Raise):
            features.has_raise = True
        if isinstance(node, (ast.If, ast.For, ast.While, ast.Try)):
            features.num_branches += 1
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node is not func:
            features.is_nested = True
    score = 0
    score += features.num_params
    score += 2 * features.num_branches
    score += 2 if features.has_raise else 0
    score += 2 if features.has_yield else 0
    score += 1 if features.has_decorator else 0
    score += 2 if features.is_nested else 0
    score += 1 if (features.has_star_args or features.has_star_kwargs) else 0
    score += max(0, features.num_lines - 3)
    features.complexity_score = score
    if score <= 4:
        features.complexity_bucket = "simple"
    elif score <= 9:
        features.complexity_bucket = "medium"
    else:
        features.complexity_bucket = "complex"
    if features.is_async:
        features.category = "async"
    elif features.has_yield:
        features.category = "generator"
    elif "property" in features.decorator_names:
        features.category = "property"
    elif "staticmethod" in features.decorator_names:
        features.category = "staticmethod"
    elif "classmethod" in features.decorator_names:
        features.category = "classmethod"
    elif features.has_decorator:
        features.category = "decorator"
    elif features.is_nested:
        features.category = "nested"
    elif features.has_raise:
        features.category = "exception"
    elif features.has_star_args or features.has_star_kwargs:
        features.category = "args_kwargs"
    elif features.num_params >= 3:
        features.category = "multi_param"
    elif features.has_type_hints:
        features.category = "type_hinted"
    else:
        features.category = "basic"
    return features

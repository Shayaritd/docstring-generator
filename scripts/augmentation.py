"""
Augmentation transforms for dataset.
"""
import ast
import random
import re
from typing import Optional, Dict

GENERIC_NAMES = ["value_a", "value_b", "value_c", "value_d", "value_e", "value_f"]

def _get_func_node(tree: ast.Module):
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return node
    return None

def rename_identifiers(example: Dict[str, str]) -> Optional[Dict[str, str]]:
    code, doc = example["input"], example["output"]
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    func = _get_func_node(tree)
    if func is None:
        return None
    positional = [a for a in func.args.args if a.arg not in ("self", "cls")]
    if not positional:
        return None
    rename_map = {}
    for i, arg in enumerate(positional):
        if i >= len(GENERIC_NAMES):
            break
        rename_map[arg.arg] = GENERIC_NAMES[i]
    if not rename_map:
        return None
    class RenameTransformer(ast.NodeTransformer):
        def visit_Name(self, node):
            if node.id in rename_map:
                node.id = rename_map[node.id]
            return node
        def visit_arg(self, node):
            if node.arg in rename_map:
                node.arg = rename_map[node.arg]
            return node
    new_tree = RenameTransformer().visit(tree)
    ast.fix_missing_locations(new_tree)
    new_code = ast.unparse(new_tree)
    new_doc = doc
    for old_name, new_name in rename_map.items():
        new_doc = re.sub(rf"\b{re.escape(old_name)}\b", new_name, new_doc)
    return {"instruction": example["instruction"], "input": new_code, "output": new_doc}

def add_type_hints(example: Dict[str, str]) -> Optional[Dict[str, str]]:
    code, doc = example["input"], example["output"]
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    func = _get_func_node(tree)
    if func is None:
        return None
    already_annotated = any(a.annotation is not None for a in func.args.args)
    if already_annotated:
        return None
    type_map = {}
    for match in re.finditer(r"^\s*(\w+)\s*\(([\w\[\], .]+)(?:, optional)?\)\s*:", doc, re.MULTILINE):
        param_name, param_type = match.group(1), match.group(2).strip()
        type_map[param_name] = param_type
    if not type_map:
        return None
    applied = False
    for arg in func.args.args:
        if arg.arg in type_map:
            arg.annotation = ast.Name(id=type_map[arg.arg], ctx=ast.Load())
            applied = True
    if not applied:
        return None
    ast.fix_missing_locations(tree)
    new_code = ast.unparse(tree)
    return {"instruction": example["instruction"], "input": new_code, "output": doc}

def remove_type_hints(example: Dict[str, str]) -> Optional[Dict[str, str]]:
    code, doc = example["input"], example["output"]
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    func = _get_func_node(tree)
    if func is None:
        return None
    had_hints = any(a.annotation is not None for a in func.args.args) or func.returns is not None
    if not had_hints:
        return None
    for arg in func.args.args + func.args.kwonlyargs:
        arg.annotation = None
    func.returns = None
    ast.fix_missing_locations(tree)
    new_code = ast.unparse(tree)
    return {"instruction": example["instruction"], "input": new_code, "output": doc}

def toggle_async(example: Dict[str, str]) -> Optional[Dict[str, str]]:
    code, doc = example["input"], example["output"]
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    func = _get_func_node(tree)
    if func is None or isinstance(func, ast.AsyncFunctionDef):
        return None
    for node in ast.walk(func):
        if isinstance(node, (ast.Yield, ast.YieldFrom)):
            return None
    new_func = ast.AsyncFunctionDef(
        name=func.name,
        args=func.args,
        body=func.body,
        decorator_list=func.decorator_list,
        returns=func.returns,
        type_comment=getattr(func, "type_comment", None),
    )
    tree.body = [new_func if n is func else n for n in tree.body]
    ast.fix_missing_locations(tree)
    new_code = ast.unparse(tree)
    return {"instruction": example["instruction"], "input": new_code, "output": doc}

def add_decorator(example: Dict[str, str], decorator: str = "functools.lru_cache()") -> Optional[Dict[str, str]]:
    code, doc = example["input"], example["output"]
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    func = _get_func_node(tree)
    if func is None or func.decorator_list:
        return None
    decorator_expr = ast.parse(decorator, mode="eval").body
    func.decorator_list = [decorator_expr]
    ast.fix_missing_locations(tree)
    new_code = ast.unparse(tree)
    return {"instruction": example["instruction"], "input": new_code, "output": doc}

AUGMENTATIONS = {
    "rename_identifiers": rename_identifiers,
    "add_type_hints": add_type_hints,
    "remove_type_hints": remove_type_hints,
    "toggle_async": toggle_async,
    "add_decorator": add_decorator,
}

def augment_example(example: Dict[str, str], seed: Optional[int] = None) -> list:
    if seed is not None:
        random.seed(seed)
    results = []
    for name, fn in AUGMENTATIONS.items():
        try:
            out = fn(example)
        except Exception:
            out = None
        if out is not None and out["input"] != example["input"]:
            results.append(out)
    return results

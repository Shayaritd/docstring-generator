import asyncio
import os
import sys
import time
from typing import List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "training"))


class ModelLoadError(Exception):
    pass


class ModelManager:

    def __init__(self, base_model_name: str, adapter_path: Optional[str] = None):
        self.base_model_name = base_model_name
        self.adapter_path = adapter_path
        self.model = None
        self.tokenizer = None
        self.device = None
        self._load_lock = asyncio.Lock()

    def load(self) -> None:
        try:
            from load_model import load_tokenizer, load_quantized_model

            self.tokenizer = load_tokenizer(self.base_model_name)
            model = load_quantized_model(self.base_model_name, attach_lora=self.adapter_path is None)

            if self.adapter_path:
                from peft import PeftModel
                model = PeftModel.from_pretrained(model, self.adapter_path)

            model.eval()
            self.model = model
            self.device = str(model.device) if hasattr(model, "device") else "unknown"

        except Exception as e:
            raise ModelLoadError(f"Failed to load model '{self.base_model_name}': {e}") from e

    async def reload_adapter(self, new_adapter_path: str) -> float:
        if self.model is None:
            raise RuntimeError("Cannot reload adapter: base model is not loaded")

        import time as _time
        from peft import PeftModel

        async with self._load_lock:
            start = _time.perf_counter()
            try:
                if hasattr(self.model, "unload"):
                    base = self.model.unload()
                else:
                    base = self.model
                new_model = PeftModel.from_pretrained(base, new_adapter_path)
                new_model.eval()
                self.model = new_model
                self.adapter_path = new_adapter_path
            except Exception as e:
                raise RuntimeError(f"Failed to reload adapter from '{new_adapter_path}': {e}") from e

            return _time.perf_counter() - start

    @property
    def is_loaded(self) -> bool:
        return self.model is not None and self.tokenizer is not None

    def _build_prompt(self, function_code: str, style: str = "Google Style", schema_info: Optional[dict] = None) -> str:
        style_instructions = {
            "Google Style": "Write clear, Google-style docstrings, with standard sections like 'Args:', 'Returns:', and 'Raises:'.",
            "NumPy Style": "Write clear, NumPy-style docstrings, with sections like 'Parameters', 'Returns', and 'Raises'.",
            "Concise Internal": "Write a short, concise docstring. Keep the explanation to one or two sentences and list parameters and return value briefly."
        }
        instruction = style_instructions.get(style, style_instructions["Google Style"])

        schema_context = ""
        if schema_info:
            params_str = ", ".join([p["name"] + (f" (default: {p['default']})" if "default" in p else "") for p in schema_info["params"]])
            schema_context = (
                f"\nFunction signature analyzed:\n"
                f"- Name: {schema_info['name']}\n"
                f"- Parameters: {params_str or 'None'}\n"
                f"- Returns: {schema_info['returns'] or 'Not explicitly annotated'}\n"
                f"- Raises: {', '.join(schema_info['raises']) if schema_info['raises'] else 'None'}\n"
                f"- Async: {schema_info['is_async']}\n"
                f"- Generator: {schema_info['is_generator']}\n"
            )

        messages = [
            {"role": "system", "content": f"You are an expert Python developer. {instruction}"},
            {"role": "user", "content": f"Generate a docstring for this Python function.{schema_context}\n\n{function_code}"},
        ]
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def _build_correction_prompt(self, function_code: str, draft_docstring: str, missing_params: list, style: str) -> str:
        missing_str = ", ".join(missing_params)
        style_instruct = "Google Style" if style != "NumPy Style" else "NumPy Style"
        messages = [
            {"role": "system", "content": "You are a precise Python developer who updates existing docstrings."},
            {"role": "user", "content": f"The following draft docstring is missing documentation for these parameters: {missing_str}.\n"
                                        f"Please output a corrected, complete {style_instruct} docstring that adds the missing parameters to the argument list. "
                                        f"Preserve all existing parameter documentation and the original summary exactly.\n\n"
                                        f"Function:\n{function_code}\n\n"
                                        f"Draft Docstring:\n{draft_docstring}"}
        ]
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def _generation_kwargs(self, max_length: int, temperature: float) -> dict:
        kwargs = {
            "max_new_tokens": max_length,
            "use_cache": True,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if temperature > 0.0:
            kwargs["do_sample"] = True
            kwargs["temperature"] = temperature
        else:
            kwargs["do_sample"] = False
        return kwargs

    def _extract_docstring(self, text: str) -> str:
        if '"""' in text:
            parts = text.split('"""')
            if len(parts) > 1:
                return parts[1].strip()
        if "'''" in text:
            parts = text.split("'''")
            if len(parts) > 1:
                return parts[1].strip()
        return text.strip()

    def generate_sync(
        self,
        function_code: str,
        max_length: int = 150,
        temperature: float = 0.0,
        style: str = "Google Style",
        enable_self_correction: bool = False,
        enable_schema_aware: bool = False,
    ) -> dict:
        if not self.is_loaded:
            raise RuntimeError("Model is not loaded")

        import torch

        schema_info = parse_signature(function_code)
        prompt = self._build_prompt(function_code, style, schema_info if enable_schema_aware else None)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        input_len = inputs.input_ids.shape[1]

        try:
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    **self._generation_kwargs(max_length, temperature),
                )
        except torch.cuda.OutOfMemoryError as e:
            raise RuntimeError(f"GPU out of memory during generation: {e}") from e

        generated_ids = output_ids[0][input_len:]
        decoded = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        docstring = self._extract_docstring(decoded)

        corrected = False
        missing_params = []
        if schema_info and schema_info["params"]:
            docstring_lower = docstring.lower()
            for param in schema_info["params"]:
                pname = param["name"]
                if not re.search(r'\b' + re.escape(pname) + r'\b', docstring_lower):
                    missing_params.append(pname)

        if enable_self_correction and missing_params:
            correction_prompt = self._build_correction_prompt(function_code, docstring, missing_params, style)
            corr_inputs = self.tokenizer(correction_prompt, return_tensors="pt").to(self.model.device)
            corr_input_len = corr_inputs.input_ids.shape[1]
            try:
                with torch.no_grad():
                    corr_output_ids = self.model.generate(
                        **corr_inputs,
                        **self._generation_kwargs(max_length + 50, temperature),
                    )
                corr_generated_ids = corr_output_ids[0][corr_input_len:]
                corr_decoded = self.tokenizer.decode(corr_generated_ids, skip_special_tokens=True)
                docstring = self._extract_docstring(corr_decoded)
                corrected = True
                
                # Re-calculate missing params after correction
                missing_params = []
                docstring_lower = docstring.lower()
                for param in schema_info["params"]:
                    pname = param["name"]
                    if not re.search(r'\b' + re.escape(pname) + r'\b', docstring_lower):
                        missing_params.append(pname)
            except Exception:
                # Fallback to first pass on error
                pass

        hallucinations = check_hallucinations(docstring, schema_info)
        confidence = calculate_confidence(docstring, schema_info, missing_params, hallucinations)
        quality = score_quality(docstring, schema_info, missing_params, hallucinations)

        return {
            "docstring": docstring,
            "quality": quality,
            "confidence": confidence,
            "hallucinations": hallucinations,
            "corrected": corrected
        }

    async def generate_async(
        self,
        function_code: str,
        max_length: int = 150,
        temperature: float = 0.0,
        style: str = "Google Style",
        enable_self_correction: bool = False,
        enable_schema_aware: bool = False,
    ) -> dict:
        return await asyncio.to_thread(
            self.generate_sync,
            function_code,
            max_length,
            temperature,
            style,
            enable_self_correction,
            enable_schema_aware,
        )

    def generate_batch(
        self,
        function_codes: List[str],
        max_length: int = 150,
        temperature: float = 0.0,
        style: str = "Google Style",
        enable_schema_aware: bool = False
    ) -> List[dict]:
        if not self.is_loaded:
            raise RuntimeError("Model is not loaded")

        import torch

        results = []
        for code in function_codes:
            # For batch mode, run synchronously sequentially to easily support all features on CPU
            res = self.generate_sync(
                code,
                max_length=max_length,
                temperature=temperature,
                style=style,
                enable_self_correction=False, # self correction in batch can be slow, disable or enable simple
                enable_schema_aware=enable_schema_aware
            )
            results.append(res)
        return results

    def generate_stream(
        self,
        function_code: str,
        max_length: int = 150,
        temperature: float = 0.0,
        style: str = "Google Style",
        enable_schema_aware: bool = False
    ):
        if not self.is_loaded:
            raise RuntimeError("Model is not loaded")

        import torch
        from threading import Thread
        from transformers import TextIteratorStreamer

        schema_info = parse_signature(function_code) if enable_schema_aware else None
        prompt = self._build_prompt(function_code, style, schema_info)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        generation_kwargs = dict(
            **inputs,
            **self._generation_kwargs(max_length, temperature),
            streamer=streamer,
        )
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        for chunk in streamer:
            yield chunk
        thread.join()


# --- CPU-friendly AST Helper Functions for Advanced Features ---

import ast
import re

def parse_signature(function_code: str) -> Optional[dict]:
    try:
        tree = ast.parse(function_code)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                params = []
                for arg in node.args.args:
                    if arg.arg not in ("self", "cls"):
                        annotation = ast.unparse(arg.annotation) if arg.annotation else None
                        params.append({"name": arg.arg, "type": annotation})
                
                defaults = [ast.unparse(d) for d in node.args.defaults]
                for i, val in enumerate(reversed(defaults)):
                    if i < len(params):
                        params[-(i+1)]["default"] = val
                
                returns = ast.unparse(node.returns) if node.returns else None
                
                raises = []
                for subnode in ast.walk(node):
                    if isinstance(subnode, ast.Raise):
                        if subnode.exc:
                            if isinstance(subnode.exc, ast.Name):
                                raises.append(subnode.exc.id)
                            elif isinstance(subnode.exc, ast.Call) and isinstance(subnode.exc.func, ast.Name):
                                raises.append(subnode.exc.func.id)
                
                is_async = isinstance(node, ast.AsyncFunctionDef)
                is_generator = any(isinstance(sn, (ast.Yield, ast.YieldFrom)) for sn in ast.walk(node))
                
                return {
                    "name": node.name,
                    "params": params,
                    "returns": returns,
                    "raises": list(set(raises)),
                    "is_async": is_async,
                    "is_generator": is_generator
                }
    except Exception:
        pass
    return None

def find_docstring_params(docstring: str) -> list:
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

def check_hallucinations(docstring: str, schema_info: dict) -> list:
    if not schema_info:
        return []
    doc_params = find_docstring_params(docstring)
    actual_params = [p["name"] for p in schema_info["params"]]
    hallucinations = []
    for dp in doc_params:
        if dp not in actual_params:
            hallucinations.append(f"Parameter '{dp}' documented but not in function signature")
    return hallucinations

def calculate_confidence(docstring: str, schema_info: dict, missing_params: list, hallucinations: list) -> int:
    confidence = 100
    if not docstring or len(docstring) < 30:
        confidence -= 25
    if missing_params:
        confidence -= 15 * len(missing_params)
    if hallucinations:
        confidence -= 20 * len(hallucinations)
    
    if schema_info:
        docstring_lower = docstring.lower()
        if schema_info["returns"] and not any(r in docstring_lower for r in ["returns", "return"]):
            confidence -= 15
        if schema_info["raises"] and not any(r in docstring_lower for r in ["raises", "raise"]):
            confidence -= 10
            
    return max(0, min(100, confidence))

def score_quality(docstring: str, schema_info: dict, missing_params: list, hallucinations: list) -> dict:
    scores = {"accuracy": 5.0, "completeness": 5.0, "clarity": 5.0, "conciseness": 5.0}
    
    if hallucinations:
        scores["accuracy"] = max(1.0, 5.0 - len(hallucinations) * 1.5)
        
    if not docstring:
        scores["completeness"] = 1.0
    else:
        comp_deductions = 0.0
        if missing_params:
            comp_deductions += len(missing_params) * 1.0
        if schema_info:
            docstring_lower = docstring.lower()
            if schema_info["returns"] and not any(r in docstring_lower for r in ["returns", "return"]):
                comp_deductions += 0.5
            if schema_info["raises"] and not any(r in docstring_lower for r in ["raises", "raise"]):
                comp_deductions += 0.5
        scores["completeness"] = max(1.0, 5.0 - comp_deductions)
        
    if not docstring:
        scores["clarity"] = 1.0
    else:
        clarity_score = 4.0
        lines = [l.strip() for l in docstring.split("\n") if l.strip()]
        if lines:
            if lines[0] and lines[0][0].isupper():
                clarity_score += 0.5
            if lines[0] and lines[0][-1] in (".", "?", "!"):
                clarity_score += 0.5
        if "Args:" in docstring or "Parameters:" in docstring:
            clarity_score = min(5.0, clarity_score + 0.5)
        scores["clarity"] = max(1.0, min(5.0, clarity_score))
        
    if not docstring:
        scores["conciseness"] = 1.0
    else:
        conciseness_score = 5.0
        if len(docstring) > 600:
            conciseness_score -= 1.5
        elif len(docstring) > 400:
            conciseness_score -= 0.8
        scores["conciseness"] = max(1.0, conciseness_score)
        
    return scores
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

    def _build_prompt(self, function_code: str) -> str:
        messages = [
            {"role": "system", "content": "You are an expert Python developer. Write clear, Google-style docstrings for the given function."},
            {"role": "user", "content": f"Generate a Google-style docstring for this Python function.\n\n{function_code}"},
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

    def generate_sync(self, function_code: str, max_length: int = 150, temperature: float = 0.0) -> str:
        if not self.is_loaded:
            raise RuntimeError("Model is not loaded")

        import torch

        prompt = self._build_prompt(function_code)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        try:
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    **self._generation_kwargs(max_length, temperature),
                )
        except torch.cuda.OutOfMemoryError as e:
            raise RuntimeError(f"GPU out of memory during generation: {e}") from e

        full_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return full_text[len(prompt):].strip() if full_text.startswith(prompt) else full_text.strip()

    async def generate_async(self, function_code: str, max_length: int = 150, temperature: float = 0.0) -> str:
        return await asyncio.to_thread(self.generate_sync, function_code, max_length, temperature)

    def generate_batch(self, function_codes: List[str], max_length: int = 150, temperature: float = 0.0) -> List[str]:
        if not self.is_loaded:
            raise RuntimeError("Model is not loaded")

        import torch

        prompts = [self._build_prompt(code) for code in function_codes]
        original_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"
        try:
            inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.model.device)
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    **self._generation_kwargs(max_length, temperature),
                )
        except torch.cuda.OutOfMemoryError as e:
            raise RuntimeError(f"GPU out of memory during batch generation (batch_size={len(function_codes)}): {e}") from e
        finally:
            self.tokenizer.padding_side = original_padding_side

        results = []
        for i, prompt in enumerate(prompts):
            full_text = self.tokenizer.decode(output_ids[i], skip_special_tokens=True)
            results.append(full_text[len(prompt):].strip() if full_text.startswith(prompt) else full_text.strip())
        return results

    def generate_stream(self, function_code: str, max_length: int = 150, temperature: float = 0.0):
        if not self.is_loaded:
            raise RuntimeError("Model is not loaded")

        import torch
        from threading import Thread
        from transformers import TextIteratorStreamer

        prompt = self._build_prompt(function_code)
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
"""
Custom TrainerCallback implementations for the training script:

  - GPUMemoryLoggingCallback: adds GPU memory stats to every log event
  - SampleGenerationCallback: generates docstrings for a few validation
    examples at the end of each epoch, prints them, and logs a W&B table
  - BLEUEvalCallback: generates completions for the full validation set at
    the end of each epoch and computes/logs a BLEU score
"""

from typing import List, Dict, Optional

from transformers import TrainerCallback

from metrics import compute_bleu


class GPUMemoryLoggingCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        try:
            import torch
            if torch.cuda.is_available():
                logs["gpu_mem_allocated_gb"] = round(torch.cuda.memory_allocated() / 1e9, 3)
                logs["gpu_mem_reserved_gb"] = round(torch.cuda.memory_reserved() / 1e9, 3)
                logs["gpu_mem_peak_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 3)
        except ImportError:
            pass


class SampleGenerationCallback(TrainerCallback):
    def __init__(self, model, tokenizer, eval_examples: List[Dict], num_samples: int = 3,
                 max_new_tokens: int = 128, use_wandb: bool = True):
        self.model = model
        self.tokenizer = tokenizer
        self.eval_examples = eval_examples[:num_samples]
        self.max_new_tokens = max_new_tokens
        self.use_wandb = use_wandb

    def _generate(self, example: Dict) -> str:
        import torch
        messages = [
            {"role": "system", "content": "You are an expert Python developer. Write clear, Google-style docstrings for the given function."},
            {"role": "user", "content": f"{example['instruction']}\n\n{example['input']}"},
        ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        full_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return full_text[len(prompt):].strip() if full_text.startswith(prompt) else full_text.strip()

    def on_epoch_end(self, args, state, control, **kwargs):
        print(f"\n--- Sample generations (epoch {state.epoch:.0f}) ---")
        rows = []
        for ex in self.eval_examples:
            try:
                generated = self._generate(ex)
            except Exception as e:
                generated = f"[generation failed: {e}]"
            print(f"  Code: {ex['input'][:60]}...")
            print(f"  Reference:  {ex['output'][:80]}...")
            print(f"  Generated:  {generated[:80]}...\n")
            rows.append([ex["input"], ex["output"], generated])

        if self.use_wandb:
            try:
                import wandb
                if wandb.run is not None:
                    table = wandb.Table(columns=["code", "reference_docstring", "generated_docstring"], data=rows)
                    wandb.log({"sample_generations": table, "epoch": state.epoch})
            except ImportError:
                pass


class BLEUEvalCallback(TrainerCallback):
    def __init__(self, model, tokenizer, eval_examples: List[Dict], max_eval_examples: Optional[int] = 50,
                 max_new_tokens: int = 128, use_wandb: bool = True):
        self.model = model
        self.tokenizer = tokenizer
        self.eval_examples = eval_examples[:max_eval_examples] if max_eval_examples else eval_examples
        self.max_new_tokens = max_new_tokens
        self.use_wandb = use_wandb

    def _generate_batch(self) -> List[str]:
        import torch
        from tqdm import tqdm

        generations = []
        for ex in tqdm(self.eval_examples, desc="BLEU eval generation"):
            messages = [
                {"role": "system", "content": "You are an expert Python developer. Write clear, Google-style docstrings for the given function."},
                {"role": "user", "content": f"{ex['instruction']}\n\n{ex['input']}"},
            ]
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs, max_new_tokens=self.max_new_tokens, do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            full_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            generations.append(full_text[len(prompt):].strip() if full_text.startswith(prompt) else full_text.strip())
        return generations

    def on_epoch_end(self, args, state, control, **kwargs):
        references = [ex["output"] for ex in self.eval_examples]
        hypotheses = self._generate_batch()
        result = compute_bleu(references, hypotheses)
        print(f"\nValidation BLEU (epoch {state.epoch:.0f}): {result['bleu']:.2f} (method: {result['method']})")

        if self.use_wandb:
            try:
                import wandb
                if wandb.run is not None:
                    wandb.log({"eval/bleu": result["bleu"], "epoch": state.epoch})
            except ImportError:
                pass

"""
Comprehensive evaluation: base vs fine-tuned on held-out test set.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "training"))
from metrics import compute_bleu
from rouge_metric import compute_rouge_l
from section_metrics import compare_sections, check_signature_detection


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-Coder-1.5B")
    parser.add_argument("--adapter_path", type=str, required=True)
    parser.add_argument("--test_path", type=str, default="../data/processed/test.jsonl")
    parser.add_argument("--n_examples", type=int, default=25)
    parser.add_argument("--max_new_tokens", type=int, default=150)
    parser.add_argument("--run_judge", action="store_true")
    parser.add_argument("--judge_model", type=str, default="claude-sonnet-4-6")
    parser.add_argument("--output_dir", type=str, default="eval_results")
    return parser.parse_args()


def load_test_examples(path: str, n: int) -> list:
    import json
    examples = []
    with open(path, "r") as f:
        for line in f:
            examples.append(json.loads(line))
    return examples[:n]


def generate_docstring(model, tokenizer, instruction: str, code: str, max_new_tokens: int) -> str:
    import torch
    messages = [
        {"role": "system", "content": "You are an expert Python developer. Write clear, Google-style docstrings."},
        {"role": "user", "content": f"{instruction}\n\n{code}"},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                   pad_token_id=tokenizer.pad_token_id)
    full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return full_text[len(prompt):].strip() if full_text.startswith(prompt) else full_text.strip()


def run_model_on_test_set(model, tokenizer, examples: list, max_new_tokens: int, label: str) -> list:
    from tqdm import tqdm
    outputs = []
    for ex in tqdm(examples, desc=f"Generating ({label})"):
        try:
            generated = generate_docstring(model, tokenizer, ex["instruction"], ex["input"], max_new_tokens)
        except Exception as e:
            print(f"WARNING: generation failed: {e}")
            generated = f"[GENERATION FAILED: {e}]"
        outputs.append(generated)
    return outputs


def compute_all_metrics(examples: list, generations: list) -> dict:
    references = [ex["output"] for ex in examples]
    bleu_result = compute_bleu(references, generations)
    rouge_result = compute_rouge_l(references, generations)
    per_example = []
    for ex, gen in zip(examples, generations):
        section_result = compare_sections(ex["output"], gen)
        signature_result = check_signature_detection(ex["input"], gen)
        per_example.append({
            "code": ex["input"],
            "reference": ex["output"],
            "generated": gen,
            "sections": section_result,
            "signature": signature_result,
        })
    exact_match_rates = [p["sections"]["exact_match_rate"] for p in per_example if p["sections"]["exact_match_rate"] is not None]
    signature_matches = [p["signature"]["signature_match"] for p in per_example]
    return {
        "bleu": bleu_result,
        "rouge_l": rouge_result,
        "avg_section_exact_match_rate": sum(exact_match_rates) / len(exact_match_rates) if exact_match_rates else None,
        "signature_match_rate": sum(signature_matches) / len(signature_matches) if signature_matches else None,
        "per_example": per_example,
    }


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Loading {args.n_examples} examples from {args.test_path}...")
    examples = load_test_examples(args.test_path, args.n_examples)

    from load_model import load_tokenizer, load_quantized_model
    tokenizer = load_tokenizer(args.base_model)

    print(f"Loading base model: {args.base_model}")
    base_model = load_quantized_model(args.base_model, attach_lora=False)
    base_generations = run_model_on_test_set(base_model, tokenizer, examples, args.max_new_tokens, "base")

    print(f"Loading fine-tuned model: {args.adapter_path}")
    from peft import PeftModel
    finetuned_model = PeftModel.from_pretrained(base_model, args.adapter_path)
    finetuned_generations = run_model_on_test_set(finetuned_model, tokenizer, examples, args.max_new_tokens, "fine-tuned")

    print("Computing metrics...")
    base_metrics = compute_all_metrics(examples, base_generations)
    finetuned_metrics = compute_all_metrics(examples, finetuned_generations)

    results = {"config": vars(args), "base": base_metrics, "finetuned": finetuned_metrics}

    if args.run_judge:
        print("Running LLM-as-judge...")
        from llm_judge import judge_batch
        base_judge_input = [{"input": ex["input"], "generated": gen} for ex, gen in zip(examples, base_generations)]
        finetuned_judge_input = [{"input": ex["input"], "generated": gen} for ex, gen in zip(examples, finetuned_generations)]
        results["base"]["judge_scores"] = judge_batch(base_judge_input, model=args.judge_model)
        results["finetuned"]["judge_scores"] = judge_batch(finetuned_judge_input, model=args.judge_model)

    results_path = os.path.join(args.output_dir, "raw_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved -> {results_path}")


if __name__ == "__main__":
    main()

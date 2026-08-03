# Docstring Generator — File Manifest

Every file delivered to date, mapped to the consolidated project structure from the earlier audit. "Delivered" means it's in your outputs folder right now — confirmed against the actual filesystem, not memory of what was discussed.

---

## Phase 1: Dataset Preparation (Days 1-2)

| Target path | Source file | Purpose |
|---|---|---|
| `data/raw/dataset.jsonl` | `dataset.jsonl` | 30 hand-written seed examples |
| `src/core/schema.py` | `schema.py` | Pydantic dataset-row validation |
| `scripts/validate_dataset.py` | `validate_dataset.py` | CLI dataset validator |
| `src/core/features.py` | `pipeline/features.py` | AST feature extraction (category/complexity) |
| `src/core/data_loader.py` | `pipeline/data_loader.py` | JSONL ↔ HF Dataset conversion |
| `src/core/augmentation.py` | `pipeline/augmentation.py` | 5 augmentation transforms |
| `src/core/quality_checks.py` | `pipeline/quality_checks.py` | Style/section/duplicate validation |
| `src/core/split_dataset.py` | `pipeline/split_dataset.py` | Stratified 80/10/10 split |
| `src/core/stats.py` | `pipeline/stats.py` | Dataset statistics |
| `src/core/visualize.py` | `pipeline/visualize.py` | Chart generation |
| `scripts/run_pipeline.py` | `pipeline/run_pipeline.py` | End-to-end pipeline orchestration |
| `data/processed/train.jsonl` | `pipeline/train.jsonl` | 108 training examples |
| `data/processed/val.jsonl` | `pipeline/val.jsonl` | 13 validation examples |
| `data/processed/test.jsonl` | `pipeline/test.jsonl` | 14 test examples |
| `data/processed/dataset_augmented.jsonl` | `pipeline/dataset_augmented.jsonl` | 135 examples pre-split |
| `data/stats/dataset_stats.json` | `pipeline/dataset_stats.json` | Computed statistics |
| `data/stats/quality_report.json` | `pipeline/quality_report.json` | Quality check results |
| `data/stats/plots/*.png` (×5) | `pipeline/plots/*.png` | Category, length, token, param distribution charts |

## Phase 2: LoRA Fine-Tuning Setup (Days 3-4)

| Target path | Source file | Purpose |
|---|---|---|
| `docker/Dockerfile.training` | `training/Dockerfile` | Training environment container |
| `requirements-training.txt` | `training/requirements.txt` | Pinned training dependencies |
| `scripts/setup.sh` | `training/setup.sh` | Install + GPU verification |
| `src/models/training_config.py` | `training/training_config.py` | LoRA + quantization + TrainingArguments config |
| `src/models/load_model.py` | `training/load_model.py` | Model/tokenizer loading, GPU checks |
| `src/utils/metrics.py` | `training/metrics.py` | BLEU computation ⚠️ *built Day 4, only just copied to outputs today* |
| `training/data_utils.py` | `training/data_utils.py` | Chat-template formatting ⚠️ *same gap, fixed today* |
| `training/callbacks.py` | `training/callbacks.py` | GPU/sample-gen/BLEU TrainerCallbacks ⚠️ *same gap, fixed today* |
| `training/config.yaml` | `training/config.yaml` | All training hyperparameters ⚠️ *same gap, fixed today* |
| `training/train.py` | `training/train.py` | Main training script ⚠️ *same gap, fixed today* |

## Phase 3: Hyperparameter Tuning (Day 5) + Evaluation (Day 6)

| Target path | Source file | Purpose |
|---|---|---|
| `sweep/search_space.py` | `sweep/search_space.py` | Optuna search space (6 hyperparameters) |
| `sweep/objective.py` | `sweep/objective.py` | Per-trial training + pruning + W&B |
| `sweep/run_sweep.py` | `sweep/run_sweep.py` | Study creation/resumption |
| `sweep/analyze_results.py` | `sweep/analyze_results.py` | Comparison table, importance plots |
| `sweep/train_best.py` | `sweep/train_best.py` | Final training with best config |
| `evaluation/rouge_metric.py` | `evaluation/rouge_metric.py` | ROUGE-L computation |
| `evaluation/section_metrics.py` | `evaluation/section_metrics.py` | Section exact-match, signature detection |
| `evaluation/llm_judge.py` | `evaluation/llm_judge.py` | Claude-as-judge scoring |
| `evaluation/run_evaluation.py` | `evaluation/run_evaluation.py` | Base vs. fine-tuned comparison |
| `evaluation/failure_analysis.py` | `evaluation/failure_analysis.py` | Failure categorization |
| `evaluation/report_generator.py` | `evaluation/report_generator.py` | Markdown report builder |

## Phase 4: Inference API (Day 7) + UI (Day 8)

| Target path | Source file | Purpose |
|---|---|---|
| `src/api/main.py` | `api/app/main.py` | FastAPI app: health/generate/batch/stream |
| `src/api/model_manager.py` | `api/app/model_manager.py` | Model serving logic ⚠️ *hot-reload adapter method added but endpoint not wired — see Option 2* |
| `src/api/schemas.py` | `api/app/schemas.py` | Request/response validation |
| `tests/test_api.py` | `api/tests/test_api.py` | Schema + endpoint tests |
| `requirements-api.txt` | `api/requirements_api.txt` | API dependencies |
| `docs/API_REFERENCE.md` | `api/API_REFERENCE.md` | Endpoint documentation |
| `ui/app_ui.py` | `ui/app_ui.py` | Streamlit UI |
| `ui/ui_helpers.py` | `ui/ui_helpers.py` | API client, docstring-merge logic (tested) |
| `requirements-ui.txt` | `ui/requirements_ui.txt` | UI dependencies |

## Phase 5: Containerization (Day 9) — PARTIALLY COMPLETE

| Target path | Status |
|---|---|
| `docker/Dockerfile` | ✅ Delivered this response — multi-stage, CUDA, health check |
| `docker/entrypoint.sh` | ✅ Delivered this response — pre-flight GPU/cache checks |
| `api/app/logging_config.py` | ✅ Delivered this response — structured JSON logging |
| `api/app/model_manager.py` | ✅ Updated this response — hot-reload `reload_adapter()` method added, tested with mocks |
| `api/app/schemas.py` | ✅ Updated this response — `AdapterReloadRequest`/`Response` added |
| `api/app/main.py` | ✅ Updated this response — `POST /admin/reload-adapter` endpoint wired to the backend method, tested with mocks |
| `docker-compose.yml` | ❌ Not built |
| `scripts/build.sh` / `run.sh` / `test.sh` | ❌ Not built |

## Not started

- Day 10: CI/CD (GitHub Actions, model versioning, regression alerting)
- Phase 6: Polish & Portfolio (README, LICENSE, docs/, model card, .gitignore, .env.example)

---

## Summary

**43 files delivered and verified** (syntax-checked at minimum; most also logic-tested with mocks/synthetic data — see each day's response for specifics).
**Phase 5 is now feature-complete except `docker-compose.yml` and the build/run/test shell scripts** — still pending.
**Everything from Day 10 and Phase 6 onward is unbuilt.**

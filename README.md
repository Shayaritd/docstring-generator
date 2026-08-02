# 📝 Docstring Generator

A QLoRA-fine-tuned LLM (Qwen2.5-Coder-1.5B) that generates Google-style docstrings for Python functions, served via FastAPI with a Streamlit UI.

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.140-green.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-24.0-blue.svg)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [API Reference](#api-reference)
- [Testing](#testing)
- [Known Gaps](#known-gaps)
- [Contributing](#contributing)
- [License](#license)

---

## 🎯 Overview

**The Problem:** Writing high-quality Google-style docstrings is time-consuming and often inconsistent across teams. Developers know what the code does, but documenting it properly takes effort.

**The Solution:** A fine-tuned LLM that automatically generates professional, consistent docstrings from Python function code, saving developer time and improving documentation quality.

**Status:** All 5 phases complete. The project is ready for real GPU training and local deployment. Containerization is functional with `docker-compose.yml` and deployment scripts.

---

## ✨ Features

### 🔄 LoRA Fine-Tuning
- QLoRA (4-bit quantization) for memory-efficient training
- Configurable LoRA hyperparameters (rank, alpha, dropout)
- Support for Qwen2.5-Coder-1.5B and Phi-3-mini-4k-instruct

### ⚡ FastAPI Inference
- Single and batch generation
- Streaming token-by-token responses
- Hot-reload adapter support (swap LoRA adapters without restart)
- Health checks and monitoring

### 🖥️ Streamlit UI
- Interactive code editor with syntax highlighting
- Pre-loaded examples
- Response time tracking
- Copy to clipboard

### 🐳 Containerization
- Multi-stage Docker build with CUDA support
- `docker-compose.yml` for orchestration
- GPU reservation pre-configured
- Model cache volume persistence

### 📊 Evaluation & Analysis
- BLEU and ROUGE-L metrics
- Section exact-match (Args/Returns/Raises/Yields)
- LLM-as-judge scoring (Claude/GPT)
- Failure analysis and report generation

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| **Language** | Python 3.11+ |
| **Deep Learning** | PyTorch, Transformers, PEFT (LoRA) |
| **Quantization** | bitsandbytes (QLoRA) |
| **Training** | TRL (SFTTrainer) |
| **API Framework** | FastAPI, Uvicorn |
| **UI Framework** | Streamlit |
| **Containerization** | Docker, Docker Compose |
| **Hyperparameter Tuning** | Optuna |
| **Evaluation** | sacrebleu, rouge_score |
| **Testing** | pytest |

---

## 📁 Project Structure
docstring-generator/
├── api/ # FastAPI inference service
│ ├── app/
│ │ ├── main.py # API entry point
│ │ ├── model_manager.py # Model loading/inference
│ │ ├── schemas.py # Pydantic models
│ │ └── logging_config.py # JSON logging
│ └── requirements_api.txt
├── training/ # QLoRA training
│ ├── train.py # Main training script
│ ├── config.yaml # Hyperparameters
│ ├── metrics.py # BLEU computation
│ ├── data_utils.py # Data loading
│ ├── callbacks.py # Training callbacks
│ └── requirements.txt
├── ui/ # Streamlit UI
│ ├── app_ui.py
│ ├── ui_helpers.py
│ └── requirements_ui.txt
├── sweep/ # Optuna hyperparameter sweep
│ ├── search_space.py
│ ├── objective.py
│ ├── run_sweep.py
│ ├── analyze_results.py
│ └── train_best.py
├── evaluation/ # Model evaluation
│ ├── rouge_metric.py
│ ├── section_metrics.py
│ ├── llm_judge.py
│ ├── run_evaluation.py
│ ├── failure_analysis.py
│ └── report_generator.py
├── scripts/ # Data pipeline
│ ├── validate_dataset.py
│ ├── features.py
│ ├── data_loader.py
│ ├── augmentation.py
│ ├── quality_checks.py
│ ├── split_dataset.py
│ ├── stats.py
│ ├── visualize.py
│ └── run_pipeline.py
├── data/ # Dataset
│ ├── raw/dataset.jsonl
│ ├── processed/train.jsonl
│ ├── processed/val.jsonl
│ └── processed/test.jsonl
├── docker/
│ ├── Dockerfile # Multi-stage CUDA build
│ └── entrypoint.sh
├── tests/
│ └── test_api.py
├── docs/
│ └── API_REFERENCE.md
├── src/core/
│ └── schema.py
├── .env.example
├── docker-compose.yml
├── requirements.txt
├── MANIFEST.md # Complete file inventory
└── README.md

text

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- CUDA-capable GPU (optional, CPU works for inference)
- Docker (for containerized deployment)

### Local Setup

```bash
# 1. Clone the repository
git clone https://github.com/Shayaritd/docstring-generator.git
cd docstring-generator

# 2. Create virtual environment
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
pip install -r api/requirements_api.txt
pip install -r ui/requirements_ui.txt

# 4. Configure environment
cp .env.example .env
# Edit .env with your values

# 5. Run the API
cd api
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
Docker Setup
bash
# 1. Build the image
./scripts/build.sh

# 2. Run the container
./scripts/run.sh

# 3. Smoke-test the deployment
./scripts/test.sh
📖 Usage
1. Prepare the Dataset
bash
python scripts/run_pipeline.py data/raw/dataset.jsonl
2. Train the Model
bash
cd training
python train.py --config config.yaml
3. Hyperparameter Sweep (Optional)
bash
cd sweep
python run_sweep.py --n_trials 20
python analyze_results.py
python train_best.py
4. Evaluate the Model
bash
cd evaluation
python run_evaluation.py --adapter_path ../training/checkpoints_best/final_model --n_examples 25
python report_generator.py --results eval_results/raw_results.json
5. Serve the API
bash
export ADAPTER_PATH=training/checkpoints_best/final_model
cd api
uvicorn app.main:app --host 0.0.0.0 --port 8000
Docs available at: http://localhost:8000/docs

6. Run the UI
bash
cd ui
streamlit run app_ui.py
7. Docker Deployment
bash
cp .env.example .env   # fill in real values first
./scripts/build.sh
./scripts/run.sh                  # API only
./scripts/run.sh --with-logging   # API + Postgres request logging
./scripts/test.sh                 # smoke-test the running deployment
Or directly with compose:

bash
docker compose up
🔌 API Reference
Endpoint	Method	Description
/health	GET	Health check with model status
/generate	POST	Generate docstring for one function
/generate/batch	POST	Generate docstrings for multiple functions
/generate/stream	POST	Stream docstring token-by-token
/admin/reload-adapter	POST	Hot-swap LoRA adapter
📄 Full API Reference: docs/API_REFERENCE.md

🧪 Testing
bash
pytest tests/test_api.py -v

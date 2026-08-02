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

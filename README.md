# Docstring Generator

QLoRA-fine-tuned LLM (Qwen2.5-Coder-1.5B) that generates Google-style Python docstrings, served via FastAPI with a Streamlit UI.

## Features

- LoRA fine-tuning with QLoRA
- FastAPI inference service
- Streamlit UI
- Batch processing
- Streaming responses
- Docker containerization
- Hot-reload adapter support

## Tech Stack

- Python 3.11
- PyTorch
- Transformers
- PEFT (LoRA)
- FastAPI
- Streamlit
- Docker

## Quick Start

\\\ash
pip install -r requirements.txt
uvicorn api.app.main:app --reload --host 0.0.0.0 --port 8000
\\\

## License

MIT

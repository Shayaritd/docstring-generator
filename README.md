Docstring Generator

A QLoRA-fine-tuned LLM (Qwen2.5-Coder-1.5B or Phi-3-mini-4k-instruct) that generates Google-style docstrings for Python functions, served via FastAPI with a Streamlit UI.

Status: Phases 1-4 complete and tested with synthetic/mocked data (real GPU training and live API testing still needed on your hardware — this project was built in a sandbox with no GPU or network access; see MANIFEST.md for exactly what was and wasn't verified). Phase 5 (containerization) is functional but docker-compose.yml and deployment scripts are still pending. Day 10 (CI/CD) and Phase 6 (polish) are not started.

Project Structure

See MANIFEST.md for the complete file-by-file breakdown of what's been built, tested, and verified so far.

├── data/                # Phase 1: dataset + pipeline outputs
├── src/core/             # AST feature extraction, augmentation, quality checks
├── training/              # Phase 2: QLoRA training script + config
├── sweep/                # Phase 3: Optuna hyperparameter sweep
├── evaluation/            # Model evaluation: BLEU/ROUGE/LLM-judge/failure analysis
├── api/                  # Phase 4: FastAPI inference service
├── ui/                   # Phase 4: Streamlit UI
├── docker/                # Phase 5: Dockerfile, entrypoint
├── tests/                 # Test suites
├── .env.example
└── MANIFEST.md            # Full inventory of delivered files
Setup
bash
python3.11 -m venv venv
source venv/bin/activate

# Training environment
pip install -r training/requirements.txt

# API + UI
pip install -r api/requirements_api.txt
pip install -r ui/requirements_ui.txt

cp .env.example .env
# edit .env with your values
Usage
1. Prepare the dataset
bash
python data/run_pipeline.py data/raw/dataset.jsonl
2. Train
bash
cd training
python train.py --config config.yaml
3. (Optional) Hyperparameter sweep
bash
cd sweep
python run_sweep.py --n_trials 20
python analyze_results.py
python train_best.py
4. Evaluate
bash
cd evaluation
python run_evaluation.py --adapter_path ../training/checkpoints_best/final_model --n_examples 25
python report_generator.py --results eval_results/raw_results.json
5. Serve the API
bash
export ADAPTER_PATH=training/checkpoints_best/final_model
cd api
uvicorn app.main:app --host 0.0.0.0 --port 8000

Docs at http://localhost:8000/docs. Full endpoint reference in api/API_REFERENCE.md.

6. Run the UI
bash
cd ui
streamlit run app_ui.py
7. Docker
bash
cp .env.example .env   # fill in real values first
./scripts/build.sh
./scripts/run.sh                  # API only
./scripts/run.sh --with-logging   # API + Postgres request logging
./scripts/test.sh                 # smoke-test the running deployment

Or directly with compose: docker compose up (add --profile logging for Postgres). GPU access, health checks, and the model-weights volume are pre-configured in docker-compose.yml.

Testing
bash
pytest api/tests/test_api.py -v

Note: most test suites in this project were verified with mocked dependencies in a sandbox without GPU/network access. Run them for real on your machine before deploying — see each module's docstring for what was and wasn't actually exercised.

Known gaps

Full list in MANIFEST.md. Highlights: no CI/CD yet, no auth on the API (including the /admin/reload-adapter endpoint — put it behind auth or an internal-only network before exposing this publicly), no LICENSE, Postgres logging schema exists but nothing writes to it yet (logs currently go to stdout as JSON — see api/app/logging_config.py).

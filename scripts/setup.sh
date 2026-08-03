#!/usr/bin/env bash
# Installs dependencies and verifies the training environment is correctly set up.
# Usage: bash setup.sh

set -e

echo "=== Step 1: Checking for NVIDIA GPU (nvidia-smi) ==="
if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: nvidia-smi not found. No NVIDIA driver detected on this machine."
    echo "QLoRA training requires an NVIDIA GPU. If you're in a container, make sure"
    echo "it was launched with --gpus all (Docker) or the equivalent GPU passthrough flag."
    exit 1
fi
nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version --format=csv
echo ""

echo "=== Step 2: Installing Python dependencies ==="
pip install --upgrade pip
pip install -r requirements.txt
echo ""

echo "=== Step 3: Verifying installation ==="
python3 - <<'EOF'
import sys

def check(name, fn):
    try:
        result = fn()
        print(f"  [OK] {name}: {result}")
        return True
    except Exception as e:
        print(f"  [FAIL] {name}: {e}")
        return False

all_ok = True

def check_torch():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("torch installed but CUDA not available")
    return f"torch {torch.__version__}, CUDA {torch.version.cuda}, GPU: {torch.cuda.get_device_name(0)}"

def check_transformers():
    import transformers
    return f"transformers {transformers.__version__}"

def check_peft():
    import peft
    return f"peft {peft.__version__}"

def check_bnb():
    import bitsandbytes as bnb
    return f"bitsandbytes {bnb.__version__}"

def check_trl():
    import trl
    return f"trl {trl.__version__}"

def check_vram():
    import torch
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    if vram_gb < 7.5:
        raise RuntimeError(f"Only {vram_gb:.1f}GB VRAM detected")
    return f"{vram_gb:.1f} GB available"

for name, fn in [
    ("PyTorch + CUDA", check_torch),
    ("transformers", check_transformers),
    ("peft", check_peft),
    ("bitsandbytes", check_bnb),
    ("trl", check_trl),
    ("VRAM capacity", check_vram),
]:
    all_ok = check(name, fn) and all_ok

print()
if all_ok:
    print("All checks passed. Environment is ready for training.")
    sys.exit(0)
else:
    print("One or more checks failed. Fix the issues above before starting training.")
    sys.exit(1)
EOF

echo ""
echo "=== Step 4: Testing model load (optional, requires internet + GPU) ==="
echo "Run this manually when ready: python3 load_model.py"

#!/bin/bash
set -eo pipefail

echo "=== LingBot-World-Fast Setup ==="

# --- System deps ---
sudo apt-get update && sudo apt-get install -y ffmpeg libgl1 git

# --- Find CUDA home ---
if [ -z "${CUDA_HOME:-}" ]; then
  for p in /usr/local/cuda /usr/local/cuda-12.4 /usr/local/cuda-12.2 /usr/local/cuda-12 /usr/lib/cuda /usr; do
    if [ -f "$p/bin/nvcc" ]; then
      export CUDA_HOME="$p"
      break
    fi
  done
  # Last resort: find nvcc anywhere
  if [ -z "${CUDA_HOME:-}" ]; then
    NVCC_PATH=$(find / -name nvcc -type f 2>/dev/null | head -1)
    if [ -n "$NVCC_PATH" ]; then
      export CUDA_HOME=$(dirname $(dirname "$NVCC_PATH"))
    fi
  fi
fi
echo "CUDA_HOME=${CUDA_HOME:-NOT FOUND}"
echo "nvcc: $(which nvcc 2>/dev/null || echo 'not in PATH')"

# --- Check torch + CUDA version ---
TORCH_CUDA=$(python3 -c "import torch; print(torch.version.cuda)" 2>/dev/null || echo "unknown")
echo "PyTorch CUDA: $TORCH_CUDA"
PYTHON_VER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "Python: $PYTHON_VER"

# --- Python deps (skip torch if already installed) ---
pip install \
  "opencv-python>=4.9.0.80" \
  "diffusers>=0.31.0" \
  "transformers>=4.49.0,<=4.51.3" \
  "tokenizers>=0.20.3" \
  "accelerate>=1.1.1" \
  tqdm \
  "imageio[ffmpeg]" \
  easydict \
  ftfy \
  imageio-ffmpeg \
  "numpy>=1.23.5,<2" \
  scipy \
  fastapi \
  uvicorn \
  python-multipart \
  huggingface_hub \
  einops

# --- Flash attention: try prebuilt wheel first, fall back to source ---
echo "Installing flash-attn..."

# Determine the right prebuilt wheel
TORCH_VER=$(python3 -c "import torch; v=torch.__version__.split('+')[0].split('.')[:2]; print('.'.join(v))")
echo "Torch version for wheel: $TORCH_VER"

# Try the prebuilt wheel matching this environment
WHEEL_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch${TORCH_VER}cxx11abiFALSE-cp${PYTHON_VER/./}-cp${PYTHON_VER/./}-linux_x86_64.whl"
echo "Trying prebuilt wheel: $WHEEL_URL"

if pip install "$WHEEL_URL" 2>/dev/null; then
  echo "Installed flash-attn from prebuilt wheel."
else
  echo "Prebuilt wheel not found, building from source..."
  if [ -z "${CUDA_HOME:-}" ]; then
    echo "ERROR: CUDA_HOME not found. Install cuda-toolkit or set CUDA_HOME manually."
    echo "  Try: sudo apt install nvidia-cuda-toolkit"
    echo "  Or:  export CUDA_HOME=/path/to/cuda && pip install flash-attn --no-build-isolation"
    exit 1
  fi
  CUDA_HOME="$CUDA_HOME" pip install flash-attn --no-build-isolation
fi

# --- Clone repo ---
if [ ! -d "/workspace/lingbot-world" ]; then
  git clone https://github.com/robbyant/lingbot-world.git /workspace/lingbot-world
fi

# --- Download models ---
echo ""
echo "Downloading lingbot-world-fast weights (~74GB)..."
huggingface-cli download robbyant/lingbot-world-fast \
  --local-dir /workspace/models/lingbot-world-fast

echo ""
echo "Downloading lingbot-world-base-cam (T5 + VAE, ~160GB)..."
huggingface-cli download robbyant/lingbot-world-base-cam \
  --local-dir /workspace/models/lingbot-world-base-cam

echo ""
echo "=== Setup complete ==="
echo "Run:  python serve.py"

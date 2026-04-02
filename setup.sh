#!/bin/bash
set -euo pipefail

echo "=== LingBot-World-Fast Setup ==="

# --- System deps ---
sudo apt-get update && sudo apt-get install -y ffmpeg libgl1 git

# --- Python deps ---
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0
pip install \
  opencv-python>=4.9.0.80 \
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
  "huggingface_hub[cli]"

pip install flash-attn --no-build-isolation

# --- Clone repo ---
if [ ! -d "/workspace/lingbot-world" ]; then
  git clone https://github.com/robbyant/lingbot-world.git /workspace/lingbot-world
fi

# --- Download models ---
echo "Downloading lingbot-world-fast weights (~74GB)..."
huggingface-cli download robbyant/lingbot-world-fast \
  --local-dir /workspace/models/lingbot-world-fast

echo "Downloading lingbot-world-base-cam (T5 + VAE, ~160GB)..."
huggingface-cli download robbyant/lingbot-world-base-cam \
  --local-dir /workspace/models/lingbot-world-base-cam

echo ""
echo "=== Setup complete ==="
echo "Run:  python serve.py"

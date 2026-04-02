#!/bin/bash
echo "=== Environment Check ==="

echo ""
echo "--- Python & Torch ---"
python3 --version 2>/dev/null || echo "python3: NOT FOUND"
python3 -c "import torch; print(f'torch {torch.__version__}  CUDA: {torch.version.cuda}  GPUs: {torch.cuda.device_count()}')" 2>/dev/null || echo "torch: NOT INSTALLED"

echo ""
echo "--- GPU ---"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "nvidia-smi: NOT FOUND"

echo ""
echo "--- Required Python packages ---"
for pkg in flash_attn diffusers transformers accelerate einops scipy imageio fastapi uvicorn easydict ftfy huggingface_hub numpy PIL cv2; do
  ver=$(python3 -c "import $pkg; print(getattr($pkg, '__version__', 'ok'))" 2>/dev/null)
  if [ $? -eq 0 ]; then
    printf "  %-20s %s\n" "$pkg" "$ver"
  else
    printf "  %-20s MISSING\n" "$pkg"
  fi
done

echo ""
echo "--- System tools ---"
for cmd in ffmpeg git nvcc; do
  loc=$(which $cmd 2>/dev/null)
  if [ -n "$loc" ]; then
    echo "  $cmd: $loc"
  else
    echo "  $cmd: NOT FOUND"
  fi
done

echo ""
echo "--- CUDA_HOME ---"
echo "  ${CUDA_HOME:-NOT SET}"

echo ""
echo "--- Model files ---"
for f in \
  /workspace/models/lingbot-world-fast/config.json \
  /workspace/models/lingbot-world-fast/model-00001-of-00016.safetensors \
  /workspace/models/lingbot-world-fast/model-00016-of-00016.safetensors \
  /workspace/models/lingbot-world-base-cam/models_t5_umt5-xxl-enc-bf16.pth \
  /workspace/models/lingbot-world-base-cam/Wan2.1_VAE.pth \
  /workspace/models/lingbot-world-base-cam/google/umt5-xxl/spiece.model \
  /workspace/lingbot-world/generate.py \
  ; do
  if [ -f "$f" ]; then
    size=$(du -sh "$f" 2>/dev/null | cut -f1)
    echo "  OK  $f ($size)"
  else
    echo "  MISSING  $f"
  fi
done

echo ""
echo "--- Repo & scripts ---"
for f in /workspace/lingbot-world /workspace/lingbot-world/wan; do
  [ -d "$f" ] && echo "  OK  $f/" || echo "  MISSING  $f/"
done
for f in streaming.py serve.py; do
  [ -f "/workspace/$f" ] && echo "  OK  /workspace/$f" || \
  [ -f "$HOME/lingbottesting/$f" ] && echo "  OK  $HOME/lingbottesting/$f" || \
  echo "  MISSING  $f (not in /workspace/ or ~/lingbottesting/)"
done

echo ""
echo "=== Done ==="

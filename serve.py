"""
LingBot-World-Fast — standalone server for bare-metal GPU nodes.
Run: python serve.py
"""

import io
import sys
import tempfile
import logging

import numpy as np
import uvicorn
from PIL import Image as PILImage
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, Response

sys.path.insert(0, "/workspace/lingbot-world")

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
FAST_DIR = "/workspace/models/lingbot-world-fast"
BASE_DIR = "/workspace/models/lingbot-world-base-cam"

# ---------------------------------------------------------------------------
# Load model at startup
# ---------------------------------------------------------------------------
log.info("Initializing pipeline...")

# streaming.py is in the same directory
from streaming import StreamingLingBot
import torch

device = torch.device("cuda:0")
pipeline = StreamingLingBot(
    fast_model_dir=FAST_DIR,
    base_model_dir=BASE_DIR,
    device=device,
)
session_active = False

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="LingBot-World-Fast")

all_frames: list[np.ndarray] = []


def _make_mp4(frames: list[np.ndarray]) -> bytes:
    import imageio
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    writer = imageio.get_writer(tmp.name, fps=16, codec="libx264", quality=8)
    for f in frames:
        writer.append_data(f)
    writer.close()
    with open(tmp.name, "rb") as fh:
        return fh.read()


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML_PAGE


@app.post("/api/start")
async def start_session(
    image: UploadFile = File(...),
    prompt: str = Form(...),
    resolution: str = Form("480*832"),
):
    global all_frames, session_active
    all_frames = []

    image_bytes = await image.read()
    img = PILImage.open(io.BytesIO(image_bytes)).convert("RGB")
    img_np = np.array(img).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)

    h, w = (480, 832) if resolution == "480*832" else (720, 1280)

    frames = pipeline.start_session(
        image_tensor=img_tensor,
        prompt=prompt,
        height=h,
        width=w,
    )
    session_active = True
    all_frames.extend(frames)
    return Response(content=_make_mp4(all_frames), media_type="video/mp4")


@app.post("/api/step/{action}")
async def step(action: str):
    global all_frames
    if not session_active or not all_frames:
        return Response(content=b"No active session", status_code=400)

    frames = pipeline.step(action=action)
    all_frames.extend(frames[1:])  # skip overlap frame
    return Response(content=_make_mp4(all_frames), media_type="video/mp4")


# ---------------------------------------------------------------------------
# HTML UI (same as Modal version)
# ---------------------------------------------------------------------------
HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LingBot-World-Fast</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: system-ui, -apple-system, sans-serif; background: #111; color: #eee; }
  .container { max-width: 1200px; margin: 0 auto; padding: 24px; }
  h1 { font-size: 1.5rem; margin-bottom: 4px; }
  .subtitle { color: #999; font-size: 0.85rem; margin-bottom: 24px; }
  .layout { display: grid; grid-template-columns: 360px 1fr; gap: 24px; }
  @media (max-width: 800px) { .layout { grid-template-columns: 1fr; } }
  .panel { display: flex; flex-direction: column; gap: 16px; }
  label { font-size: 0.8rem; color: #aaa; text-transform: uppercase; letter-spacing: 0.05em; }
  input[type="file"], select, textarea {
    width: 100%; padding: 10px; border: 1px solid #333; border-radius: 8px;
    background: #1a1a1a; color: #eee; font-size: 0.9rem;
  }
  textarea { resize: vertical; min-height: 80px; font-family: inherit; }
  .preview { width: 100%; max-height: 200px; object-fit: contain; border-radius: 8px; display: none; }
  .btn-primary {
    padding: 12px; border: none; border-radius: 8px; font-size: 1rem; font-weight: 600;
    background: #e67e22; color: #fff; cursor: pointer; transition: background 0.2s;
  }
  .btn-primary:hover { background: #d35400; }
  .btn-primary:disabled { background: #555; cursor: not-allowed; }
  .controls { display: grid; grid-template-columns: repeat(4, 1fr); gap: 6px; }
  .ctrl-btn {
    padding: 10px 4px; border: 1px solid #333; border-radius: 6px; background: #222;
    color: #ddd; font-size: 0.75rem; cursor: pointer; text-align: center; transition: background 0.15s;
  }
  .ctrl-btn:hover { background: #444; }
  .ctrl-btn:disabled { opacity: 0.4; cursor: not-allowed; }
  .ctrl-btn.wide { grid-column: span 2; }
  .status { font-size: 0.8rem; color: #888; min-height: 20px; }
  .status.loading { color: #e67e22; }
  .status.error { color: #e74c3c; }
  .video-wrap {
    background: #1a1a1a; border-radius: 12px; overflow: hidden;
    display: flex; align-items: center; justify-content: center; min-height: 400px;
  }
  .video-wrap video { width: 100%; display: block; }
  .placeholder { color: #555; font-size: 0.9rem; }
  kbd {
    background: #333; border: 1px solid #555; border-radius: 3px;
    padding: 1px 5px; font-size: 0.7rem; color: #ccc;
  }
</style>
</head>
<body>
<div class="container">
  <h1>LingBot-World-Fast</h1>
  <p class="subtitle">Interactive camera control &middot; 8&times;A100</p>
  <div class="layout">
    <div class="panel">
      <div>
        <label>Input Image</label>
        <input type="file" id="imageInput" accept="image/*">
        <img id="preview" class="preview" alt="preview">
      </div>
      <div>
        <label>Prompt</label>
        <textarea id="prompt" placeholder="Describe the scene you want to explore..."></textarea>
      </div>
      <div>
        <label>Resolution</label>
        <select id="resolution">
          <option value="480*832">480 &times; 832</option>
          <option value="720*1280">720 &times; 1280</option>
        </select>
      </div>
      <button class="btn-primary" id="startBtn" onclick="startSession()">Start Session</button>
      <div class="status" id="status"></div>
      <div>
        <label>Camera Controls</label>
        <div class="controls" id="controls">
          <button class="ctrl-btn" onclick="step('look_up')">Look Up</button>
          <button class="ctrl-btn" onclick="step('turn_left')">Turn Left</button>
          <button class="ctrl-btn" onclick="step('turn_right')">Turn Right</button>
          <button class="ctrl-btn" onclick="step('look_down')">Look Down</button>
          <button class="ctrl-btn" onclick="step('forward')"><kbd>W</kbd> Forward</button>
          <button class="ctrl-btn" onclick="step('left')"><kbd>A</kbd> Left</button>
          <button class="ctrl-btn" onclick="step('right')"><kbd>D</kbd> Right</button>
          <button class="ctrl-btn" onclick="step('back')"><kbd>S</kbd> Back</button>
          <button class="ctrl-btn" onclick="step('up')">Up</button>
          <button class="ctrl-btn" onclick="step('down')">Down</button>
          <button class="ctrl-btn wide" onclick="step('idle')">Idle (no move)</button>
        </div>
      </div>
    </div>
    <div class="video-wrap" id="videoWrap">
      <span class="placeholder">Upload an image and start a session to begin</span>
    </div>
  </div>
</div>
<script>
const statusEl = document.getElementById('status');
const videoWrap = document.getElementById('videoWrap');
const startBtn = document.getElementById('startBtn');
const controlBtns = document.querySelectorAll('.ctrl-btn');
let busy = false;

document.getElementById('imageInput').addEventListener('change', (e) => {
  const file = e.target.files[0];
  if (file) {
    const prev = document.getElementById('preview');
    prev.src = URL.createObjectURL(file);
    prev.style.display = 'block';
  }
});

document.addEventListener('keydown', (e) => {
  if (e.target.tagName === 'TEXTAREA' || e.target.tagName === 'INPUT') return;
  const map = {
    w: 'forward', a: 'left', s: 'back', d: 'right',
    ArrowUp: 'look_up', ArrowDown: 'look_down',
    ArrowLeft: 'turn_left', ArrowRight: 'turn_right',
    q: 'up', e: 'down', ' ': 'idle',
  };
  const action = map[e.key];
  if (action) { e.preventDefault(); step(action); }
});

function setStatus(msg, cls) {
  statusEl.textContent = msg;
  statusEl.className = 'status ' + (cls || '');
}

function setDisabled(disabled) {
  busy = disabled;
  startBtn.disabled = disabled;
  controlBtns.forEach(b => b.disabled = disabled);
}

async function startSession() {
  const fileInput = document.getElementById('imageInput');
  const prompt = document.getElementById('prompt').value.trim();
  const resolution = document.getElementById('resolution').value;
  if (!fileInput.files.length) { setStatus('Please upload an image.', 'error'); return; }
  if (!prompt) { setStatus('Please enter a prompt.', 'error'); return; }
  setDisabled(true);
  setStatus('Starting session...', 'loading');
  const form = new FormData();
  form.append('image', fileInput.files[0]);
  form.append('prompt', prompt);
  form.append('resolution', resolution);
  try {
    const res = await fetch('/api/start', { method: 'POST', body: form });
    if (!res.ok) throw new Error(await res.text());
    const blob = await res.blob();
    showVideo(blob);
    setStatus('Session active. Use controls or WASD keys to navigate.');
  } catch (err) {
    setStatus('Error: ' + err.message, 'error');
  } finally {
    setDisabled(false);
  }
}

async function step(action) {
  if (busy) return;
  setDisabled(true);
  setStatus('Generating ' + action + '...', 'loading');
  try {
    const res = await fetch('/api/step/' + action, { method: 'POST' });
    if (!res.ok) throw new Error(await res.text());
    const blob = await res.blob();
    showVideo(blob);
    setStatus('Done. Use controls or WASD keys for next step.');
  } catch (err) {
    setStatus('Error: ' + err.message, 'error');
  } finally {
    setDisabled(false);
  }
}

function showVideo(blob) {
  const url = URL.createObjectURL(blob);
  videoWrap.innerHTML = '<video src="' + url + '" controls autoplay loop></video>';
}
</script>
</body>
</html>
"""

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    log.info("Starting server on port 8000...")
    uvicorn.run(app, host="0.0.0.0", port=8000)

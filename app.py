import modal

# ---------------------------------------------------------------------------
# Modal setup
# ---------------------------------------------------------------------------
REPO_URL = "https://github.com/robbyant/lingbot-world.git"
FAST_MODEL_ID = "robbyant/lingbot-world-fast"
BASE_MODEL_ID = "robbyant/lingbot-world-base-cam"
FAST_DIR = "/model-fast"
BASE_DIR = "/model-base"
REPO_DIR = "/repo"
GPU_COUNT = 2
GPU_TYPE = "A100-80GB"

FLASH_ATTN_WHEEL = (
    "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/"
    "flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"
)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "ffmpeg", "libgl1", "libglib2.0-0")
    .pip_install(
        "torch==2.6.0",
        "torchvision==0.21.0",
        "torchaudio==2.6.0",
        "opencv-python>=4.9.0.80",
        "diffusers>=0.31.0",
        "transformers>=4.49.0,<=4.51.3",
        "tokenizers>=0.20.3",
        "accelerate>=1.1.1",
        "tqdm",
        "imageio[ffmpeg]",
        "easydict",
        "ftfy",
        "imageio-ffmpeg",
        "numpy>=1.23.5,<2",
        "scipy",
        "fastapi",
        "python-multipart",
        "huggingface_hub[cli]",
        FLASH_ATTN_WHEEL,
    )
    .run_commands(f"git clone {REPO_URL} {REPO_DIR}")
    .run_commands(
        f"huggingface-cli download {FAST_MODEL_ID} --local-dir {FAST_DIR}"
    )
    .run_commands(
        f"huggingface-cli download {BASE_MODEL_ID} --local-dir {BASE_DIR}"
    )
    .add_local_file("streaming.py", remote_path="/root/streaming.py")
)

app = modal.App("lingbot-world-fast", image=image)

# ---------------------------------------------------------------------------
# GPU inference class – stays warm between requests
# ---------------------------------------------------------------------------
@app.cls(
    gpu=f"{GPU_TYPE}:{GPU_COUNT}",
    timeout=3600,
    scaledown_window=300,
)
class Inference:
    @modal.enter()
    def load_model(self):
        import sys
        sys.path.insert(0, REPO_DIR)
        sys.path.insert(0, "/root")

        from streaming import StreamingLingBot
        import torch

        self.device = torch.device("cuda:0")
        self.pipeline = StreamingLingBot(
            fast_model_dir=FAST_DIR,
            base_model_dir=BASE_DIR,
            device=self.device,
        )
        self.session_active = False

    @modal.method()
    def start_session(
        self, image_bytes: bytes, prompt: str, height: int, width: int
    ) -> list[bytes]:
        import torch
        from PIL import Image as PILImage
        import io
        import numpy as np

        img = PILImage.open(io.BytesIO(image_bytes)).convert("RGB")
        img_np = np.array(img).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)

        frames = self.pipeline.start_session(
            image_tensor=img_tensor,
            prompt=prompt,
            height=height,
            width=width,
        )
        self.session_active = True
        return _encode_frames(frames)

    @modal.method()
    def step(self, action: str) -> list[bytes]:
        if not self.session_active:
            raise RuntimeError("No active session.")
        frames = self.pipeline.step(action=action)
        return _encode_frames(frames)


def _encode_frames(frames) -> list[bytes]:
    import io
    from PIL import Image as PILImage
    encoded = []
    for f in frames:
        img = PILImage.fromarray(f)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=90)
        encoded.append(buf.getvalue())
    return encoded


# ---------------------------------------------------------------------------
# FastAPI web UI
# ---------------------------------------------------------------------------
@app.function(timeout=3600)
@modal.asgi_app()
def web():
    import io
    import tempfile
    import numpy as np
    from PIL import Image as PILImage
    from fastapi import FastAPI, UploadFile, File, Form
    from fastapi.responses import HTMLResponse, Response
    import imageio

    web_app = FastAPI()
    inference = Inference()

    # Shared state
    all_frames: list[np.ndarray] = []

    def _decode(encoded: list[bytes]) -> list[np.ndarray]:
        return [np.array(PILImage.open(io.BytesIO(b))) for b in encoded]

    def _make_mp4(frames: list[np.ndarray]) -> bytes:
        tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        writer = imageio.get_writer(tmp.name, fps=16, codec="libx264", quality=8)
        for f in frames:
            writer.append_data(f)
        writer.close()
        with open(tmp.name, "rb") as fh:
            return fh.read()

    @web_app.get("/", response_class=HTMLResponse)
    async def index():
        return HTML_PAGE

    @web_app.post("/api/start")
    async def start_session(
        image: UploadFile = File(...),
        prompt: str = Form(...),
        resolution: str = Form("480*832"),
    ):
        nonlocal all_frames
        all_frames = []

        image_bytes = await image.read()
        h, w = (480, 832) if resolution == "480*832" else (720, 1280)

        encoded = inference.start_session.remote(
            image_bytes=image_bytes,
            prompt=prompt,
            height=h,
            width=w,
        )
        new_frames = _decode(encoded)
        all_frames.extend(new_frames)
        return Response(
            content=_make_mp4(all_frames),
            media_type="video/mp4",
        )

    @web_app.post("/api/step/{action}")
    async def step(action: str):
        nonlocal all_frames
        if not all_frames:
            return Response(content=b"No active session", status_code=400)

        encoded = inference.step.remote(action=action)
        new_frames = _decode(encoded)
        all_frames.extend(new_frames[1:])  # skip overlap frame
        return Response(
            content=_make_mp4(all_frames),
            media_type="video/mp4",
        )

    return web_app


# ---------------------------------------------------------------------------
# HTML / CSS / JS – single page app
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

  /* Left panel */
  .panel { display: flex; flex-direction: column; gap: 16px; }
  label { font-size: 0.8rem; color: #aaa; text-transform: uppercase; letter-spacing: 0.05em; }
  input[type="file"], select, textarea {
    width: 100%; padding: 10px; border: 1px solid #333; border-radius: 8px;
    background: #1a1a1a; color: #eee; font-size: 0.9rem;
  }
  textarea { resize: vertical; min-height: 80px; font-family: inherit; }
  .preview { width: 100%; max-height: 200px; object-fit: contain; border-radius: 8px; display: none; }

  /* Buttons */
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

  /* Status */
  .status { font-size: 0.8rem; color: #888; min-height: 20px; }
  .status.loading { color: #e67e22; }
  .status.error { color: #e74c3c; }

  /* Video */
  .video-wrap {
    background: #1a1a1a; border-radius: 12px; overflow: hidden;
    display: flex; align-items: center; justify-content: center; min-height: 400px;
  }
  .video-wrap video { width: 100%; display: block; }
  .placeholder { color: #555; font-size: 0.9rem; }

  /* Keyboard hints */
  kbd {
    background: #333; border: 1px solid #555; border-radius: 3px;
    padding: 1px 5px; font-size: 0.7rem; color: #ccc;
  }
</style>
</head>
<body>
<div class="container">
  <h1>LingBot-World-Fast</h1>
  <p class="subtitle">Interactive camera control &middot; 8&times;A100-80GB via Modal</p>

  <div class="layout">
    <!-- Left panel -->
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

    <!-- Right panel: video -->
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

// Preview uploaded image
document.getElementById('imageInput').addEventListener('change', (e) => {
  const file = e.target.files[0];
  if (file) {
    const prev = document.getElementById('preview');
    prev.src = URL.createObjectURL(file);
    prev.style.display = 'block';
  }
});

// Keyboard shortcuts
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
  setStatus('Starting session... (cold start may take ~2 min)', 'loading');

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

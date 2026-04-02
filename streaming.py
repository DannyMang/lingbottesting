"""
Streaming interactive inference for LingBot-World-Fast.

Sliding-window approach: each chunk uses the last frame of the previous
chunk as its init image, with user-supplied camera movements applied per step.
"""

import logging
import math
import os
import sys
import random

import numpy as np
import torch
import torch.cuda.amp as amp
import torchvision.transforms.functional as TF
from einops import rearrange
from tqdm import tqdm

# Add lingbot-world repo to path — works on both Modal (/repo) and bare metal (/workspace)
import pathlib
for _p in ["/repo", "/workspace/lingbot-world"]:
    if pathlib.Path(_p).exists():
        sys.path.insert(0, _p)
        break

from wan.modules.model import WanModel
from wan.modules.t5 import T5EncoderModel
from wan.modules.vae2_1 import Wan2_1_VAE
from wan.utils.cam_utils import (
    compute_relative_poses,
    get_Ks_transformed,
    get_plucker_embeddings,
    interpolate_camera_poses,
)
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

import time

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
log = logging.getLogger(__name__)


def gpu_mem_summary() -> str:
    """Return a compact string of GPU memory usage across all devices."""
    if not torch.cuda.is_available():
        return "no CUDA"
    parts = []
    for i in range(torch.cuda.device_count()):
        alloc = torch.cuda.memory_allocated(i) / 1e9
        total = torch.cuda.get_device_properties(i).total_memory / 1e9
        parts.append(f"GPU{i}: {alloc:.1f}/{total:.0f}GB")
    return " | ".join(parts)


class CudaTimer:
    """CUDA-event based timer. Records GPU-side timestamps.

    Usage:
        # Immediate (syncs in __exit__, serializes pipeline — fine for profiling):
        with CudaTimer("label"):
            ...

        # Deferred (collect timers, flush at end to avoid serialization):
        timers = []
        with CudaTimer("step1", defer=True, collect=timers): ...
        with CudaTimer("step2", defer=True, collect=timers): ...
        CudaTimer.flush(timers)
    """
    def __init__(self, label: str, defer: bool = False, collect: list | None = None):
        self.label = label
        self.defer = defer
        self.collect = collect
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)

    def __enter__(self):
        self.start_event.record()
        return self

    def __exit__(self, *exc):
        self.end_event.record()
        if self.defer and self.collect is not None:
            self.collect.append(self)
        else:
            torch.cuda.synchronize()
            ms = self.start_event.elapsed_time(self.end_event)
            log.info(f"[PROFILE] {self.label}: {ms:.1f}ms | {gpu_mem_summary()}")

    @staticmethod
    def flush(timers: list):
        """Sync once and print all deferred timers."""
        torch.cuda.synchronize()
        for t in timers:
            ms = t.start_event.elapsed_time(t.end_event)
            log.info(f"[PROFILE] {t.label}: {ms:.1f}ms")
        log.info(f"[PROFILE] memory after flush: {gpu_mem_summary()}")
        timers.clear()


# Alias for non-GPU code (CPU-only loads etc.) where CUDA events don't apply
class WallTimer:
    """Simple wall-clock timer for CPU-bound operations."""
    def __init__(self, label: str):
        self.label = label
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self
    def __exit__(self, *exc):
        log.info(f"[PROFILE] {self.label}: {time.perf_counter() - self.t0:.2f}s")


# ---------------------------------------------------------------------------
# Camera helpers
# ---------------------------------------------------------------------------

def make_default_intrinsics(n_frames: int, h: int = 480, w: int = 832) -> torch.Tensor:
    """Create reasonable default camera intrinsics [fx, fy, cx, cy]."""
    fx = fy = w * 0.8  # ~60-degree horizontal FOV
    cx, cy = w / 2.0, h / 2.0
    K = torch.tensor([[fx, fy, cx, cy]], dtype=torch.float32)
    return K.repeat(n_frames, 1)


def action_to_pose_delta(action: str, move_scale: float = 0.3, rot_deg: float = 8.0) -> np.ndarray:
    """
    Convert a user action string to a 4x4 SE3 delta matrix.

    Supported actions:
        forward / back / left / right  - translation
        up / down                       - translation
        turn_left / turn_right          - yaw rotation
        look_up / look_down             - pitch rotation
        idle                            - identity (no movement)
    """
    from scipy.spatial.transform import Rotation

    T = np.eye(4)
    rad = np.deg2rad(rot_deg)

    if action == "forward":
        T[2, 3] = -move_scale  # -Z is forward in OpenCV convention
    elif action == "back":
        T[2, 3] = move_scale
    elif action == "left":
        T[0, 3] = -move_scale
    elif action == "right":
        T[0, 3] = move_scale
    elif action == "up":
        T[1, 3] = -move_scale
    elif action == "down":
        T[1, 3] = move_scale
    elif action == "turn_left":
        T[:3, :3] = Rotation.from_euler("y", rad).as_matrix()
    elif action == "turn_right":
        T[:3, :3] = Rotation.from_euler("y", -rad).as_matrix()
    elif action == "look_up":
        T[:3, :3] = Rotation.from_euler("x", rad).as_matrix()
    elif action == "look_down":
        T[:3, :3] = Rotation.from_euler("x", -rad).as_matrix()
    else:  # idle
        pass

    return T


def build_chunk_poses(current_pose: np.ndarray, action: str, n_frames: int = 17) -> np.ndarray:
    """
    Build a smooth camera trajectory for one chunk.

    Returns array of shape [n_frames, 4, 4] going from current_pose
    to current_pose @ delta over n_frames steps.
    """
    delta = action_to_pose_delta(action)
    end_pose = current_pose @ delta

    poses = np.zeros((n_frames, 4, 4))
    for i in range(n_frames):
        alpha = i / max(n_frames - 1, 1)
        # linear interp on translation, slerp on rotation
        poses[i] = current_pose.copy()
        poses[i, :3, 3] = (1 - alpha) * current_pose[:3, 3] + alpha * end_pose[:3, 3]
    # Use scipy slerp for proper rotation interpolation
    poses_torch = interpolate_camera_poses(
        src_indices=np.array([0.0, n_frames - 1.0]),
        src_rot_mat=np.stack([current_pose[:3, :3], end_pose[:3, :3]]),
        src_trans_vec=np.stack([current_pose[:3, 3], end_pose[:3, 3]]),
        tgt_indices=np.linspace(0, n_frames - 1, n_frames),
    )
    return poses_torch.numpy(), end_pose


# ---------------------------------------------------------------------------
# Streaming pipeline
# ---------------------------------------------------------------------------

class StreamingLingBot:
    """
    Interactive sliding-window inference for LingBot-World-Fast.

    Architecture:
        - Single distilled DiT (loaded from lingbot-world-fast weights)
        - T5 text encoder + VAE from lingbot-world-base-cam
        - Camera control via Plucker embeddings

    Each ``step()`` call:
        1. Takes the last decoded frame as the new init image.
        2. Builds camera poses from the user action.
        3. Runs diffusion (few steps) to generate the next chunk.
        4. Decodes and returns new RGB frames.
    """

    CHUNK_FRAMES = 17       # video frames per chunk (4 latent frames + 1 init)
    SAMPLE_STEPS = 30       # denoising steps (Fast model can use fewer)
    SAMPLE_SHIFT = 3.0      # noise schedule shift for 480p
    GUIDE_SCALE = 5.0
    NUM_TRAIN_TIMESTEPS = 1000

    def __init__(
        self,
        fast_model_dir: str,
        base_model_dir: str,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.device = device
        self.dtype = dtype

        with WallTimer("Load T5 encoder (CPU)"):
            self.text_encoder = T5EncoderModel(
                text_len=512,
                dtype=torch.bfloat16,
                device=torch.device("cpu"),
                checkpoint_path=os.path.join(base_model_dir, "models_t5_umt5-xxl-enc-bf16.pth"),
                tokenizer_path=os.path.join(base_model_dir, "google", "umt5-xxl"),
            )

        with CudaTimer("Load VAE"):
            self.vae = Wan2_1_VAE(
                vae_pth=os.path.join(base_model_dir, "Wan2.1_VAE.pth"),
                device=device,
            )
        self.vae_stride = (4, 8, 8)
        self.patch_size = (1, 2, 2)

        with WallTimer("Load Fast DiT (device_map=auto)"):
            self.model = WanModel.from_pretrained(
                fast_model_dir,
                torch_dtype=dtype,
                control_type="cam",
                device_map="auto",
            )
            self.model.eval().requires_grad_(False)

        # Negative prompt (from config)
        self.neg_prompt = (
            "画面突变，色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，"
            "整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，"
            "画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，"
            "静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
        )

        # Session state (set by start_session)
        self._context = None
        self._context_null = None
        self._current_pose = None
        self._last_frame_tensor = None   # [3, H, W] in [-1, 1]
        self._h = 0
        self._w = 0
        self._lat_h = 0
        self._lat_w = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start_session(
        self,
        image_tensor: torch.Tensor,
        prompt: str,
        height: int = 480,
        width: int = 832,
    ) -> list[np.ndarray]:
        """
        Start a new interactive session.

        Args:
            image_tensor: [3, H, W] float tensor in [0, 1]
            prompt: text description
            height, width: output resolution

        Returns:
            List of RGB frames as uint8 numpy arrays [H, W, 3]
        """
        img = image_tensor.to(self.device) * 2.0 - 1.0  # to [-1, 1]

        self._h = height
        self._w = width
        self._lat_h = round(
            np.sqrt(height * width * (height / width))
            // self.vae_stride[1] // self.patch_size[1] * self.patch_size[1]
        )
        self._lat_w = round(
            np.sqrt(height * width / (height / width))
            // self.vae_stride[2] // self.patch_size[2] * self.patch_size[2]
        )
        # Recompute exact pixel dims
        self._h = self._lat_h * self.vae_stride[1]
        self._w = self._lat_w * self.vae_stride[2]

        with CudaTimer("T5 to GPU"):
            self.text_encoder.model.to(self.device)
        with CudaTimer("Encode prompt"):
            self._context = self.text_encoder([prompt], self.device)
        with CudaTimer("Encode neg prompt"):
            self._context_null = self.text_encoder([self.neg_prompt], self.device)
        with CudaTimer("T5 offload to CPU"):
            self.text_encoder.model.cpu()
            torch.cuda.empty_cache()

        # Init camera at identity
        self._current_pose = np.eye(4)

        # Resize input image
        img_resized = torch.nn.functional.interpolate(
            img[None], size=(self._h, self._w), mode="bicubic"
        ).squeeze(0)  # [3, h, w]

        # Generate first chunk with no camera movement + torch profiler
        log.info("Running first chunk with torch.profiler...")
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
        ) as prof:
            frames = self._generate_chunk(img_resized, action="idle")

        log.info("\n" + prof.key_averages().table(
            sort_by="cuda_time_total", row_limit=25
        ))
        # Export chrome trace for detailed analysis if needed
        prof.export_chrome_trace("/tmp/lingbot_first_chunk_trace.json")
        log.info("Chrome trace saved to /tmp/lingbot_first_chunk_trace.json")

        return frames

    def step(self, action: str = "forward") -> list[np.ndarray]:
        """
        Generate the next chunk given a user camera action.

        Args:
            action: one of forward/back/left/right/up/down/
                    turn_left/turn_right/look_up/look_down/idle

        Returns:
            List of new RGB frames as uint8 numpy [H, W, 3]
        """
        assert self._last_frame_tensor is not None, "Call start_session first"
        return self._generate_chunk(self._last_frame_tensor, action=action)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _generate_chunk(
        self, init_img: torch.Tensor, action: str
    ) -> list[np.ndarray]:
        """
        Generate one chunk of video frames.

        Args:
            init_img: [3, h, w] tensor in [-1, 1]
            action: camera action string
        """
        F = self.CHUNK_FRAMES
        h, w = self._h, self._w
        lat_h, lat_w = self._lat_h, self._lat_w
        lat_f = (F - 1) // self.vae_stride[0] + 1

        max_seq_len = lat_f * lat_h * lat_w // (
            self.patch_size[1] * self.patch_size[2]
        )

        _t = []  # deferred timers — flushed at end of chunk
        with CudaTimer("Build camera poses + Plucker embeddings", defer=True, collect=_t):
            poses_np, new_pose = build_chunk_poses(
                self._current_pose, action, n_frames=F
            )
            self._current_pose = new_pose

            Ks = make_default_intrinsics(F, h=480, w=832)
            Ks = get_Ks_transformed(
                Ks, height_org=480, width_org=832,
                height_resize=h, width_resize=w,
                height_final=h, width_final=w,
            )
            Ks = Ks[0]

            c2ws_infer = interpolate_camera_poses(
                src_indices=np.linspace(0, F - 1, F),
                src_rot_mat=poses_np[:, :3, :3],
                src_trans_vec=poses_np[:, :3, 3],
                tgt_indices=np.linspace(0, F - 1, lat_f),
            )
            c2ws_infer = compute_relative_poses(c2ws_infer, framewise=True)
            Ks_rep = Ks.repeat(len(c2ws_infer), 1)

            c2ws_infer = c2ws_infer.to(self.device)
            Ks_rep = Ks_rep.to(self.device)

            c2ws_plucker_emb = get_plucker_embeddings(
                c2ws_infer, Ks_rep, h, w, only_rays_d=False
            )
            c2ws_plucker_emb = rearrange(
                c2ws_plucker_emb,
                "f (h c1) (w c2) c -> (f h w) (c c1 c2)",
                c1=int(h // lat_h),
                c2=int(w // lat_w),
            )
            c2ws_plucker_emb = c2ws_plucker_emb[None, ...]
            c2ws_plucker_emb = rearrange(
                c2ws_plucker_emb,
                "b (f h w) c -> b c f h w",
                f=lat_f, h=lat_h, w=lat_w,
            ).to(self.dtype)

            dit_cond_dict = {"c2ws_plucker_emb": c2ws_plucker_emb.chunk(1, dim=0)}

        with CudaTimer("VAE encode init image", defer=True, collect=_t):
            video_input = torch.cat(
                [
                    init_img[:, None, :, :].cpu(),
                    torch.zeros(3, F - 1, h, w),
                ],
                dim=1,
            ).to(self.device)

            y = self.vae.encode([video_input])[0]

        # Build mask: 1 for first frame, 0 for rest
        msk = torch.ones(1, F, lat_h, lat_w, device=self.device)
        msk[:, 1:] = 0
        msk = torch.cat(
            [torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]],
            dim=1,
        )
        msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
        msk = msk.transpose(1, 2)[0]  # [4, lat_f, lat_h, lat_w]
        y = torch.cat([msk, y])        # [20, lat_f, lat_h, lat_w]

        # --- Noise + diffusion sampling ---
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(random.randint(0, sys.maxsize))

        noise = torch.randn(
            16, lat_f, lat_h, lat_w,
            dtype=torch.float32,
            generator=seed_g,
            device=self.device,
        )

        scheduler = FlowUniPCMultistepScheduler(
            num_train_timesteps=self.NUM_TRAIN_TIMESTEPS,
            shift=1,
            use_dynamic_shifting=False,
        )
        scheduler.set_timesteps(
            self.SAMPLE_STEPS, device=self.device, shift=self.SAMPLE_SHIFT
        )

        arg_c = {
            "context": [self._context[0]],
            "seq_len": max_seq_len,
            "y": [y],
            "dit_cond_dict": dit_cond_dict,
        }
        arg_null = {
            "context": self._context_null,
            "seq_len": max_seq_len,
            "y": [y],
            "dit_cond_dict": dit_cond_dict,
        }

        latent = noise
        log.info(f"Starting denoising loop: {self.SAMPLE_STEPS} steps")
        step_events = []  # (label, start_event, end_event) for per-step timing
        with CudaTimer(f"Denoising ({self.SAMPLE_STEPS} steps total)", defer=True, collect=_t):
            with torch.amp.autocast("cuda", dtype=self.dtype), torch.no_grad():
                for step_i, t in enumerate(scheduler.timesteps):
                    se = torch.cuda.Event(enable_timing=True)
                    ee = torch.cuda.Event(enable_timing=True)
                    se.record()

                    timestep = torch.stack([t]).to(self.device)
                    inp = [latent.to(self.device)]

                    pred_cond = self.model(inp, t=timestep, **arg_c)[0]
                    pred_uncond = self.model(inp, t=timestep, **arg_null)[0]

                    pred = pred_uncond + self.GUIDE_SCALE * (pred_cond - pred_uncond)

                    latent = scheduler.step(
                        pred.unsqueeze(0), t, latent.unsqueeze(0),
                        return_dict=False, generator=seed_g,
                    )[0].squeeze(0)

                    ee.record()
                    if step_i < 3 or step_i == self.SAMPLE_STEPS - 1:
                        step_events.append((f"  Step {step_i+1}/{self.SAMPLE_STEPS}", se, ee))

        with CudaTimer("VAE decode", defer=True, collect=_t):
            with torch.no_grad():
                videos = self.vae.decode([latent])
        video = videos[0]  # [3, F, H, W] in [-1, 1]

        # Save last frame for next chunk
        self._last_frame_tensor = video[:, -1, :, :].clone()  # [3, H, W]

        # Convert to uint8 numpy frames
        frames = []
        for i in range(video.shape[1]):
            frame = video[:, i, :, :].clamp(-1, 1).add(1).div(2)  # [0, 1]
            frame = (frame.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            frames.append(frame)

        # Flush all deferred timers + per-step events (single sync point)
        CudaTimer.flush(_t)
        for label, se, ee in step_events:
            log.info(f"[PROFILE] {label}: {se.elapsed_time(ee):.1f}ms")

        return frames

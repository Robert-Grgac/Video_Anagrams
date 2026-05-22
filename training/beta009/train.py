"""BETA9 training: additive face-target loss with configurable target image.

Replaces beta-008's spatially-weighted ``(1 + α · mask) · MSE`` with the
additive decomposition

    loss = MSE(v_pred, v_target)
         + λ_face · mean_face_region( MSE(v_pred, v_face_target) )
         + λ_consistency · MSE(v_pred, v_pred_ema)        # if self-distill on

where ``v_face_target = noise − z_face`` is the FM velocity that would point
toward a *different* clean latent — the VAE-encoded face image. We use the
SAME ``noise`` and ``sigma`` sampled for the step, so ``v_face_target`` and
``v_pred`` live in the same v-space and their MSE is well-defined. (Note:
because ``noise`` cancels in the difference, this is algebraically identical
to comparing the model's clean prediction ``z_t − σ · v_pred`` to ``z_face``
in the face region — just expressed in v-space for symmetry with the FM loss.)

The face-region MSE is normalized by the mass of the mask (so λ_face is
interpretable as "extra MSEs of face-region accuracy" regardless of how
much of the frame the face covers). The full-frame ``loss_fm`` is the
unweighted beta-007 quantity, so curves are directly comparable to beta-007.

Two target choices controlled by ``--face_target_subdir``:

- ``raw_face``: pulls the face region toward identity + interior features
  (encoded raw face image — pixel-level face content).
- ``silhouette``: pulls the face region toward shape/contour only, no
  identity (encoded silhouette drawing).

Both load (3, H, W) uint8 tensors from ``<cache_dir>/<face_target_subdir>/face_{idx}.pt``
produced by ``precompute_raw_face.py`` / ``precompute_silhouette.py``, then
VAE-encode them on-the-fly at startup (~30s for 100 faces) and cache the
normalized latents in GPU memory keyed by ``face_idx``. No new precompute
pass required.

Everything else matches the beta-007/008 recipe: 1 epoch (309 effective
steps) on the high-noise expert, single-CN cold-start, EMA(0.99,
update_after=10), self-distillation OFF-by-default behind
``--use_self_distillation``, manual-gate Accelerate accumulation. Eval
protocol is unchanged from beta-008.
"""
from __future__ import annotations

import argparse
import atexit
import gc
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from training.dataset_beta import BetaPairDataset
from training.utils import (
    cast_respecting_fp32_modules,
    detect_boundary_ratio,
    mean_residual_l2,
    _collate_keep_meta,
    _save_safetensors,
    _format_seconds,
)
from training.utils import (
    _maybe_force_native_attention,
    _build_controlnet_from_checkpoint,
    _save_state_dict,
)
from wan_t2v_controlnet_pipeline import prompt_clean

print("[boot] module-level imports complete", flush=True)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _git_sha() -> str:
    try:
        import subprocess
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(Path(__file__).parent.parent.parent),
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


WAN_T5_MAX_SEQ_LEN = 226


# ---------------- args ----------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--cache_dir", type=str, required=True)
    p.add_argument("--control_subdir", type=str, default="silhouette",
                   help="Subdir of cache_dir from which to load the ControlNet "
                        "input modality. Defaults to 'silhouette' (matches the "
                        "beta-007_silhouette / beta-008 recipe).")
    p.add_argument("--face_mask_subdir", type=str, default="silhouette",
                   help="Cache subdir to load per-face binary masks from. The "
                        "(3, H, W) uint8 silhouette is thresholded to a {0,1} "
                        "mask and avg-pooled to latent resolution. The mask "
                        "decides WHERE the additive face loss applies; it is "
                        "independent of --face_target_subdir and "
                        "--control_subdir.")
    p.add_argument("--face_target_subdir", type=str, default="raw_face",
                   choices=("raw_face", "silhouette"),
                   help="Which precomputed face image to VAE-encode as the "
                        "additive loss target. 'raw_face' pulls face region "
                        "toward identity + interior features; 'silhouette' "
                        "pulls toward shape/contour only. Loaded from "
                        "<cache_dir>/<face_target_subdir>/face_{idx}.pt.")
    p.add_argument("--targets_dir", type=str, required=True,
                   help="Directory containing PTDiffusion target JPGs named "
                        "face_{idx}_{slug}.jpg. Used as the MSE reference at eval.")
    p.add_argument("--base_model_path", type=str, required=True)
    p.add_argument("--controlnet_config_repo", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--inference_output_dir", type=str, default=None)
    p.add_argument("--card_path", type=str, default=None)
    p.add_argument("--wandb_project", type=str, default="wan-controlnet-beta")
    p.add_argument("--run_name", type=str, required=True)

    p.add_argument("--num_frames", type=int, default=9)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=512)

    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--lr_decay_per_epoch", type=float, default=1.0)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--num_epochs", type=int, default=1)
    p.add_argument("--max_steps", type=int, default=None)
    p.add_argument("--micro_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=32)
    p.add_argument("--ema_decay", type=float, default=0.99)
    p.add_argument("--ema_update_after_step", type=int, default=10)
    p.add_argument("--loss_ema_window", type=int, default=20)

    p.add_argument("--num_cn_layers", type=int, default=None)
    p.add_argument("--controlnet_stride", type=int, default=3)

    p.add_argument("--use_self_distillation", action="store_true",
                   help="Add lambda_consistency * MSE(v_pred_live, v_pred_ema).")
    p.add_argument("--lambda_consistency", type=float, default=0.5)

    # Additive face-target loss.
    p.add_argument("--lambda_face", type=float, default=5.0,
                   help="Coefficient on the additive face-region MSE term. "
                        "The face term is normalized by mask mass, so λ_face=5 "
                        "means ~5× the per-element gradient strength inside "
                        "the face region relative to the full-frame loss_fm. "
                        "Set to 0 to disable.")

    p.add_argument("--num_train_timesteps_for_sampling", type=int, default=1000)
    p.add_argument("--boundary_ratio_override", type=float, default=None)
    p.add_argument("--checkpoint_every", type=int, default=50)
    p.add_argument("--memory_tripwire_gb", type=float, default=90.0)

    p.add_argument("--eval_size", type=int, default=100)
    p.add_argument("--periodic_eval_size", type=int, default=10)
    p.add_argument("--periodic_eval_every", type=int, default=10)
    p.add_argument("--inference_steps", type=int, default=50)
    p.add_argument("--inference_guidance_scale", type=float, default=5.0)
    p.add_argument("--inference_controlnet_weight", type=float, default=1.0)
    p.add_argument("--inference_controlnet_end", type=float, default=None,
                   help="If unset, computed dynamically so the CN runs only "
                        "while σ ≥ boundary_ratio (high-noise expert).")
    p.add_argument("--negative_prompt", type=str, default="bad quality, worst quality")
    p.add_argument("--skip_periodic_eval", action="store_true")
    p.add_argument("--skip_final_eval", action="store_true")

    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


_RESULTS_STATE: dict = {}
_RESULTS_PATH: Optional[Path] = None
_CARD_PATH: Optional[Path] = None
_EVAL_LOG_PATH: Optional[Path] = None
_EVAL_LOG: dict = {"periodic": [], "final": []}


def _write_results() -> None:
    if _RESULTS_PATH is None:
        return
    try:
        _RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        _RESULTS_PATH.write_text(json.dumps(_RESULTS_STATE, indent=2, default=str))
    except Exception as e:
        print(f"[results] WARN failed to write {_RESULTS_PATH}: {e}",
              file=sys.stderr)


def _write_eval_log() -> None:
    if _EVAL_LOG_PATH is None:
        return
    try:
        _EVAL_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        _EVAL_LOG_PATH.write_text(json.dumps(_EVAL_LOG, indent=2, default=str))
    except Exception as e:
        print(f"[eval] WARN failed to write {_EVAL_LOG_PATH}: {e}",
              file=sys.stderr)


def _atexit_hook() -> None:
    if _RESULTS_STATE.get("status") not in (None, "running"):
        return
    _RESULTS_STATE["status"] = "failed"
    _RESULTS_STATE["date_finished"] = _now_iso()
    if "date_started" in _RESULTS_STATE:
        try:
            t0 = datetime.fromisoformat(_RESULTS_STATE["date_started"])
            t1 = datetime.fromisoformat(_RESULTS_STATE["date_finished"])
            _RESULTS_STATE["wall_time"] = str(t1 - t0)
        except Exception:
            pass
    _write_results()
    _write_eval_log()
    if _CARD_PATH is not None:
        try:
            from training.autofill_card import autofill
            autofill(_CARD_PATH)
        except Exception as e:
            print(f"[autofill] atexit fill failed: {e}", file=sys.stderr)


# ------------- eval helpers (identical to beta-008) -------------

def _load_target_image(targets_dir: Path, face_idx: int, slug: str,
                       height: int, width: int) -> np.ndarray:
    from PIL import Image
    p = targets_dir / f"face_{face_idx}_{slug}.jpg"
    if not p.exists():
        raise FileNotFoundError(f"Target JPG not found: {p}")
    img = Image.open(str(p)).convert("RGB")
    if img.size != (width, height):
        img = img.resize((width, height), Image.BICUBIC)
    return np.asarray(img, dtype=np.float32) / 255.0


def _save_video(frames_np: np.ndarray, path: Path, fps: int = 8) -> None:
    import imageio
    arr = (np.clip(frames_np, 0.0, 1.0) * 255).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(path), list(arr), fps=fps, codec="libx264")


def _frames_target_mse(frames: np.ndarray, target_hwc: np.ndarray) -> float:
    target_T = np.broadcast_to(target_hwc[None, ...], frames.shape)
    diff = frames.astype(np.float32) - target_T.astype(np.float32)
    return float(np.mean(diff * diff))


def _frames_target_ssim(frames: np.ndarray, target_hwc: np.ndarray,
                        device: Optional[str] = None) -> float:
    from pytorch_msssim import ssim
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    f = torch.from_numpy(np.ascontiguousarray(frames)).permute(0, 3, 1, 2).float().to(dev)
    t = (torch.from_numpy(np.ascontiguousarray(target_hwc))
         .permute(2, 0, 1).float().unsqueeze(0).to(dev)
         .expand(f.shape[0], -1, -1, -1).contiguous())
    with torch.no_grad():
        s = ssim(f, t, data_range=1.0, size_average=True)
    return float(s.item())


def _build_eval_periodic_splits(records: list[dict]) -> tuple[list[int], list[int], list[int]]:
    faces = sorted({r["face_idx"] for r in records})
    slugs = sorted({r["slug"] for r in records})
    if len(faces) != 100 or len(slugs) != 100:
        raise RuntimeError(
            f"Expected 100 distinct faces and 100 distinct slugs; "
            f"got {len(faces)} faces and {len(slugs)} slugs."
        )
    eval_pair_set = {(faces[i], slugs[i]) for i in range(100)}
    periodic_pair_set = {(faces[i], slugs[(i + 50) % 100]) for i in range(10)}
    assert eval_pair_set.isdisjoint(periodic_pair_set)

    by_pair: dict[tuple[int, str], int] = {}
    for idx, r in enumerate(records):
        key = (r["face_idx"], r["slug"])
        if key in by_pair:
            raise RuntimeError(f"Duplicate (face_idx, slug) pair: {key}")
        by_pair[key] = idx

    eval_indices = sorted(by_pair[p] for p in eval_pair_set)
    periodic_indices = sorted(by_pair[p] for p in periodic_pair_set)
    eval_idx_set = set(eval_indices)
    train_indices = [i for i in range(len(records)) if i not in eval_idx_set]

    train_idx_set = set(train_indices)
    assert all(i in train_idx_set for i in periodic_indices)
    assert len(eval_indices) == 100 and len(periodic_indices) == 10
    return train_indices, eval_indices, periodic_indices


def _compute_cn_end_high_noise(base_model_path: str, num_inference_steps: int,
                               boundary_ratio: float, device: torch.device) -> tuple[float, int]:
    from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
    sched = FlowMatchEulerDiscreteScheduler.from_pretrained(
        base_model_path, subfolder="scheduler",
    )
    sched.set_timesteps(num_inference_steps, device=device)
    sigmas = sched.sigmas[:-1].detach().cpu()
    below = (sigmas < boundary_ratio).nonzero(as_tuple=False)
    if below.numel() == 0:
        return 1.0, num_inference_steps
    first_low = int(below[0].item())
    return first_low / num_inference_steps, first_low


# ------------- z_face encoding (the only new helper) -------------

def encode_face_targets_to_latent(
    face_image_dir: Path,
    unique_face_idxs: list[int],
    vae,
    num_frames: int,
    height: int,
    width: int,
    device: torch.device,
) -> dict[int, torch.Tensor]:
    """VAE-encode each face image as the additive-loss target latent.

    Mirrors ``precompute_beta.encode_latent``: load (3, H, W) uint8, convert
    to [-1, 1] float, replicate to (3, T, H, W), encode, normalize by
    ``vae.config.latents_{mean,std}``, store as bf16 on device.

    Returns ``{face_idx: z_face}`` with each ``z_face`` of shape
    ``(C, T_lat, H_lat, W_lat)`` — matches z_real in shape, so the additive
    loss term has the same broadcasting pattern as ``loss_fm``.
    """
    if not face_image_dir.exists():
        raise FileNotFoundError(
            f"face_target_subdir resolves to {face_image_dir}, which does not "
            f"exist. Run precompute_raw_face.py or precompute_silhouette.py."
        )

    z_dim = vae.config.z_dim
    latents_mean = torch.tensor(
        vae.config.latents_mean, dtype=vae.dtype, device=device,
    ).view(1, z_dim, 1, 1, 1)
    latents_std = torch.tensor(
        vae.config.latents_std, dtype=vae.dtype, device=device,
    ).view(1, z_dim, 1, 1, 1)

    z_faces: dict[int, torch.Tensor] = {}
    with torch.no_grad():
        for fi in unique_face_idxs:
            p = face_image_dir / f"face_{fi}.pt"
            if not p.exists():
                raise FileNotFoundError(
                    f"z_face source missing for face_idx={fi}: expected {p}"
                )
            img_u8 = torch.load(p, map_location="cpu",
                                weights_only=True)  # (3, H, W) uint8
            if img_u8.shape != (3, height, width):
                raise RuntimeError(
                    f"face_{fi}.pt has shape {tuple(img_u8.shape)}, expected "
                    f"(3, {height}, {width})"
                )
            x = img_u8.float() / 127.5 - 1.0           # (3, H, W) in [-1, 1]
            x = x.unsqueeze(1).expand(-1, num_frames, -1, -1)  # (3, T, H, W)
            x = x.unsqueeze(0).to(device=device, dtype=vae.dtype)
            z = vae.encode(x).latent_dist.mean         # (1, C, T_lat, H_lat, W_lat)
            z_norm = (z - latents_mean) / latents_std
            z_faces[int(fi)] = z_norm.squeeze(0).to(torch.bfloat16).contiguous()

    return z_faces


# ------------- main -------------

def main() -> None:
    global _RESULTS_PATH, _CARD_PATH, _EVAL_LOG_PATH
    print("[boot] main() entered", flush=True)
    cfg = parse_args()
    print("[boot] args parsed", flush=True)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    inf_out_dir = Path(cfg.inference_output_dir) if cfg.inference_output_dir \
        else output_dir.parent / "outputs"
    inf_out_dir = inf_out_dir / cfg.run_name
    (inf_out_dir / "periodic").mkdir(parents=True, exist_ok=True)
    (inf_out_dir / "final").mkdir(parents=True, exist_ok=True)

    if cfg.card_path:
        _CARD_PATH = Path(cfg.card_path)
        _RESULTS_PATH = _CARD_PATH.parent / f"{cfg.run_name}_results.json"
        _EVAL_LOG_PATH = _CARD_PATH.parent / f"{cfg.run_name}_eval.json"
    else:
        _RESULTS_PATH = Path("training_cards") / "beta009" / f"{cfg.run_name}_results.json"
        _EVAL_LOG_PATH = Path("training_cards") / "beta009" / f"{cfg.run_name}_eval.json"

    init_mode = "cold"
    effective_batch = cfg.micro_batch_size * cfg.gradient_accumulation_steps
    _RESULTS_STATE.update({
        "status": "running",
        "date_started": _now_iso(),
        "git_sha": _git_sha(),
        "cluster_partition": os.environ.get("SLURM_JOB_PARTITION", "unknown"),
        "run_name": cfg.run_name,
        "init_mode": init_mode,
        "expert_trained": "high-noise (transformer)",
        "micro_batch_size": cfg.micro_batch_size,
        "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
        "effective_batch_size": effective_batch,
        "ema_decay": cfg.ema_decay,
        "ema_update_after_step": cfg.ema_update_after_step,
        "loss_ema_window": cfg.loss_ema_window,
        "num_cn_layers_override": cfg.num_cn_layers,
        "controlnet_stride": cfg.controlnet_stride,
        "use_self_distillation": cfg.use_self_distillation,
        "lambda_consistency": cfg.lambda_consistency if cfg.use_self_distillation else None,
        "lambda_face": cfg.lambda_face,
        "control_subdir": cfg.control_subdir,
        "face_mask_subdir": cfg.face_mask_subdir,
        "face_target_subdir": cfg.face_target_subdir,
        "eval_size": cfg.eval_size,
        "periodic_eval_size": cfg.periodic_eval_size,
        "periodic_eval_every": cfg.periodic_eval_every,
        "inference_steps": cfg.inference_steps,
    })
    _write_results()
    print("[boot] initial _write_results done", flush=True)
    atexit.register(_atexit_hook)

    import wandb
    print("[boot] wandb imported", flush=True)
    wandb_mode = os.environ.get("WANDB_MODE", "online")
    print(f"[boot] calling wandb.init(mode={wandb_mode}) ...", flush=True)
    wandb_run = wandb.init(
        project=cfg.wandb_project,
        name=cfg.run_name,
        config=vars(cfg),
        mode=wandb_mode,
    )
    print("[boot] wandb.init() returned", flush=True)
    wandb_url = wandb_run.get_url() if wandb_mode == "online" else f"offline:{wandb_run.dir}"
    _RESULTS_STATE["wandb_url"] = wandb_url

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("BETA9 training requires CUDA; got CPU.")
    try:
        major, minor = torch.cuda.get_device_capability(0)
        gpu_name = torch.cuda.get_device_name(0)
        print(f"[gpu] {gpu_name} (compute_cap={major}.{minor})")
        _RESULTS_STATE["gpu_name"] = gpu_name
        _RESULTS_STATE["gpu_compute_cap"] = f"{major}.{minor}"
    except Exception:
        pass

    from accelerate import Accelerator
    accelerator = Accelerator(gradient_accumulation_steps=cfg.gradient_accumulation_steps)
    print(f"[accel] gradient_accumulation_steps={cfg.gradient_accumulation_steps} "
          f"(num_processes={accelerator.num_processes})")

    from wan_controlnet import WanControlnet
    from wan_transformer import CustomWanTransformer3DModel
    from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
    from diffusers import AutoencoderKLWan
    from transformers import AutoTokenizer, UMT5EncoderModel
    print("[boot] core ML imports done (wan_controlnet, wan_transformer, diffusers, transformers)", flush=True)

    print(f"[load] high-noise transformer (transformer) from {cfg.base_model_path} ...")
    transformer = CustomWanTransformer3DModel.from_pretrained(
        cfg.base_model_path, subfolder="transformer", torch_dtype=torch.bfloat16,
    )
    transformer.requires_grad_(False).eval().to(device)
    transformer.enable_gradient_checkpointing()
    _maybe_force_native_attention(transformer, "transformer")

    boundary_ratio, boundary_src = detect_boundary_ratio(
        cfg.base_model_path, dict(transformer.config),
        override=cfg.boundary_ratio_override,
    )

    print(f"[load] controlnet config from {cfg.controlnet_config_repo} ...")
    config = WanControlnet.load_config(cfg.controlnet_config_repo)
    if cfg.num_cn_layers is not None:
        old_L = config.get("num_layers", "?")
        config["num_layers"] = cfg.num_cn_layers
        print(f"[arch] num_layers override: {old_L} -> {cfg.num_cn_layers}")
    controlnet = WanControlnet.from_config(config)
    cast_respecting_fp32_modules(controlnet, torch.bfloat16)
    fp32_params = [n for n, p in controlnet.named_parameters()
                   if p.dtype == torch.float32]
    assert any("norm" in n or "time_embedder" in n or "scale_shift" in n
               for n in fp32_params), \
        "Expected norm/time_embedder/scale_shift params kept in fp32"
    print("[cold-start] using fresh from_config weights (output projections zeroed)")

    controlnet.enable_gradient_checkpointing()
    controlnet.train().to(device)
    _maybe_force_native_attention(controlnet, "controlnet")

    trainable_params = sum(p.numel() for p in controlnet.parameters())
    _RESULTS_STATE["trainable_params"] = trainable_params
    print(f"[controlnet] trainable params: {trainable_params:,}")

    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        cfg.base_model_path, subfolder="scheduler",
    )
    scheduler.set_timesteps(cfg.num_train_timesteps_for_sampling, device=device)
    sigmas = scheduler.sigmas[:-1].to(device)
    timesteps_full = scheduler.timesteps.to(device)

    high_noise_indices = torch.where(sigmas >= boundary_ratio)[0]
    if high_noise_indices.numel() == 0:
        print(f"[warn] No timesteps satisfy sigma >= {boundary_ratio}; "
              "falling back to upper 50%.")
        high_noise_indices = torch.arange(0, len(sigmas) // 2, device=device)
        boundary_src += "+fallback_upper_50pct"
    print(f"[boundary] ratio={boundary_ratio} ({boundary_src}); "
          f"high={len(high_noise_indices)} of {len(sigmas)} timesteps in regime")
    _RESULTS_STATE["boundary_sigma"] = boundary_ratio
    _RESULTS_STATE["high_noise_rule"] = boundary_src

    import bitsandbytes as bnb
    optimizer = bnb.optim.AdamW8bit(
        controlnet.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay,
    )

    from ema_pytorch import EMA
    ema = EMA(
        controlnet,
        beta=cfg.ema_decay,
        update_after_step=cfg.ema_update_after_step,
        update_every=1,
    )
    ema.to(device)

    # --- Train / eval data split ---
    full_dataset = BetaPairDataset(cfg.cache_dir, num_frames=cfg.num_frames,
                                   control_subdir=cfg.control_subdir)
    total_n = len(full_dataset)
    if total_n != 10000:
        raise RuntimeError(
            f"Expected 10000 records (100 faces × 100 prompts); got {total_n}."
        )
    train_indices, eval_indices, periodic_indices = _build_eval_periodic_splits(
        full_dataset.records
    )
    n_train = len(train_indices)
    train_dataset = Subset(full_dataset, train_indices)
    print(f"[data] total={total_n} → train={len(train_dataset)} "
          f"eval={len(eval_indices)} periodic={len(periodic_indices)}")
    _RESULTS_STATE["pair_count"] = n_train
    _RESULTS_STATE["eval_count"] = len(eval_indices)
    _RESULTS_STATE["periodic_eval_count"] = len(periodic_indices)

    # --- Pre-cache the per-face latent mask (where the additive loss applies). ---
    # The Wan VAE is 8× spatial; 512 → 64. Threshold the silhouette to {0,1},
    # avg-pool by 8 to (h_lat, w_lat) — soft values capture partial boundary
    # coverage cleanly. Independent of the face-target choice below.
    mask_dir = Path(cfg.cache_dir) / cfg.face_mask_subdir
    if not mask_dir.exists():
        raise FileNotFoundError(
            f"--face_mask_subdir='{cfg.face_mask_subdir}' resolves to {mask_dir}; "
            "run precompute_silhouette.py first."
        )
    spatial_factor = 8
    h_lat = cfg.height // spatial_factor
    w_lat = cfg.width // spatial_factor
    unique_face_idxs = sorted({r["face_idx"] for r in full_dataset.records})
    face_masks_latent: dict[int, torch.Tensor] = {}
    for fi in unique_face_idxs:
        mask_path = mask_dir / f"face_{fi}.pt"
        if not mask_path.exists():
            raise FileNotFoundError(
                f"face mask missing for face_idx={fi}: expected {mask_path}"
            )
        raw = torch.load(mask_path, map_location="cpu", weights_only=True)
        m_bin = (raw[0] > 0).float()
        m_lat = F.avg_pool2d(
            m_bin.unsqueeze(0).unsqueeze(0),
            kernel_size=spatial_factor, stride=spatial_factor,
        ).squeeze(0).squeeze(0).contiguous()
        face_masks_latent[int(fi)] = m_lat
    coverage_mean = float(
        torch.stack(list(face_masks_latent.values())).mean().item()
    )
    print(f"[mask] cached {len(face_masks_latent)} masks from {mask_dir} "
          f"at {h_lat}×{w_lat}; mean face coverage={coverage_mean:.3f}")
    _RESULTS_STATE["face_mask_coverage_mean"] = round(coverage_mean, 4)
    _RESULTS_STATE["face_mask_latent_hw"] = [h_lat, w_lat]

    train_loader = DataLoader(
        train_dataset, batch_size=cfg.micro_batch_size, shuffle=True,
        num_workers=cfg.num_workers,
        persistent_workers=cfg.num_workers > 0,
        pin_memory=True, drop_last=True,
        collate_fn=_collate_keep_meta,
    )
    micro_steps_per_epoch = len(train_loader)
    max_eff_from_epoch = micro_steps_per_epoch // cfg.gradient_accumulation_steps
    if cfg.max_steps is None:
        cfg.max_steps = max_eff_from_epoch * cfg.num_epochs
    print(f"[data] train: {len(train_dataset)} pairs, {micro_steps_per_epoch} micro-steps/epoch, "
          f"{max_eff_from_epoch} effective-steps/epoch, num_epochs={cfg.num_epochs}, "
          f"max_steps={cfg.max_steps}")

    # --- Pre-load text encoder for neg prompt; drop after. ---
    print(f"[load] tokenizer + text_encoder (one-shot, for neg prompt) ...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model_path, subfolder="tokenizer")
    text_encoder = UMT5EncoderModel.from_pretrained(
        cfg.base_model_path, subfolder="text_encoder", torch_dtype=torch.bfloat16,
    ).eval().to(device)

    print(f"[load] encoding negative prompt embed ('{cfg.negative_prompt}') ...")
    with torch.no_grad():
        neg_inputs = tokenizer(
            [prompt_clean(cfg.negative_prompt)],
            padding="max_length", max_length=WAN_T5_MAX_SEQ_LEN,
            truncation=True, add_special_tokens=True,
            return_attention_mask=True, return_tensors="pt",
        )
        neg_ids = neg_inputs.input_ids.to(device)
        neg_mask = neg_inputs.attention_mask.to(device)
        neg_lens = neg_mask.gt(0).sum(dim=1).long()
        neg_embeds = text_encoder(neg_ids, neg_mask).last_hidden_state.to(torch.bfloat16)
        neg_embeds = [u[:v] for u, v in zip(neg_embeds, neg_lens)]
        neg_embeds = torch.stack(
            [torch.cat([u, u.new_zeros(WAN_T5_MAX_SEQ_LEN - u.size(0), u.size(1))])
             for u in neg_embeds], dim=0
        )
    del text_encoder, tokenizer, neg_inputs, neg_ids, neg_mask, neg_lens
    gc.collect()
    torch.cuda.empty_cache()
    print(f"[load] text_encoder dropped; neg_embeds shape={tuple(neg_embeds.shape)}")

    # --- Load VAE; encode z_face latents (additive-target source); keep VAE for eval. ---
    print(f"[load] vae ...")
    vae = AutoencoderKLWan.from_pretrained(
        cfg.base_model_path, subfolder="vae", torch_dtype=torch.bfloat16,
    ).eval().to(device)

    face_image_dir = Path(cfg.cache_dir) / cfg.face_target_subdir
    print(f"[z_face] encoding {len(unique_face_idxs)} face images from "
          f"{face_image_dir} ({cfg.face_target_subdir}) ...")
    t_enc0 = time.perf_counter()
    z_faces_latent = encode_face_targets_to_latent(
        face_image_dir=face_image_dir,
        unique_face_idxs=unique_face_idxs,
        vae=vae, num_frames=cfg.num_frames,
        height=cfg.height, width=cfg.width, device=device,
    )
    t_enc = time.perf_counter() - t_enc0
    sample_z = next(iter(z_faces_latent.values()))
    print(f"[z_face] encoded {len(z_faces_latent)} latents in {t_enc:.1f}s; "
          f"shape={tuple(sample_z.shape)} dtype={sample_z.dtype} "
          f"device={sample_z.device}")
    _RESULTS_STATE["z_face_count"] = len(z_faces_latent)
    _RESULTS_STATE["z_face_shape"] = list(sample_z.shape)
    _RESULTS_STATE["z_face_encode_seconds"] = round(t_enc, 1)

    print(f"[load] low-noise transformer (transformer_2, eval-only) ...")
    transformer_2 = CustomWanTransformer3DModel.from_pretrained(
        cfg.base_model_path, subfolder="transformer_2", torch_dtype=torch.bfloat16,
    )
    transformer_2.requires_grad_(False).eval().to(device)
    _maybe_force_native_attention(transformer_2, "transformer_2 (eval)")

    # Pre-stage eval sample blobs.
    from PIL import Image
    cache_dir = Path(cfg.cache_dir)
    targets_dir = Path(cfg.targets_dir)
    eval_records = [full_dataset.records[i] for i in eval_indices]
    periodic_records = [full_dataset.records[i] for i in periodic_indices]

    def _stage(records: list[dict]) -> list[dict]:
        out = []
        for j, rec in enumerate(records):
            control_path = cache_dir / cfg.control_subdir / Path(rec["canny_path"]).name
            canny_u8 = torch.load(control_path, map_location="cpu",
                                  weights_only=True)
            canny_img = Image.fromarray(canny_u8.permute(1, 2, 0).numpy())
            prompt_embed = torch.load(cache_dir / rec["prompt_path"], map_location="cpu",
                                      weights_only=True).to(torch.bfloat16)
            if prompt_embed.dim() == 2:
                prompt_embed = prompt_embed.unsqueeze(0)
            assert prompt_embed.shape[1] == WAN_T5_MAX_SEQ_LEN
            target_hwc = _load_target_image(targets_dir, rec["face_idx"], rec["slug"],
                                            cfg.height, cfg.width)
            out.append({
                "eval_idx": j,
                "face_idx": rec["face_idx"],
                "slug": rec["slug"],
                "canny_img": canny_img,
                "prompt_embed": prompt_embed,
                "target_hwc": target_hwc,
            })
        return out

    periodic_eval_samples = _stage(periodic_records)
    eval_samples = _stage(eval_records)
    print(f"[eval] pre-staged periodic={len(periodic_eval_samples)} "
          f"final={len(eval_samples)}")

    # --- Build the inference pipeline once ---
    from wan_t2v_controlnet_pipeline import WanTextToVideoControlnetPipeline

    eval_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        cfg.base_model_path, subfolder="scheduler",
    )
    pipe = WanTextToVideoControlnetPipeline(
        tokenizer=None, text_encoder=None,
        transformer=transformer, transformer_2=transformer_2, vae=vae,
        controlnet=controlnet, scheduler=eval_scheduler,
        boundary_ratio=boundary_ratio,
    )
    print(f"[pipe] inference pipeline built (no CPU offload, all modules resident)")

    if cfg.inference_controlnet_end is None:
        cn_end_fraction, first_low_idx = _compute_cn_end_high_noise(
            cfg.base_model_path, cfg.inference_steps, boundary_ratio, device,
        )
        print(f"[cn-end] high-noise-only: σ first drops below {boundary_ratio} at "
              f"step {first_low_idx}/{cfg.inference_steps} → cn_end={cn_end_fraction:.4f}")
    else:
        cn_end_fraction = float(cfg.inference_controlnet_end)
        print(f"[cn-end] override from --inference_controlnet_end={cn_end_fraction}")
    _RESULTS_STATE["cn_end_fraction"] = round(cn_end_fraction, 6)

    # --- Training-loop scratch ---
    global_step = 0
    micro_step = 0
    final_loss = float("nan")
    grad_assert_done = False

    accum_losses: list[torch.Tensor] = []
    accum_losses_fm: list[torch.Tensor] = []
    accum_losses_face: list[torch.Tensor] = []
    accum_losses_consist: list[torch.Tensor] = []
    accum_residual_l2: list[float] = []
    accum_sigmas: list[float] = []
    accum_t_last = 0.0

    eff_step_log: list[tuple[float, float]] = []
    loss_ema_value: Optional[float] = None
    loss_ema_alpha = 2.0 / (cfg.loss_ema_window + 1)

    optimizer.zero_grad(set_to_none=True)
    t_train_start = time.perf_counter()

    done = False
    for epoch in range(cfg.num_epochs):
        if done:
            break
        for step, batch in enumerate(train_loader):
            with accelerator.accumulate(controlnet):
                canny = batch["canny"].to(device, non_blocking=True)
                z_real = batch["latent"].to(device, non_blocking=True)
                prompt_embeds = batch["prompt_embeds"].to(device, non_blocking=True)
                B = z_real.shape[0]
                face_idxs_b = batch["face_idx"]

                sel = torch.randint(0, len(high_noise_indices), (B,), device=device)
                t_idx = high_noise_indices[sel]
                sigma = sigmas[t_idx].to(z_real.dtype)
                t = timesteps_full[t_idx]
                sigma_b = sigma.view(B, 1, 1, 1, 1)

                noise = torch.randn_like(z_real)
                z_t = (1.0 - sigma_b) * z_real + sigma_b * noise
                v_target = (noise - z_real).float()

                # Build the additive face target in v-space at the SAME noise/σ
                # as v_target. Mathematically MSE(v_pred, v_face_target) is the
                # same as MSE(z_t - σ·v_pred, z_face) — see module docstring.
                z_face_b = torch.stack(
                    [z_faces_latent[int(fi)] for fi in face_idxs_b], dim=0,
                ).to(device=z_real.device, dtype=z_real.dtype)
                v_face_target = (noise - z_face_b).float()

                # Self-distillation teacher forward (unchanged from beta-008).
                v_pred_ema = None
                if cfg.use_self_distillation:
                    with torch.no_grad():
                        cn_states_ema = ema.ema_model(
                            hidden_states=z_t, timestep=t,
                            encoder_hidden_states=prompt_embeds,
                            controlnet_states=canny, return_dict=False,
                        )[0]
                        cn_for_tx_ema = [s.to(dtype=transformer.dtype)
                                         for s in cn_states_ema] \
                            if isinstance(cn_states_ema, (tuple, list)) \
                            else cn_states_ema.to(dtype=transformer.dtype)
                        v_pred_ema = transformer(
                            hidden_states=z_t, timestep=t,
                            encoder_hidden_states=prompt_embeds,
                            controlnet_states=cn_for_tx_ema,
                            controlnet_weight=1.0,
                            controlnet_stride=cfg.controlnet_stride,
                            return_dict=False,
                        )[0].float()
                        del cn_states_ema, cn_for_tx_ema

                controlnet_states = controlnet(
                    hidden_states=z_t, timestep=t,
                    encoder_hidden_states=prompt_embeds,
                    controlnet_states=canny, return_dict=False,
                )[0]
                cn_for_tx = [s.to(dtype=transformer.dtype) for s in controlnet_states] \
                    if isinstance(controlnet_states, (tuple, list)) \
                    else controlnet_states.to(dtype=transformer.dtype)

                v_pred = transformer(
                    hidden_states=z_t, timestep=t,
                    encoder_hidden_states=prompt_embeds,
                    controlnet_states=cn_for_tx,
                    controlnet_weight=1.0,
                    controlnet_stride=cfg.controlnet_stride,
                    return_dict=False,
                )[0]

                # --- Losses ---
                v_pred_f = v_pred.float()
                # 1. Full-frame FM loss (unweighted, matches beta-007 baseline).
                loss_fm = F.mse_loss(v_pred_f, v_target)
                # 2. Additive face-region term: face-region-averaged MSE
                #    between v_pred and v_face_target. Normalized by mask
                #    mass × C × T_lat so the unit matches loss_fm.
                masks_b = torch.stack(
                    [face_masks_latent[int(fi)] for fi in face_idxs_b], dim=0,
                ).to(device=v_pred_f.device, dtype=torch.float32)
                masks_b = masks_b.unsqueeze(1).unsqueeze(1)  # (B, 1, 1, h_lat, w_lat)
                diff2_face = (v_pred_f - v_face_target) ** 2
                denom_face = (masks_b.sum() *
                              v_pred_f.shape[1] *
                              v_pred_f.shape[2]).clamp_min(1.0)
                loss_face = (masks_b * diff2_face).sum() / denom_face
                # 3. Self-distillation consistency (unchanged). When the
                #    flag is off, loss_consistency is exactly zero and adds
                #    no gradient — same total as without the term.
                if cfg.use_self_distillation and v_pred_ema is not None:
                    loss_consistency = F.mse_loss(v_pred_f, v_pred_ema)
                else:
                    loss_consistency = torch.zeros((), device=v_pred_f.device)

                loss = (
                    loss_fm
                    + cfg.lambda_face * loss_face
                    + cfg.lambda_consistency * loss_consistency
                )

                accelerator.backward(loss)

                accum_losses.append(loss.detach())
                accum_losses_fm.append(loss_fm.detach())
                accum_losses_face.append(loss_face.detach())
                accum_losses_consist.append(loss_consistency.detach())
                accum_residual_l2.append(mean_residual_l2(controlnet_states))
                accum_sigmas.extend(sigma.detach().float().cpu().tolist())
                accum_t_last = float(t[-1].item())
                micro_step += 1

                if not grad_assert_done:
                    tx_with_grad = [
                        n for n, p in transformer.named_parameters()
                        if p.grad is not None and p.grad.abs().sum() > 0
                    ]
                    assert not tx_with_grad, (
                        f"Transformer should have no grads but found {len(tx_with_grad)}"
                    )
                    cn_with_grad = any(
                        p.grad is not None and p.grad.abs().sum() > 0
                        for p in controlnet.parameters()
                    )
                    assert cn_with_grad, "No controlnet param has nonzero grad after step 1"
                    grad_assert_done = True
                    print("[assert] grad-flow check passed at micro-step 1")

            if accelerator.sync_gradients:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    controlnet.parameters(), cfg.grad_clip
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                ema.update()
                global_step += 1
                accum_loss_mean = float(torch.stack(accum_losses).mean().item())
                accum_loss_fm_mean = float(torch.stack(accum_losses_fm).mean().item())
                accum_loss_face_mean = float(torch.stack(accum_losses_face).mean().item())
                accum_loss_consist_mean = float(torch.stack(accum_losses_consist).mean().item())
                final_loss = accum_loss_mean
                sigma_mean = float(np.mean(accum_sigmas))
                sigma_std = float(np.std(accum_sigmas)) if len(accum_sigmas) > 1 else 0.0
                eff_step_log.append((sigma_mean, accum_loss_mean))

                if loss_ema_value is None:
                    loss_ema_value = accum_loss_mean
                else:
                    loss_ema_value = (loss_ema_alpha * accum_loss_mean
                                      + (1.0 - loss_ema_alpha) * loss_ema_value)

                peak_mem = torch.cuda.max_memory_allocated() / 1e9
                try:
                    ema_decay_current = float(ema.get_current_decay())
                except Exception:
                    ema_decay_current = float(getattr(ema, "beta", cfg.ema_decay))
                wandb.log({
                    "loss": accum_loss_mean,
                    "loss_fm": accum_loss_fm_mean,
                    "loss_face": accum_loss_face_mean,
                    "loss_consistency": accum_loss_consist_mean,
                    "loss_ema": loss_ema_value,
                    "grad_norm": grad_norm.item() if torch.is_tensor(grad_norm) else float(grad_norm),
                    "lr": optimizer.param_groups[0]["lr"],
                    "controlnet_residual_norm": float(np.mean(accum_residual_l2)),
                    "timestep": accum_t_last,
                    "sigma": sigma_mean,
                    "sigma_std": sigma_std,
                    "step": global_step,
                    "epoch": epoch,
                    "gpu_mem_gb": peak_mem,
                    "ema_decay_current": ema_decay_current,
                    "samples_seen": global_step * effective_batch,
                }, step=global_step)

                accum_losses = []
                accum_losses_fm = []
                accum_losses_face = []
                accum_losses_consist = []
                accum_residual_l2 = []
                accum_sigmas = []

                if peak_mem > cfg.memory_tripwire_gb:
                    raise RuntimeError(
                        f"GPU memory {peak_mem:.2f}GB exceeded tripwire "
                        f"{cfg.memory_tripwire_gb}GB at step {global_step}."
                    )

                if (not cfg.skip_periodic_eval
                        and global_step % cfg.periodic_eval_every == 0):
                    try:
                        run_periodic_eval(
                            cfg=cfg, global_step=global_step, pipe=pipe,
                            controlnet=controlnet,
                            eval_samples=periodic_eval_samples,
                            neg_embeds=neg_embeds,
                            inf_out_dir=inf_out_dir / "periodic",
                            wandb=wandb, cn_end_fraction=cn_end_fraction,
                        )
                    except Exception as e:
                        print(f"[periodic-eval] step {global_step} failed: {e}")
                        traceback.print_exc()

                if global_step % cfg.checkpoint_every == 0:
                    ema_ckpt = output_dir / f"{cfg.run_name}_step{global_step}.safetensors"
                    _save_state_dict(ema.ema_model.state_dict(), ema_ckpt)
                    print(f"[ckpt] {ema_ckpt}  (EMA)")

                if cfg.max_steps is not None and global_step >= cfg.max_steps:
                    print(f"[stop] reached --max_steps={cfg.max_steps} (effective)")
                    done = True
                    break

        if not done and cfg.lr_decay_per_epoch != 1.0:
            for pg in optimizer.param_groups:
                pg["lr"] *= cfg.lr_decay_per_epoch
            new_lr = optimizer.param_groups[0]["lr"]
            print(f"[lr] epoch {epoch} complete; decayed lr -> {new_lr:.3e}")
            try:
                wandb.log({"lr_epoch_end": new_lr, "epoch_completed": epoch},
                          step=global_step)
            except Exception:
                pass

    if micro_step % cfg.gradient_accumulation_steps != 0:
        leftover = micro_step % cfg.gradient_accumulation_steps
        print(f"[stop] discarding {leftover} micro-step(s) of partial accumulation")
        optimizer.zero_grad(set_to_none=True)

    final_ema = output_dir / f"{cfg.run_name}_final.safetensors"
    final_raw = output_dir / f"{cfg.run_name}_final_raw.safetensors"
    _save_state_dict(ema.ema_model.state_dict(), final_ema)
    _save_safetensors(controlnet, final_raw)
    print(f"[ckpt] {final_ema}  (EMA, canonical)")
    print(f"[ckpt] {final_raw}  (raw, debug)")

    wall_time_s = time.perf_counter() - t_train_start
    high_avg = float(np.mean([ls for _sg, ls in eff_step_log])) if eff_step_log else float("nan")
    _RESULTS_STATE.update({
        "final_loss": round(final_loss, 6),
        "loss_ema_final": (round(float(loss_ema_value), 6)
                           if loss_ema_value is not None else "nan"),
        "gpu_peak_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
        "wall_time": _format_seconds(wall_time_s),
        "wall_time_seconds": round(wall_time_s, 1),
        "global_steps_completed": global_step,
        "micro_steps_completed": micro_step,
        "final_checkpoint": str(final_ema),
        "final_checkpoint_raw": str(final_raw),
        "high_phase_avg_loss": round(high_avg, 6),
    })
    _write_results()

    if not cfg.skip_final_eval:
        try:
            print(f"[final-eval] running {len(eval_samples)} eval inferences on EMA ...")
            ema_cn = _build_controlnet_from_checkpoint(
                cfg.controlnet_config_repo, final_ema,
            )
            cast_respecting_fp32_modules(ema_cn, torch.bfloat16)
            ema_cn.to(device)
            _maybe_force_native_attention(ema_cn, "controlnet (final-eval EMA)")
            pipe.register_modules(controlnet=ema_cn)
            run_final_eval(
                cfg=cfg, pipe=pipe, eval_samples=eval_samples,
                neg_embeds=neg_embeds, inf_out_dir=inf_out_dir / "final",
                wandb=wandb, global_step=global_step,
                cn_end_fraction=cn_end_fraction,
            )
        except Exception as e:
            print(f"[final-eval] failed: {e}")
            traceback.print_exc()
            _RESULTS_STATE["final_eval_error"] = str(e)
            _write_results()

    _write_eval_log()
    _RESULTS_STATE["status"] = "completed"
    _RESULTS_STATE["date_finished"] = _now_iso()
    _write_results()

    if _CARD_PATH is not None:
        try:
            from training.autofill_card import autofill
            autofill(_CARD_PATH)
        except Exception as e:
            print(f"[autofill] WARN: {e}")

    wandb.finish()
    loss_ema_str = f"{loss_ema_value:.4f}" if loss_ema_value is not None else "nan"
    print(f"[done] {cfg.run_name} | loss={final_loss:.4f} | "
          f"loss_ema={loss_ema_str} | wall={_RESULTS_STATE['wall_time']} | "
          f"face_target={cfg.face_target_subdir} λ_face={cfg.lambda_face}")


# ---------------- eval runners (identical to beta-008) ----------------

def _run_one_inference(cfg, pipe, sample: dict, neg_embeds: torch.Tensor,
                       seed: int, cn_end_fraction: float) -> np.ndarray:
    pos_emb = sample["prompt_embed"].to(neg_embeds.device)
    if pos_emb.dim() == 2:
        pos_emb = pos_emb.unsqueeze(0)
    generator = torch.Generator().manual_seed(seed)
    out = pipe(
        controlnet_frames=[sample["canny_img"]] * cfg.num_frames,
        prompt_embeds=pos_emb,
        negative_prompt_embeds=neg_embeds,
        height=cfg.height, width=cfg.width,
        num_frames=cfg.num_frames,
        num_inference_steps=cfg.inference_steps,
        guidance_scale=cfg.inference_guidance_scale,
        controlnet_weight=cfg.inference_controlnet_weight,
        controlnet_stride=cfg.controlnet_stride,
        controlnet_guidance_start=0.0,
        controlnet_guidance_end=cn_end_fraction,
        generator=generator,
        output_type="np",
    )
    return out.frames[0]


def run_periodic_eval(cfg, global_step: int, pipe, controlnet,
                      eval_samples: list, neg_embeds: torch.Tensor,
                      inf_out_dir: Path, wandb,
                      cn_end_fraction: float) -> None:
    inf_out_dir.mkdir(parents=True, exist_ok=True)
    was_training = controlnet.training
    controlnet.eval()
    t0 = time.perf_counter()
    per_sample = []
    try:
        with torch.no_grad():
            for s in eval_samples:
                frames = _run_one_inference(cfg, pipe, s, neg_embeds,
                                            seed=cfg.seed,
                                            cn_end_fraction=cn_end_fraction)
                mse = _frames_target_mse(frames, s["target_hwc"])
                ssim_val = _frames_target_ssim(frames, s["target_hwc"])
                mp4 = inf_out_dir / (
                    f"step{global_step:05d}_sample{s['eval_idx']:03d}"
                    f"_face{s['face_idx']}_{s['slug']}.mp4"
                )
                _save_video(frames, mp4, fps=8)
                per_sample.append({
                    "global_step": global_step,
                    "eval_idx": s["eval_idx"],
                    "face_idx": s["face_idx"],
                    "slug": s["slug"],
                    "mse": mse, "ssim": ssim_val,
                    "mp4": str(mp4),
                })
    finally:
        if was_training:
            controlnet.train()
    avg_mse = float(np.mean([r["mse"] for r in per_sample])) if per_sample else float("nan")
    avg_ssim = float(np.mean([r["ssim"] for r in per_sample])) if per_sample else float("nan")
    elapsed = time.perf_counter() - t0
    log_payload = {"eval/mse_avg": avg_mse, "eval/ssim_avg": avg_ssim,
                   "eval/wall_s": elapsed}
    for r in per_sample:
        log_payload[f"eval/mse_sample_{r['eval_idx']:02d}"] = r["mse"]
        log_payload[f"eval/ssim_sample_{r['eval_idx']:02d}"] = r["ssim"]
    wandb.log(log_payload, step=global_step)
    _EVAL_LOG["periodic"].append({
        "global_step": global_step,
        "avg_mse": avg_mse, "avg_ssim": avg_ssim,
        "wall_s": round(elapsed, 1), "samples": per_sample,
    })
    _write_eval_log()
    print(f"[periodic-eval] step={global_step} avg_mse={avg_mse:.5f} "
          f"avg_ssim={avg_ssim:.4f} ({len(per_sample)} samples, {elapsed:.1f}s)")


def run_final_eval(cfg, pipe, eval_samples: list, neg_embeds: torch.Tensor,
                   inf_out_dir: Path, wandb, global_step: int,
                   cn_end_fraction: float) -> None:
    inf_out_dir.mkdir(parents=True, exist_ok=True)
    pipe.controlnet.eval()
    per_sample = []
    t0 = time.perf_counter()
    with torch.no_grad():
        for i, s in enumerate(eval_samples):
            frames = _run_one_inference(cfg, pipe, s, neg_embeds,
                                        seed=cfg.seed,
                                        cn_end_fraction=cn_end_fraction)
            mse = _frames_target_mse(frames, s["target_hwc"])
            ssim_val = _frames_target_ssim(frames, s["target_hwc"])
            mp4 = inf_out_dir / (
                f"sample{s['eval_idx']:03d}"
                f"_face{s['face_idx']}_{s['slug']}.mp4"
            )
            _save_video(frames, mp4, fps=8)
            per_sample.append({
                "eval_idx": s["eval_idx"],
                "face_idx": s["face_idx"],
                "slug": s["slug"],
                "mse": mse, "ssim": ssim_val, "mp4": str(mp4),
            })
            if (i + 1) % 10 == 0:
                print(f"[final-eval] {i+1}/{len(eval_samples)} done, "
                      f"running avg_mse={np.mean([r['mse'] for r in per_sample]):.5f} "
                      f"avg_ssim={np.mean([r['ssim'] for r in per_sample]):.4f}")
            _EVAL_LOG["final"] = per_sample
            _write_eval_log()
    avg_mse = float(np.mean([r["mse"] for r in per_sample])) if per_sample else float("nan")
    avg_ssim = float(np.mean([r["ssim"] for r in per_sample])) if per_sample else float("nan")
    elapsed = time.perf_counter() - t0
    table = wandb.Table(
        columns=["eval_idx", "face_idx", "slug", "mse", "ssim"],
        data=[[r["eval_idx"], r["face_idx"], r["slug"], r["mse"], r["ssim"]]
              for r in per_sample],
    )
    wandb.log({
        "eval_final/mse_avg": avg_mse,
        "eval_final/ssim_avg": avg_ssim,
        "eval_final/wall_s": elapsed,
        "eval_final/per_sample_table": table,
    }, step=global_step)
    _RESULTS_STATE["final_eval_mse_avg"] = round(avg_mse, 6)
    _RESULTS_STATE["final_eval_ssim_avg"] = round(avg_ssim, 6)
    _RESULTS_STATE["final_eval_wall_s"] = round(elapsed, 1)
    _RESULTS_STATE["final_eval_count"] = len(per_sample)
    _write_results()
    print(f"[final-eval] avg_mse={avg_mse:.5f} avg_ssim={avg_ssim:.4f} "
          f"({len(per_sample)} samples, {elapsed:.1f}s)")


if __name__ == "__main__":
    main()

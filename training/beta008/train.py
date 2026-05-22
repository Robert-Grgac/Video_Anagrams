"""BETA8 training: beta-007_silhouette recipe + D1 spatial face-weighted FM loss.

Same single-expert cold-start setup as ``training/beta007/train.py``; the only
substantive change is the supervised reconstruction loss. The FM MSE is now
weighted per-latent-position by ``1 + α · mask``, where ``mask`` is the
silhouette of the target face downsampled to latent resolution. The
self-distillation consistency term remains uniform (unweighted) — its job is
to regularize live↔EMA agreement on the prediction itself, independent of
where the face sits in the frame.

Background: beta-007 ablations across canny / silhouette / raw-face inputs
all produced final SSIM within Δ0.03 of each other, and the CN-on vs CN-off
diagnostic on beta-007_v2 showed the CN *was* participating (different output
for the same seed) but not producing face structure. The interpretation is
that the FM loss gradient is dominated by the ~98% of latent positions that
sit outside the face region, where the prompt alone reconstructs the target
cheaply. Spatial weighting tilts the gradient so face-region accuracy
contributes a larger share, forcing the CN to do useful work in those
positions.

Three deliberate deviations from train_beta6 (which beta-007 was specced from):

1. **Gradient accumulation via Accelerate** — uses
   ``accelerator.accumulate(controlnet)`` + ``accelerator.backward(loss)``
   instead of manual ``loss / accum_steps`` + ``loss.backward()``.
   Reference: https://huggingface.co/docs/accelerate/usage_guides/gradient_accumulation
2. **Train / eval split** — first 9900 manifest records train, last 100
   are held out for eval (no overlap). Train set drives 1 epoch
   (= floor(9900 / 32) = 309 effective optimizer steps). Eval set drives
   periodic-during-training and final-after-training inference MSE.
3. **High-noise expert only** — the trained CN is wired into ``transformer``
   (sibling of beta-005), not ``transformer_2``. This gives a high-CN
   suitable for slotting into the dual pipeline.

End-of-run protocol: 100-sample inference run on the eval set using the
EMA controlnet, with per-sample pixel-space MSE against the target JPG.

Eval-MSE storage: each periodic eval (every 10 effective steps) and the
final 100-sample eval write to (a) wandb scalars + a wandb.Table, and
(b) a sibling JSON ``training_cards/beta-007_eval.json`` for offline access.
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


# T5 padding length expected by the Wan transformer's cross-attention.
# MUST match precompute_beta.py --max_seq_len. Feeding any other length to the
# transformer corrupts its conditioning silently (no shape error, just noise
# output at inference). See discussion in training_cards/beta007/beta-007.md.
WAN_T5_MAX_SEQ_LEN = 226


# ---------------- main ----------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--cache_dir", type=str, required=True)
    p.add_argument("--control_subdir", type=str, default="canny",
                   help="Subdir of cache_dir from which to load the ControlNet "
                        "input modality. Defaults to 'canny' (existing cache). "
                        "Set 'silhouette' to train against the option-H "
                        "face-mask + interior-contour map produced by "
                        "training/precompute_silhouette.py. All control "
                        "modalities share the same (3, H, W) uint8 schema "
                        "so no other code path changes.")
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
    p.add_argument("--lr_decay_per_epoch", type=float, default=1.0,
                   help="Multiplicative factor applied to lr at the end of each "
                        "completed epoch (e.g., 0.9 = decay by 10% per epoch). "
                        "Default 1.0 = no decay.")
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--num_epochs", type=int, default=1)
    p.add_argument("--max_steps", type=int, default=None,
                   help="Optional hard cap on EFFECTIVE optimizer steps. If None, "
                        "num_epochs * floor(num_train / accum) effective steps "
                        "(= num_epochs * 309 for 9900/32).")
    p.add_argument("--micro_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=32)
    p.add_argument("--ema_decay", type=float, default=0.99)
    p.add_argument("--ema_update_after_step", type=int, default=10)
    p.add_argument("--loss_ema_window", type=int, default=20)

    # Architecture overrides — None = use the HED config defaults.
    p.add_argument("--num_cn_layers", type=int, default=None,
                   help="Override the controlnet config's num_layers. None = config default.")
    p.add_argument("--controlnet_stride", type=int, default=3)

    p.add_argument("--use_self_distillation", action="store_true",
                   help="Add lambda_consistency * MSE(v_pred_live, v_pred_ema) to the FM loss.")
    p.add_argument("--lambda_consistency", type=float, default=0.5)

    # D1 — spatial face-weighted FM loss.
    p.add_argument("--face_weight_alpha", type=float, default=2.0,
                   help="Per-latent-position FM-loss weight = 1 + α · mask, where "
                        "mask in [0,1] is the per-face silhouette downsampled to "
                        "latent resolution. α=0 disables D1 (uniform FM loss, "
                        "equivalent to beta-007). The self-distillation "
                        "consistency term is NOT weighted by this mask.")
    p.add_argument("--face_mask_subdir", type=str, default="silhouette",
                   help="Cache subdir to load per-face binary masks from for the "
                        "D1 spatial weighting. The (3, H, W) uint8 silhouette "
                        "produced by training/precompute_silhouette.py is "
                        "thresholded to a {0,1} mask and avg-pooled by H/H_lat "
                        "to latent resolution. Independent of --control_subdir; "
                        "you can train with canny as the CN input and still mask "
                        "the loss by the silhouette.")

    p.add_argument("--num_train_timesteps_for_sampling", type=int, default=1000)
    p.add_argument("--boundary_ratio_override", type=float, default=None)
    p.add_argument("--checkpoint_every", type=int, default=50)
    p.add_argument("--memory_tripwire_gb", type=float, default=90.0)

    # Eval protocol
    p.add_argument("--eval_size", type=int, default=100,
                   help="Hold-out eval set size, taken from the TAIL of the manifest.")
    p.add_argument("--periodic_eval_size", type=int, default=10,
                   help="Number of fixed eval samples to run during training (subset of eval set, indices 0..k-1).")
    p.add_argument("--periodic_eval_every", type=int, default=10,
                   help="Run periodic eval every N EFFECTIVE optimizer steps.")
    p.add_argument("--inference_steps", type=int, default=50,
                   help="Number of denoising steps per inference call (periodic + final).")
    p.add_argument("--inference_guidance_scale", type=float, default=5.0)
    p.add_argument("--inference_controlnet_weight", type=float, default=1.0)
    p.add_argument("--inference_controlnet_end", type=float, default=None,
                   help="If unset (default), compute dynamically so the CN runs only "
                        "while σ ≥ boundary_ratio (high-noise expert active). Set "
                        "explicitly to a float in [0,1] to override.")
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


# ------------- eval helpers -------------

def _load_target_image(targets_dir: Path, face_idx: int, slug: str,
                       height: int, width: int) -> np.ndarray:
    """Load `face_{idx}_{slug}.jpg`, resize to (H, W), return float [0, 1] HWC."""
    from PIL import Image
    p = targets_dir / f"face_{face_idx}_{slug}.jpg"
    if not p.exists():
        raise FileNotFoundError(f"Target JPG not found: {p}")
    img = Image.open(str(p)).convert("RGB")
    if img.size != (width, height):
        img = img.resize((width, height), Image.BICUBIC)
    arr = np.asarray(img, dtype=np.float32) / 255.0  # [H, W, 3] in [0, 1]
    return arr


def _save_video(frames_np: np.ndarray, path: Path, fps: int = 8) -> None:
    import imageio
    arr = (np.clip(frames_np, 0.0, 1.0) * 255).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(path), list(arr), fps=fps, codec="libx264")


def _frames_target_mse(frames: np.ndarray, target_hwc: np.ndarray) -> float:
    """Pixel-space MSE between rendered video frames and the (replicated) target JPG.

    frames: [T, H, W, 3] in [0, 1]
    target_hwc: [H, W, 3] in [0, 1]
    Returns mean squared error averaged over T*H*W*3.
    """
    target_T = np.broadcast_to(target_hwc[None, ...], frames.shape)
    diff = frames.astype(np.float32) - target_T.astype(np.float32)
    return float(np.mean(diff * diff))


def _frames_target_ssim(frames: np.ndarray, target_hwc: np.ndarray,
                        device: Optional[str] = None) -> float:
    """Mean per-frame SSIM between rendered video frames and the replicated target.

    frames: [T, H, W, 3] in [0, 1]
    target_hwc: [H, W, 3] in [0, 1]
    Returns scalar SSIM averaged over T frames (data_range=1.0).
    """
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
    """Build deterministic face/prompt-stratified train / eval / periodic-eval splits.

    Both the eval set and the periodic-eval set are *Latin pairings* — every
    record uses a distinct face_idx AND a distinct slug, so no two samples
    inside the same set share either axis.

    - **Eval set (100 records)**: identity matching — sorted_faces[i] paired
      with sorted_slugs[i].
    - **Periodic-eval set (10 records)**: shift-by-50 matching on the first
      10 sorted faces — sorted_faces[i] paired with sorted_slugs[(i+50) % 100]
      for i in 0..9.
    - **Train set (9900 records)**: every record whose (face_idx, slug) pair
      is NOT in the eval set. The 10 periodic-eval pairs are *inside* the
      train set on purpose — periodic eval is an overfit check on samples
      the model actually trains on.

    Returns (train_indices, eval_indices, periodic_indices), each a list of
    indices into ``records[]``. ``periodic_indices`` is a subset of
    ``train_indices``.
    """
    faces = sorted({r["face_idx"] for r in records})
    slugs = sorted({r["slug"] for r in records})
    if len(faces) != 100 or len(slugs) != 100:
        raise RuntimeError(
            f"Expected 100 distinct faces and 100 distinct slugs in the manifest; "
            f"got {len(faces)} faces and {len(slugs)} slugs."
        )

    eval_pair_set = {(faces[i], slugs[i]) for i in range(100)}
    periodic_pair_set = {(faces[i], slugs[(i + 50) % 100]) for i in range(10)}
    assert eval_pair_set.isdisjoint(periodic_pair_set), \
        "Periodic and eval pair sets overlap — split logic is broken."

    by_pair: dict[tuple[int, str], int] = {}
    for idx, r in enumerate(records):
        key = (r["face_idx"], r["slug"])
        if key in by_pair:
            raise RuntimeError(f"Duplicate (face_idx, slug) pair in manifest: {key}")
        by_pair[key] = idx
    if len(by_pair) != len(records):
        raise RuntimeError("Manifest contains duplicate (face_idx, slug) pairs.")

    eval_indices = sorted(by_pair[p] for p in eval_pair_set)
    periodic_indices = sorted(by_pair[p] for p in periodic_pair_set)
    eval_idx_set = set(eval_indices)
    train_indices = [i for i in range(len(records)) if i not in eval_idx_set]

    train_idx_set = set(train_indices)
    assert all(i in train_idx_set for i in periodic_indices), \
        "Periodic-eval indices are not a subset of train indices."
    assert len(eval_indices) == 100 and len(periodic_indices) == 10
    return train_indices, eval_indices, periodic_indices


def _compute_cn_end_high_noise(base_model_path: str, num_inference_steps: int,
                               boundary_ratio: float, device: torch.device) -> tuple[float, int]:
    """Step-fraction at which σ first drops below ``boundary_ratio`` for the
    pipeline's FlowMatch Euler scheduler with ``num_inference_steps`` steps.

    Passing this value as ``controlnet_guidance_end`` makes the pipeline's
    step-index gate (``current_sampling_percent < cn_end``) exactly mirror
    the σ-based gate that picks the high-noise expert (``t >= boundary_ts``).
    Result: the CN is computed and injected only into the high-noise
    transformer, never into transformer_2.

    Returns (cn_end_fraction, first_low_idx). If every step is high-noise,
    returns (1.0, num_inference_steps).
    """
    from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
    sched = FlowMatchEulerDiscreteScheduler.from_pretrained(
        base_model_path, subfolder="scheduler",
    )
    sched.set_timesteps(num_inference_steps, device=device)
    sigmas = sched.sigmas[:-1].detach().cpu()  # drop trailing 0
    below = (sigmas < boundary_ratio).nonzero(as_tuple=False)
    if below.numel() == 0:
        return 1.0, num_inference_steps
    first_low = int(below[0].item())
    return first_low / num_inference_steps, first_low


# ------------- main -------------

def main() -> None:
    global _RESULTS_PATH, _CARD_PATH, _EVAL_LOG_PATH
    cfg = parse_args()
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
        _RESULTS_PATH = _CARD_PATH.parent / f"{_CARD_PATH.stem}_results.json"
        _EVAL_LOG_PATH = _CARD_PATH.parent / f"{_CARD_PATH.stem}_eval.json"
    else:
        _RESULTS_PATH = Path("training_cards") / "beta008" / f"{cfg.run_name}_results.json"
        _EVAL_LOG_PATH = Path("training_cards") / "beta008" / f"{cfg.run_name}_eval.json"

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
        "face_weight_alpha": cfg.face_weight_alpha,
        "face_mask_subdir": cfg.face_mask_subdir,
        "eval_size": cfg.eval_size,
        "periodic_eval_size": cfg.periodic_eval_size,
        "periodic_eval_every": cfg.periodic_eval_every,
        "inference_steps": cfg.inference_steps,
    })
    _write_results()
    atexit.register(_atexit_hook)

    import wandb
    wandb_mode = os.environ.get("WANDB_MODE", "online")
    wandb_run = wandb.init(
        project=cfg.wandb_project,
        name=cfg.run_name,
        config=vars(cfg),
        mode=wandb_mode,
    )
    wandb_url = wandb_run.get_url() if wandb_mode == "online" else f"offline:{wandb_run.dir}"
    _RESULTS_STATE["wandb_url"] = wandb_url

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("BETA7 training requires CUDA; got CPU.")
    try:
        major, minor = torch.cuda.get_device_capability(0)
        gpu_name = torch.cuda.get_device_name(0)
        print(f"[gpu] {gpu_name} (compute_cap={major}.{minor})")
        _RESULTS_STATE["gpu_name"] = gpu_name
        _RESULTS_STATE["gpu_compute_cap"] = f"{major}.{minor}"
    except Exception:
        pass

    # --- Accelerator (gradient accumulation gating) ---
    from accelerate import Accelerator
    accelerator = Accelerator(gradient_accumulation_steps=cfg.gradient_accumulation_steps)
    print(f"[accel] gradient_accumulation_steps={cfg.gradient_accumulation_steps} "
          f"(num_processes={accelerator.num_processes})")

    # --- Models ---
    from wan_controlnet import WanControlnet
    from wan_transformer import CustomWanTransformer3DModel
    from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
    from diffusers import AutoencoderKLWan
    from transformers import AutoTokenizer, UMT5EncoderModel

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

    print(f"[load] controlnet config from {cfg.controlnet_config_repo} (architecture only) ...")
    config = WanControlnet.load_config(cfg.controlnet_config_repo)
    if cfg.num_cn_layers is not None:
        old_L = config.get("num_layers", "?")
        config["num_layers"] = cfg.num_cn_layers
        max_used_block = (cfg.num_cn_layers - 1) * cfg.controlnet_stride
        if max_used_block >= 40:
            wasted = sum(1 for i in range(cfg.num_cn_layers)
                         if i * cfg.controlnet_stride >= 40)
            print(f"[arch] WARN num_layers={cfg.num_cn_layers} × stride={cfg.controlnet_stride} "
                  f"would waste {wasted} CN layer(s).", file=sys.stderr)
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

    # --- Train / eval data split (face × prompt stratified) ---
    full_dataset = BetaPairDataset(cfg.cache_dir, num_frames=cfg.num_frames,
                                   control_subdir=cfg.control_subdir)
    total_n = len(full_dataset)
    if total_n != 10000:
        raise RuntimeError(
            f"Expected 10000 records (100 faces × 100 prompts); got {total_n}. "
            "The face/prompt-stratified split assumes the full 100×100 manifest."
        )
    train_indices, eval_indices, periodic_indices = _build_eval_periodic_splits(
        full_dataset.records
    )
    n_train = len(train_indices)
    if cfg.eval_size != len(eval_indices):
        print(f"[data] WARN: --eval_size={cfg.eval_size} ignored; stratified split "
              f"produces {len(eval_indices)} eval records (one per face).")
    if cfg.periodic_eval_size != len(periodic_indices):
        print(f"[data] WARN: --periodic_eval_size={cfg.periodic_eval_size} ignored; "
              f"stratified split produces {len(periodic_indices)} periodic-eval records.")
    train_dataset = Subset(full_dataset, train_indices)
    eval_faces = sorted({full_dataset.records[i]["face_idx"] for i in eval_indices})
    periodic_faces = sorted({full_dataset.records[i]["face_idx"] for i in periodic_indices})
    print(f"[data] total={total_n} → train={len(train_dataset)} "
          f"eval={len(eval_indices)} periodic={len(periodic_indices)}")
    print(f"[data] eval faces: {len(eval_faces)} distinct "
          f"(identity matching face_i ↔ sorted_slug_i)")
    print(f"[data] periodic faces: {periodic_faces} "
          f"(shift-by-50 matching, subset of train)")
    _RESULTS_STATE["pair_count"] = n_train
    _RESULTS_STATE["eval_count"] = len(eval_indices)
    _RESULTS_STATE["periodic_eval_count"] = len(periodic_indices)

    # --- D1: pre-load per-face spatial loss-weight masks at latent resolution ---
    # The Wan VAE spatially compresses 8×, so a 512×512 input maps to 64×64 latent
    # positions. We threshold the silhouette to a binary {0,1} mask (face interior
    # + contour lines vs. background), then avg-pool by H//H_lat to get soft values
    # in [0,1] at latent resolution. Pooled values capture partial coverage on the
    # silhouette boundary, avoiding the hard-edge aliasing of a thresholded
    # downsample. Cached per unique face_idx; lookup at micro-batch time is a dict
    # hit, no I/O in the training hot path.
    mask_dir = Path(cfg.cache_dir) / cfg.face_mask_subdir
    if not mask_dir.exists():
        raise FileNotFoundError(
            f"--face_mask_subdir='{cfg.face_mask_subdir}' resolves to {mask_dir}, "
            f"which does not exist. Run precompute_silhouette.py first."
        )
    spatial_factor = cfg.height // (cfg.height // 8)  # = 8 for the Wan VAE
    h_lat = cfg.height // spatial_factor
    w_lat = cfg.width // spatial_factor
    unique_face_idxs = sorted({r["face_idx"] for r in full_dataset.records})
    face_masks_latent: dict[int, torch.Tensor] = {}
    for fi in unique_face_idxs:
        mask_path = mask_dir / f"face_{fi}.pt"
        if not mask_path.exists():
            raise FileNotFoundError(
                f"D1 mask missing for face_idx={fi}: expected {mask_path}"
            )
        raw = torch.load(mask_path, map_location="cpu",
                         weights_only=True)  # (3, H, W) uint8
        # Silhouette uses values {0, fill, line}; anything > 0 is "inside face".
        m_bin = (raw[0] > 0).float()  # (H, W) in {0, 1}
        m_lat = F.avg_pool2d(
            m_bin.unsqueeze(0).unsqueeze(0),
            kernel_size=spatial_factor, stride=spatial_factor,
        ).squeeze(0).squeeze(0)  # (h_lat, w_lat) in [0, 1]
        face_masks_latent[int(fi)] = m_lat.contiguous()
    coverage_mean = float(
        torch.stack(list(face_masks_latent.values())).mean().item()
    )
    print(f"[d1] cached {len(face_masks_latent)} face masks from {mask_dir} "
          f"at {h_lat}x{w_lat}; mean face coverage="
          f"{coverage_mean:.3f}; α={cfg.face_weight_alpha}")
    _RESULTS_STATE["face_mask_coverage_mean"] = round(coverage_mean, 4)
    _RESULTS_STATE["face_mask_latent_hw"] = [h_lat, w_lat]

    train_loader = DataLoader(
        train_dataset, batch_size=cfg.micro_batch_size, shuffle=True,
        num_workers=cfg.num_workers,
        persistent_workers=cfg.num_workers > 0,
        pin_memory=True,
        drop_last=True,
        collate_fn=_collate_keep_meta,
    )
    micro_steps_per_epoch = len(train_loader)
    max_eff_from_epoch = micro_steps_per_epoch // cfg.gradient_accumulation_steps
    if cfg.max_steps is None:
        cfg.max_steps = max_eff_from_epoch * cfg.num_epochs
    print(f"[data] train: {len(train_dataset)} pairs, {micro_steps_per_epoch} micro-steps/epoch, "
          f"{max_eff_from_epoch} effective-steps/epoch, num_epochs={cfg.num_epochs}, "
          f"max_steps={cfg.max_steps}")

    # Manual-gating pattern: we use `accelerator.accumulate(controlnet)` only for
    # the boundary signal (`sync_gradients`) and for `accelerator.backward(loss)`'s
    # automatic 1/N loss scaling. We do NOT call `accelerator.prepare(...)` — so
    # `optimizer.step()` and `optimizer.zero_grad()` are NOT auto-gated. We gate
    # them manually with `if accelerator.sync_gradients:` after the accumulate
    # context exits. Easier to read than the prepared/wrapped flow.

    # --- Pre-load eval supporting models (text encoder used once, then dropped) ---
    print(f"[load] tokenizer + text_encoder (one-shot, for eval prompt encoding) ...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model_path, subfolder="tokenizer")
    text_encoder = UMT5EncoderModel.from_pretrained(
        cfg.base_model_path, subfolder="text_encoder", torch_dtype=torch.bfloat16,
    ).eval().to(device)

    targets_dir = Path(cfg.targets_dir)
    eval_records = [full_dataset.records[i] for i in eval_indices]
    periodic_records = [full_dataset.records[i] for i in periodic_indices]

    print(f"[load] vae ...")
    vae = AutoencoderKLWan.from_pretrained(
        cfg.base_model_path, subfolder="vae", torch_dtype=torch.bfloat16,
    ).eval().to(device)

    print(f"[load] low-noise transformer (transformer_2, eval-only) ...")
    transformer_2 = CustomWanTransformer3DModel.from_pretrained(
        cfg.base_model_path, subfolder="transformer_2", torch_dtype=torch.bfloat16,
    )
    transformer_2.requires_grad_(False).eval().to(device)
    _maybe_force_native_attention(transformer_2, "transformer_2 (eval)")

    # Encode neg-prompt + each eval sample's positive prompt ONCE, then drop text encoder.
    # We use the dataset cache's prompt_embeds for positives (already encoded), but
    # the pipeline also wants a negative prompt embed at CFG > 1.0 — encode that here.
    print(f"[load] encoding negative prompt embed ('{cfg.negative_prompt}') "
          f"at max_seq_len={WAN_T5_MAX_SEQ_LEN} ...")
    with torch.no_grad():
        neg_inputs = tokenizer(
            [prompt_clean(cfg.negative_prompt)],
            padding="max_length",
            max_length=WAN_T5_MAX_SEQ_LEN,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
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
    # Drop text encoder + tokenizer (we have everything we need).
    del text_encoder, tokenizer, neg_inputs, neg_ids, neg_mask, neg_lens
    gc.collect()
    torch.cuda.empty_cache()
    print(f"[load] text_encoder dropped; neg_embeds shape={tuple(neg_embeds.shape)}")

    # Pre-stage eval sample blobs (canny img + prompt embed + target jpg).
    # Two disjoint lists:
    #   - periodic_eval_samples: 10 records from the TRAIN split (overfit check)
    #   - eval_samples: 100 records from the held-out eval split (final)
    from PIL import Image
    cache_dir = Path(cfg.cache_dir)

    def _stage(records: list[dict]) -> list[dict]:
        out = []
        for j, rec in enumerate(records):
            # Route through the chosen control subdir; the manifest path is the
            # canny one but we only use its basename.
            control_path = cache_dir / cfg.control_subdir / Path(rec["canny_path"]).name
            canny_u8 = torch.load(control_path, map_location="cpu",
                                  weights_only=True)
            canny_img = Image.fromarray(canny_u8.permute(1, 2, 0).numpy())
            prompt_embed = torch.load(cache_dir / rec["prompt_path"], map_location="cpu",
                                      weights_only=True).to(torch.bfloat16)
            if prompt_embed.dim() == 2:
                prompt_embed = prompt_embed.unsqueeze(0)
            assert prompt_embed.shape[1] == WAN_T5_MAX_SEQ_LEN, (
                f"Cached prompt embed for {rec['slug']} has seq_len="
                f"{prompt_embed.shape[1]}, expected {WAN_T5_MAX_SEQ_LEN}. "
                f"Re-run precompute_beta.py with --max_seq_len {WAN_T5_MAX_SEQ_LEN} "
                f"(delete $cache_dir/prompts/ first to force re-encode)."
            )
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
    print(f"[eval] pre-staged periodic={len(periodic_eval_samples)} (from train, overfit check) "
          f"final={len(eval_samples)} (held-out)")

    # --- Build the inference pipeline once, reusing already-loaded modules ---
    from wan_t2v_controlnet_pipeline import WanTextToVideoControlnetPipeline

    eval_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        cfg.base_model_path, subfolder="scheduler",
    )
    pipe = WanTextToVideoControlnetPipeline(
        tokenizer=None,
        text_encoder=None,
        transformer=transformer,
        transformer_2=transformer_2,
        vae=vae,
        controlnet=controlnet,
        scheduler=eval_scheduler,
        boundary_ratio=boundary_ratio,
    )
    # All modules already on CUDA — DiffusionPipeline._execution_device auto-derives.
    # Skip enable_model_cpu_offload(): we'll call this pipe 31 + 100 times, offload
    # churn would dominate, and Blackwell has the headroom.
    print(f"[pipe] inference pipeline built (no CPU offload, all modules resident)")

    # Dynamic controlnet_guidance_end: pin CN to the high-noise expert only.
    # The pipeline gates CN computation by step-fraction (i/N), so we find the
    # smallest step index where σ < boundary_ratio and pass that fraction as
    # controlnet_guidance_end. Result: CN runs only while transformer (high-noise)
    # is the active expert, never while transformer_2 (low-noise) is active.
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
            # `accumulate` only does two things for us:
            #   1. Sets `accelerator.sync_gradients` True every Nth iter (boundary).
            #   2. Makes `accelerator.backward(loss)` divide loss by N internally.
            # Everything else (clip, step, zero_grad, EMA) is gated manually below.
            with accelerator.accumulate(controlnet):
                canny = batch["canny"].to(device, non_blocking=True)
                z_real = batch["latent"].to(device, non_blocking=True)
                prompt_embeds = batch["prompt_embeds"].to(device, non_blocking=True)
                B = z_real.shape[0]

                sel = torch.randint(0, len(high_noise_indices), (B,), device=device)
                t_idx = high_noise_indices[sel]
                sigma = sigmas[t_idx].to(z_real.dtype)
                t = timesteps_full[t_idx]
                sigma_b = sigma.view(B, 1, 1, 1, 1)

                noise = torch.randn_like(z_real)
                z_t = (1.0 - sigma_b) * z_real + sigma_b * noise
                v_target = (noise - z_real).float()

                # Self-distillation teacher forward
                v_pred_ema = None
                if cfg.use_self_distillation:
                    with torch.no_grad():
                        cn_states_ema = ema.ema_model(
                            hidden_states=z_t,
                            timestep=t,
                            encoder_hidden_states=prompt_embeds,
                            controlnet_states=canny,
                            return_dict=False,
                        )[0]
                        if isinstance(cn_states_ema, (tuple, list)):
                            cn_for_tx_ema = [s.to(dtype=transformer.dtype)
                                             for s in cn_states_ema]
                        else:
                            cn_for_tx_ema = cn_states_ema.to(dtype=transformer.dtype)
                        v_pred_ema = transformer(
                            hidden_states=z_t,
                            timestep=t,
                            encoder_hidden_states=prompt_embeds,
                            controlnet_states=cn_for_tx_ema,
                            controlnet_weight=1.0,
                            controlnet_stride=cfg.controlnet_stride,
                            return_dict=False,
                        )[0].float()
                        del cn_states_ema, cn_for_tx_ema

                controlnet_states = controlnet(
                    hidden_states=z_t,
                    timestep=t,
                    encoder_hidden_states=prompt_embeds,
                    controlnet_states=canny,
                    return_dict=False,
                )[0]
                if isinstance(controlnet_states, (tuple, list)):
                    controlnet_states_for_tx = [s.to(dtype=transformer.dtype)
                                                for s in controlnet_states]
                else:
                    controlnet_states_for_tx = controlnet_states.to(dtype=transformer.dtype)

                v_pred = transformer(
                    hidden_states=z_t,
                    timestep=t,
                    encoder_hidden_states=prompt_embeds,
                    controlnet_states=controlnet_states_for_tx,
                    controlnet_weight=1.0,
                    controlnet_stride=cfg.controlnet_stride,
                    return_dict=False,
                )[0]

                # D1: spatial face-weighted FM loss. v_pred shape is
                # (B, C, T_lat, H_lat, W_lat); the per-face mask is
                # (H_lat, W_lat) and broadcasts over C and T_lat.
                face_idxs_b = batch["face_idx"]
                masks_b = torch.stack(
                    [face_masks_latent[int(fi)] for fi in face_idxs_b], dim=0,
                ).to(v_pred.device, dtype=torch.float32)  # (B, H_lat, W_lat)
                masks_b = masks_b.unsqueeze(1).unsqueeze(1)  # (B, 1, 1, H_lat, W_lat)
                weight_map = 1.0 + cfg.face_weight_alpha * masks_b
                diff2 = (v_pred.float() - v_target) ** 2
                loss_fm = (weight_map * diff2).mean()
                if cfg.use_self_distillation and v_pred_ema is not None:
                    # Consistency stays uniform — it regularizes live↔EMA
                    # agreement on the prediction itself, not on target fitting.
                    loss_consistency = F.mse_loss(v_pred.float(), v_pred_ema)
                    loss = loss_fm + cfg.lambda_consistency * loss_consistency
                else:
                    loss_consistency = torch.zeros((), device=v_pred.device)
                    loss = loss_fm

                # Auto-scales loss by 1/accumulation_steps before backward.
                accelerator.backward(loss)

                accum_losses.append(loss.detach())
                accum_losses_fm.append(loss_fm.detach())
                accum_losses_consist.append(loss_consistency.detach())
                accum_residual_l2.append(mean_residual_l2(controlnet_states))
                accum_sigmas.extend(sigma.detach().float().cpu().tolist())
                accum_t_last = float(t[-1].item())
                micro_step += 1

                if not grad_assert_done:
                    # Asserted INSIDE the accumulate block, after the first backward,
                    # before any potential step/zero_grad — guaranteed to see grads.
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

            # ---------------- Boundary actions (manual gate) ----------------
            # Outside the `accumulate` block, sync_gradients reflects whether THIS
            # iteration was the Nth (= full accumulation) step. Only then do we
            # clip, step, zero, EMA-update, log, eval, checkpoint.
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
                accum_losses_consist = []
                accum_residual_l2 = []
                accum_sigmas = []

                if peak_mem > cfg.memory_tripwire_gb:
                    raise RuntimeError(
                        f"GPU memory {peak_mem:.2f}GB exceeded tripwire "
                        f"{cfg.memory_tripwire_gb}GB at step {global_step}."
                    )

                # --- Periodic eval (LIVE CN on the fixed 10 training-set samples) ---
                if (not cfg.skip_periodic_eval
                        and global_step % cfg.periodic_eval_every == 0):
                    try:
                        run_periodic_eval(
                            cfg=cfg,
                            global_step=global_step,
                            pipe=pipe,
                            controlnet=controlnet,
                            eval_samples=periodic_eval_samples,
                            neg_embeds=neg_embeds,
                            inf_out_dir=inf_out_dir / "periodic",
                            wandb=wandb,
                            cn_end_fraction=cn_end_fraction,
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
            print(f"[lr] epoch {epoch} complete; decayed lr by "
                  f"{cfg.lr_decay_per_epoch} -> {new_lr:.3e}")
            try:
                wandb.log({"lr_epoch_end": new_lr, "epoch_completed": epoch},
                          step=global_step)
            except Exception:
                pass

    if micro_step % cfg.gradient_accumulation_steps != 0:
        leftover = micro_step % cfg.gradient_accumulation_steps
        print(f"[stop] discarding {leftover} micro-step(s) of partial accumulation at exit")
        optimizer.zero_grad(set_to_none=True)

    # --- Final checkpoints ---
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

    # --- Final 100-sample eval on EMA controlnet ---
    if not cfg.skip_final_eval:
        # Swap the live CN in the pipe for an EMA-loaded copy.
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
                cfg=cfg,
                pipe=pipe,
                eval_samples=eval_samples,
                neg_embeds=neg_embeds,
                inf_out_dir=inf_out_dir / "final",
                wandb=wandb,
                global_step=global_step,
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
          f"loss_ema={loss_ema_str} | wall={_RESULTS_STATE['wall_time']} | init={init_mode}")


# ---------------- eval runners ----------------

def _run_one_inference(cfg, pipe, sample: dict, neg_embeds: torch.Tensor,
                       seed: int, cn_end_fraction: float) -> np.ndarray:
    """Run one inference call; return numpy frames [T, H, W, 3] in [0, 1]."""
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
    """Run inference on the fixed periodic-eval samples with the LIVE controlnet.

    Samples come from the TRAINING split (overfit-check) and use a CN-end
    fraction that confines the CN to the high-noise expert only.
    """
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
                    "mse": mse,
                    "ssim": ssim_val,
                    "mp4": str(mp4),
                })
    finally:
        if was_training:
            controlnet.train()
    avg_mse = float(np.mean([r["mse"] for r in per_sample])) if per_sample else float("nan")
    avg_ssim = float(np.mean([r["ssim"] for r in per_sample])) if per_sample else float("nan")
    elapsed = time.perf_counter() - t0
    log_payload = {
        "eval/mse_avg": avg_mse,
        "eval/ssim_avg": avg_ssim,
        "eval/wall_s": elapsed,
    }
    for r in per_sample:
        log_payload[f"eval/mse_sample_{r['eval_idx']:02d}"] = r["mse"]
        log_payload[f"eval/ssim_sample_{r['eval_idx']:02d}"] = r["ssim"]
    wandb.log(log_payload, step=global_step)
    _EVAL_LOG["periodic"].append({
        "global_step": global_step,
        "avg_mse": avg_mse,
        "avg_ssim": avg_ssim,
        "wall_s": round(elapsed, 1),
        "samples": per_sample,
    })
    _write_eval_log()
    print(f"[periodic-eval] step={global_step} avg_mse={avg_mse:.5f} "
          f"avg_ssim={avg_ssim:.4f} ({len(per_sample)} samples, {elapsed:.1f}s)")


def run_final_eval(cfg, pipe, eval_samples: list, neg_embeds: torch.Tensor,
                   inf_out_dir: Path, wandb, global_step: int,
                   cn_end_fraction: float) -> None:
    """Run inference on ALL eval samples with the EMA controlnet (already swapped in).

    The eval samples come from the held-out split (100 face/prompt-unique pairs);
    CN-end fraction confines the CN to the high-noise expert only.
    """
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
                "mse": mse,
                "ssim": ssim_val,
                "mp4": str(mp4),
            })
            if (i + 1) % 10 == 0:
                print(f"[final-eval] {i+1}/{len(eval_samples)} done, "
                      f"running avg_mse={np.mean([r['mse'] for r in per_sample]):.5f} "
                      f"avg_ssim={np.mean([r['ssim'] for r in per_sample]):.4f}")
            # Incremental dump so a wall-time crash leaves something usable.
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

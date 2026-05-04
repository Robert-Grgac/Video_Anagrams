"""BETA training: cold-start WanControlnet on Canny -> static-video pairs.

* High-noise expert only (`transformer/` subfolder); `transformer_2/` left on disk
  and is loaded only for the end-of-run inference smoke.
* Flow-matching MSE loss with timesteps restricted to the high-noise regime.
* Mixed precision bf16, but `_keep_in_fp32_modules` (norms, time embedder,
  scale_shift_table) are preserved per the diffusers convention; a blanket
  `.to(bf16)` would silently demote them.
* End-of-run inference: 1 (face, prompt) pair through the full pipeline.

Run lifecycle JSON (`training_cards/{run_id}_results.json`) is written at start,
updated at end, and keyed for `training/autofill_card.py`. An `atexit` hook
catches mid-run crashes so the JSON still reports `status: failed`.
"""
from __future__ import annotations

import argparse
import atexit
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
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from training.dataset_beta import BetaPairDataset


# ---------------- helpers ----------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _git_sha() -> str:
    try:
        import subprocess
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(Path(__file__).parent.parent),
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def cast_respecting_fp32_modules(model: nn.Module, dtype: torch.dtype) -> None:
    """Cast every parameter to ``dtype`` except those whose qualified name
    matches a substring in ``model._keep_in_fp32_modules``.

    Diffusers' built-in loaders honor that list; constructing a model from a
    bare config does not, so a blanket ``.to(bf16)`` would silently demote
    norms / sinusoidal time embeds / scale_shift tables.
    """
    keep = list(getattr(model, "_keep_in_fp32_modules", []) or [])
    skipped: list[str] = []
    casted: list[str] = []
    for name, param in model.named_parameters():
        if any(k in name for k in keep):
            skipped.append(name)
        else:
            param.data = param.data.to(dtype)
            casted.append(name)
    for name, buf in model.named_buffers():
        if any(k in name for k in keep):
            continue
        buf.data = buf.data.to(dtype)
    print(f"[cast] {len(casted)} params -> {dtype}; "
          f"{len(skipped)} kept fp32 (e.g. {skipped[:3]})")


def detect_boundary_ratio(base_model_path: str | Path,
                          transformer_config: dict,
                          override: Optional[float] = None) -> tuple[float, str]:
    """Return (boundary_ratio, source_string).

    Priority: explicit ``override`` -> ``transformer.config.boundary_ratio``
    -> ``model_index.json`` of the base pipeline -> default ``0.5`` (upper 50%).
    """
    if override is not None:
        return float(override), "cli_override"
    for key in ("boundary_ratio", "boundary_sigma"):
        v = transformer_config.get(key)
        if v is not None:
            return float(v), f"transformer.config.{key}"
    mi = Path(base_model_path) / "model_index.json"
    if mi.exists():
        try:
            data = json.loads(mi.read_text())
            v = data.get("boundary_ratio")
            if v is not None:
                return float(v), "model_index.json.boundary_ratio"
        except Exception:
            pass
    return 0.5, "fallback_upper_50pct"


def mean_residual_l2(residuals) -> float:
    if residuals is None:
        return 0.0
    if isinstance(residuals, (list, tuple)):
        vals = [r.detach().float().pow(2).mean().sqrt().item() for r in residuals]
        return float(np.mean(vals))
    return float(residuals.detach().float().pow(2).mean().sqrt().item())


# ---------------- main ----------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--cache_dir", type=str, required=True)
    p.add_argument("--base_model_path", type=str, required=True)
    p.add_argument("--controlnet_config_repo", type=str, required=True,
                   help="HF repo id or local path containing a controlnet config.json. "
                        "Weights are NOT loaded; cold init only.")
    p.add_argument("--output_dir", type=str, required=True,
                   help="Where checkpoint .safetensors files are written.")
    p.add_argument("--inference_output_dir", type=str, default=None,
                   help="Where the end-of-run inference mp4 is saved. Defaults to output_dir/../outputs.")
    p.add_argument("--card_path", type=str, default=None,
                   help="Path to the training card markdown for autofill.")
    p.add_argument("--wandb_project", type=str, default="wan-controlnet-beta")
    p.add_argument("--run_name", type=str, required=True)

    p.add_argument("--num_frames", type=int, default=9)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=512)

    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--num_epochs", type=int, default=1)
    p.add_argument("--num_train_timesteps_for_sampling", type=int, default=1000,
                   help="How densely to discretize the schedule for training-time sampling.")
    p.add_argument("--boundary_ratio_override", type=float, default=None,
                   help="If set, bypass auto-detection of the high/low noise boundary.")
    p.add_argument("--checkpoint_every", type=int, default=2000)
    p.add_argument("--memory_tripwire_gb", type=float, default=43.0)

    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_steps", type=int, default=None,
                   help="If set, cap training at this many optimizer steps (smoke).")
    p.add_argument("--skip_inference_smoke", action="store_true")
    return p.parse_args()


# Module-level state for the atexit hook
_RESULTS_STATE: dict = {}
_RESULTS_PATH: Optional[Path] = None
_CARD_PATH: Optional[Path] = None


def _write_results() -> None:
    if _RESULTS_PATH is None:
        return
    try:
        _RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        _RESULTS_PATH.write_text(json.dumps(_RESULTS_STATE, indent=2, default=str))
    except Exception as e:
        print(f"[results] WARN failed to write {_RESULTS_PATH}: {e}",
              file=sys.stderr)


def _atexit_hook() -> None:
    """If main() exited without setting status to a terminal value, mark failed."""
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
    if _CARD_PATH is not None:
        try:
            from training.autofill_card import autofill
            autofill(_CARD_PATH)
        except Exception as e:
            print(f"[autofill] atexit fill failed: {e}", file=sys.stderr)


def main() -> None:
    global _RESULTS_PATH, _CARD_PATH
    cfg = parse_args()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    inf_out_dir = Path(cfg.inference_output_dir) if cfg.inference_output_dir \
        else output_dir.parent / "outputs"
    inf_out_dir.mkdir(parents=True, exist_ok=True)

    if cfg.card_path:
        _CARD_PATH = Path(cfg.card_path)
        _RESULTS_PATH = _CARD_PATH.parent / f"{_CARD_PATH.stem}_results.json"
    else:
        _RESULTS_PATH = Path("training_cards") / f"{cfg.run_name}_results.json"

    _RESULTS_STATE.update({
        "status": "running",
        "date_started": _now_iso(),
        "git_sha": _git_sha(),
        "cluster_partition": os.environ.get("SLURM_JOB_PARTITION", "unknown"),
        "run_name": cfg.run_name,
    })
    _write_results()
    atexit.register(_atexit_hook)

    # --- wandb ---
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
        raise RuntimeError("BETA training requires CUDA; got CPU.")

    # --- Models ---
    from wan_transformer import CustomWanTransformer3DModel
    from wan_controlnet import WanControlnet
    from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

    print(f"[load] high-noise transformer from {cfg.base_model_path}/transformer ...")
    transformer = CustomWanTransformer3DModel.from_pretrained(
        cfg.base_model_path, subfolder="transformer", torch_dtype=torch.bfloat16,
    )
    transformer.requires_grad_(False).eval().to(device)
    # Frozen, but cn_for_tx (input) needs grad → activations would be retained
    # for backward without checkpointing. Recompute on demand to fit in A40 VRAM.
    transformer.enable_gradient_checkpointing()

    print(f"[load] controlnet config from {cfg.controlnet_config_repo} (architecture only) ...")
    config = WanControlnet.load_config(cfg.controlnet_config_repo)
    controlnet = WanControlnet.from_config(config)
    cast_respecting_fp32_modules(controlnet, torch.bfloat16)
    # Sanity asserts (catch upstream regressions silently demoting fragile layers)
    fp32_params = [n for n, p in controlnet.named_parameters()
                   if p.dtype == torch.float32]
    assert any("norm" in n or "time_embedder" in n or "scale_shift" in n
               for n in fp32_params), \
        "Expected norm/time_embedder/scale_shift params kept in fp32"
    assert any(p.dtype == torch.bfloat16 for p in controlnet.parameters()), \
        "Expected most controlnet params to be bf16"
    controlnet.enable_gradient_checkpointing()
    controlnet.train().to(device)

    trainable_params = sum(p.numel() for p in controlnet.parameters())
    _RESULTS_STATE["trainable_params"] = trainable_params
    print(f"[controlnet] trainable params: {trainable_params:,}")

    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        cfg.base_model_path, subfolder="scheduler",
    )
    scheduler.set_timesteps(cfg.num_train_timesteps_for_sampling, device=device)
    sigmas = scheduler.sigmas[:-1].to(device)        # (N,)
    timesteps_full = scheduler.timesteps.to(device)  # (N,)

    boundary_ratio, boundary_src = detect_boundary_ratio(
        cfg.base_model_path, dict(transformer.config),
        override=cfg.boundary_ratio_override,
    )
    high_noise_mask = sigmas >= boundary_ratio
    high_noise_indices = torch.where(high_noise_mask)[0]
    if high_noise_indices.numel() == 0:
        print(f"[warn] No timesteps satisfy sigma >= {boundary_ratio}; "
              "falling back to upper 50%.")
        high_noise_indices = torch.arange(0, len(sigmas) // 2, device=device)
        boundary_src += "+fallback_upper_50pct"
    print(f"[high-noise] boundary_ratio={boundary_ratio} "
          f"({boundary_src}); {len(high_noise_indices)} of {len(sigmas)} "
          f"timesteps in regime")
    _RESULTS_STATE["boundary_sigma"] = boundary_ratio
    _RESULTS_STATE["high_noise_rule"] = boundary_src

    # --- Optimizer ---
    import bitsandbytes as bnb
    optimizer = bnb.optim.AdamW8bit(
        controlnet.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay,
    )

    # --- Data ---
    dataset = BetaPairDataset(cfg.cache_dir, num_frames=cfg.num_frames)
    _RESULTS_STATE["pair_count"] = len(dataset)
    loader = DataLoader(
        dataset, batch_size=1, shuffle=True,
        num_workers=cfg.num_workers,
        persistent_workers=cfg.num_workers > 0,
        pin_memory=True,
        collate_fn=_collate_keep_meta,
    )
    steps_per_epoch = len(loader)
    print(f"[data] {len(dataset)} pairs, {steps_per_epoch} steps/epoch, "
          f"{cfg.num_epochs} epochs")

    # --- Training loop ---
    global_step = 0
    final_loss = float("nan")
    t_train_start = time.perf_counter()
    grad_assert_done = False

    for epoch in range(cfg.num_epochs):
        for step, batch in enumerate(loader):
            canny = batch["canny"].to(device, non_blocking=True)            # (1, 3, T, H, W)
            z_real = batch["latent"].to(device, non_blocking=True)          # (1, C, T_lat, H_lat, W_lat)
            prompt_embeds = batch["prompt_embeds"].to(device, non_blocking=True)  # (1, L, D)

            # 1. Sample noise + timestep in HIGH-NOISE regime
            noise = torch.randn_like(z_real)
            sel = torch.randint(0, len(high_noise_indices), (1,), device=device).item()
            t_idx = high_noise_indices[sel].item()
            sigma = sigmas[t_idx].to(z_real.dtype)
            t = timesteps_full[t_idx].expand(z_real.shape[0])

            # 2. Build noisy latent and FM target
            z_t = (1.0 - sigma) * z_real + sigma * noise
            v_target = (noise - z_real).float()

            # 3. ControlNet forward
            controlnet_states = controlnet(
                hidden_states=z_t,
                timestep=t,
                encoder_hidden_states=prompt_embeds,
                controlnet_states=canny,
                return_dict=False,
            )[0]
            # cast residuals to transformer dtype to match the addition site
            if isinstance(controlnet_states, (tuple, list)):
                controlnet_states_for_tx = [
                    s.to(dtype=transformer.dtype) for s in controlnet_states
                ]
            else:
                controlnet_states_for_tx = controlnet_states.to(dtype=transformer.dtype)

            # 4. Transformer forward (frozen, but gradients flow through residuals)
            v_pred = transformer(
                hidden_states=z_t,
                timestep=t,
                encoder_hidden_states=prompt_embeds,
                controlnet_states=controlnet_states_for_tx,
                controlnet_weight=1.0,
                controlnet_stride=3,
                return_dict=False,
            )[0]

            # 5. Loss in fp32 (FM targets fp32, predictions cast up)
            loss = F.mse_loss(v_pred.float(), v_target)

            # 6. Backward
            loss.backward()

            if not grad_assert_done:
                # Step-1 checks: transformer must NOT have grads; controlnet
                # must have at least one nonzero grad.
                tx_with_grad = [
                    n for n, p in transformer.named_parameters()
                    if p.grad is not None and p.grad.abs().sum() > 0
                ]
                assert not tx_with_grad, (
                    f"Transformer should have no grads but found {len(tx_with_grad)} "
                    f"({tx_with_grad[:3]}...)"
                )
                cn_with_grad = any(
                    p.grad is not None and p.grad.abs().sum() > 0
                    for p in controlnet.parameters()
                )
                assert cn_with_grad, "No controlnet param has nonzero grad after step 1"
                grad_assert_done = True
                print("[assert] grad-flow check passed at step 1")

            grad_norm = torch.nn.utils.clip_grad_norm_(
                controlnet.parameters(), cfg.grad_clip
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            # 7. Logging
            global_step += 1
            peak_mem = torch.cuda.max_memory_allocated() / 1e9
            wandb.log({
                "loss": loss.item(),
                "grad_norm": grad_norm.item() if torch.is_tensor(grad_norm) else float(grad_norm),
                "lr": optimizer.param_groups[0]["lr"],
                "controlnet_residual_norm": mean_residual_l2(controlnet_states),
                "timestep": float(t[0].item()),
                "sigma": float(sigma.item()),
                "step": global_step,
                "epoch": epoch,
                "gpu_mem_gb": peak_mem,
            }, step=global_step)
            final_loss = loss.item()

            # Memory tripwire
            if peak_mem > cfg.memory_tripwire_gb:
                raise RuntimeError(
                    f"GPU memory {peak_mem:.2f}GB exceeded tripwire "
                    f"{cfg.memory_tripwire_gb}GB at step {global_step}. "
                    "Suggest dropping T (e.g. T=5), smaller resolution, or 2-GPU DDP."
                )

            # 8. Periodic checkpoint
            if global_step % cfg.checkpoint_every == 0:
                ckpt_path = output_dir / f"{cfg.run_name}_step{global_step}.safetensors"
                _save_safetensors(controlnet, ckpt_path)
                print(f"[ckpt] {ckpt_path}")

            if cfg.max_steps is not None and global_step >= cfg.max_steps:
                print(f"[stop] reached --max_steps={cfg.max_steps}")
                break
        if cfg.max_steps is not None and global_step >= cfg.max_steps:
            break

    # --- Final checkpoint ---
    final_ckpt = output_dir / f"{cfg.run_name}_final.safetensors"
    _save_safetensors(controlnet, final_ckpt)
    print(f"[ckpt] {final_ckpt}")

    wall_time_s = time.perf_counter() - t_train_start
    _RESULTS_STATE.update({
        "final_loss": round(final_loss, 6),
        "gpu_peak_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
        "wall_time": _format_seconds(wall_time_s),
        "wall_time_seconds": round(wall_time_s, 1),
        "global_steps_completed": global_step,
        "final_checkpoint": str(final_ckpt),
    })
    _write_results()

    # --- End-of-run inference smoke ---
    if not cfg.skip_inference_smoke:
        try:
            mp4_path = inf_out_dir / f"{cfg.run_name}_final.mp4"
            run_inference_smoke(
                cfg=cfg, controlnet=controlnet, dataset=dataset,
                base_model_path=cfg.base_model_path, device=device,
                mp4_path=mp4_path,
            )
            _RESULTS_STATE["inference_mp4"] = str(mp4_path)
            try:
                wandb.log({"inference_video": wandb.Video(str(mp4_path))},
                          step=global_step)
            except Exception as e:
                print(f"[wandb] could not log video: {e}")
        except Exception as e:
            print(f"[inference-smoke] failed: {e}")
            traceback.print_exc()
            _RESULTS_STATE["inference_error"] = str(e)

    # --- Mark complete and autofill ---
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
    print(f"[done] {cfg.run_name} | loss={final_loss:.4f} | wall={_RESULTS_STATE['wall_time']}")


# ---------------- support pieces ----------------

def _collate_keep_meta(samples):
    """Tensor-stack tensor fields, list-collect scalar fields."""
    out = {}
    for k in samples[0]:
        if torch.is_tensor(samples[0][k]):
            out[k] = torch.stack([s[k] for s in samples], dim=0)
        else:
            out[k] = [s[k] for s in samples]
    return out


def _save_safetensors(model: nn.Module, path: Path) -> None:
    from safetensors.torch import save_file
    sd = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(sd, str(path))


def _format_seconds(s: float) -> str:
    s = int(s)
    h, r = divmod(s, 3600)
    m, sec = divmod(r, 60)
    return f"{h:02d}:{m:02d}:{sec:02d}"


def run_inference_smoke(cfg, controlnet, dataset, base_model_path: str,
                        device: torch.device, mp4_path: Path) -> None:
    """Run the existing pipeline once on a held-out (face, prompt) pair."""
    from diffusers import AutoencoderKLWan, WanTransformer3DModel
    from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
    from transformers import AutoTokenizer, UMT5EncoderModel
    from wan_transformer import CustomWanTransformer3DModel
    from wan_t2v_controlnet_pipeline import WanTextToVideoControlnetPipeline
    from PIL import Image

    print(f"[inference-smoke] loading full pipeline ...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, subfolder="tokenizer")
    text_encoder = UMT5EncoderModel.from_pretrained(
        base_model_path, subfolder="text_encoder", torch_dtype=torch.bfloat16,
    ).eval()
    vae = AutoencoderKLWan.from_pretrained(
        base_model_path, subfolder="vae", torch_dtype=torch.bfloat16,
    ).eval()
    transformer = CustomWanTransformer3DModel.from_pretrained(
        base_model_path, subfolder="transformer", torch_dtype=torch.bfloat16,
    ).eval()
    transformer_2 = WanTransformer3DModel.from_pretrained(
        base_model_path, subfolder="transformer_2", torch_dtype=torch.bfloat16,
    ).eval()
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        base_model_path, subfolder="scheduler",
    )

    boundary_ratio, _ = detect_boundary_ratio(
        base_model_path, dict(transformer.config),
        override=cfg.boundary_ratio_override,
    )

    pipe = WanTextToVideoControlnetPipeline(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        transformer=transformer,
        transformer_2=transformer_2,
        vae=vae,
        controlnet=controlnet,
        scheduler=scheduler,
        boundary_ratio=boundary_ratio,
    )
    pipe.to(device)

    # Pull one sample's canny + slug -> long prompt text
    rec = dataset.records[0]
    cache_dir = dataset.cache_dir
    canny_u8 = torch.load(cache_dir / rec["canny_path"], map_location="cpu")
    canny_img = Image.fromarray(canny_u8.permute(1, 2, 0).numpy())  # (H, W, 3)
    from training.input_prompts import PROMPTS_BATCH_1, PROMPTS_BATCH_2
    prompt_text = {**PROMPTS_BATCH_1, **PROMPTS_BATCH_2}[rec["slug"]]

    print(f"[inference-smoke] face_idx={rec['face_idx']} slug={rec['slug']}")
    out = pipe(
        controlnet_frames=[canny_img] * cfg.num_frames,
        prompt=prompt_text,
        negative_prompt="bad quality, worst quality",
        height=cfg.height, width=cfg.width,
        num_frames=cfg.num_frames,
        num_inference_steps=30,
        guidance_scale=5.0,
        controlnet_weight=1.0,
        controlnet_stride=3,
        output_type="np",
    )
    frames = out.frames[0]  # (T, H, W, 3) in [0, 1] float32
    _save_video(frames, mp4_path, fps=8)
    print(f"[inference-smoke] wrote {mp4_path}")


def _save_video(frames_np: np.ndarray, path: Path, fps: int = 8) -> None:
    import imageio
    arr = (np.clip(frames_np, 0.0, 1.0) * 255).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(path), list(arr), fps=fps, codec="libx264")


if __name__ == "__main__":
    main()

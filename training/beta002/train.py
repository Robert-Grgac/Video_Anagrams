"""BETA2 training: phase-alternating two-expert WanControlnet training.

Sequel to ``train_beta.py``. Same flow-matching MSE loss and cold-start
controlnet, but the active transformer expert and the sigma sampling regime
both rotate every ``--cycle_steps`` optimizer steps. Only one expert is resident
on GPU at a time; swaps free + reload from the local diffusers cache.

End-of-run inference writes two videos: one at
``controlnet_guidance_end=0.125`` (sanity vs beta-001) and one at
``controlnet_guidance_end=1.0`` (the new capability beta-002 was trained for),
both at ``controlnet_weight=1.0``.

Run lifecycle JSON (``training_cards/{run_id}_results.json``) is written at
start, updated at end, and keyed for ``training/autofill_card.py``.
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
from torch.utils.data import DataLoader

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


# ---------------- expert swap ----------------

def load_expert(base_model_path: str, subfolder: str, device: torch.device):
    """Load one transformer expert and prep it for frozen-with-residuals use."""
    from wan_transformer import CustomWanTransformer3DModel
    nxt = CustomWanTransformer3DModel.from_pretrained(
        base_model_path, subfolder=subfolder, torch_dtype=torch.bfloat16,
    )
    nxt.requires_grad_(False).eval()
    nxt.enable_gradient_checkpointing()
    return nxt.to(device)


def free_then_load_expert(base_model_path: str, new_subfolder: str,
                          device: torch.device) -> object:
    """Free dangling CUDA blocks then load + move the new expert to GPU.

    The CALLER must release its own reference to the previous transformer
    BEFORE calling this (e.g. ``del transformer``). Otherwise the old expert
    stays resident through ``from_pretrained``, doubling peak VRAM and OOMing
    on a 45 GB A40.
    """
    gc.collect()
    torch.cuda.empty_cache()
    return load_expert(base_model_path, new_subfolder, device)


def next_subfolder_for(current_phase: str) -> str:
    return "transformer_2" if current_phase == "high" else "transformer"


# ---------------- main ----------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--cache_dir", type=str, required=True)
    p.add_argument("--base_model_path", type=str, required=True)
    p.add_argument("--controlnet_config_repo", type=str, required=True,
                   help="HF repo id or local path containing a controlnet config.json. "
                        "Weights are NOT loaded; cold init only.")
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--inference_output_dir", type=str, default=None)
    p.add_argument("--card_path", type=str, default=None)
    p.add_argument("--wandb_project", type=str, default="wan-controlnet-beta")
    p.add_argument("--run_name", type=str, required=True)

    p.add_argument("--num_frames", type=int, default=9)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=512)

    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--num_epochs", type=int, default=1)
    p.add_argument("--max_steps", type=int, default=10000)
    p.add_argument("--cycle_steps", type=int, default=500,
                   help="Number of optimizer steps per phase before swapping the active expert.")
    p.add_argument("--start_phase", type=str, default="high",
                   choices=["high", "low"],
                   help="Which expert to load first.")

    p.add_argument("--num_train_timesteps_for_sampling", type=int, default=1000)
    p.add_argument("--boundary_ratio_override", type=float, default=None)
    p.add_argument("--checkpoint_every", type=int, default=2000)
    p.add_argument("--memory_tripwire_gb", type=float, default=43.0)

    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
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
        _RESULTS_PATH = Path("training_cards") / "beta002" / f"{cfg.run_name}_results.json"

    _RESULTS_STATE.update({
        "status": "running",
        "date_started": _now_iso(),
        "git_sha": _git_sha(),
        "cluster_partition": os.environ.get("SLURM_JOB_PARTITION", "unknown"),
        "run_name": cfg.run_name,
        "cycle_steps": cfg.cycle_steps,
        "start_phase": cfg.start_phase,
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
        raise RuntimeError("BETA2 training requires CUDA; got CPU.")

    # --- Models ---
    from wan_controlnet import WanControlnet
    from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

    phase = cfg.start_phase
    start_subfolder = "transformer" if phase == "high" else "transformer_2"
    print(f"[load] starting expert: {start_subfolder} (phase={phase}) ...")
    transformer = load_expert(cfg.base_model_path, start_subfolder, device)

    # detect boundary BEFORE we may swap away from this transformer's config.
    # Both experts in the same checkpoint share the boundary, so detecting once
    # is correct.
    boundary_ratio, boundary_src = detect_boundary_ratio(
        cfg.base_model_path, dict(transformer.config),
        override=cfg.boundary_ratio_override,
    )

    print(f"[load] controlnet config from {cfg.controlnet_config_repo} (architecture only) ...")
    config = WanControlnet.load_config(cfg.controlnet_config_repo)
    controlnet = WanControlnet.from_config(config)
    cast_respecting_fp32_modules(controlnet, torch.bfloat16)
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
    sigmas = scheduler.sigmas[:-1].to(device)
    timesteps_full = scheduler.timesteps.to(device)

    high_noise_indices = torch.where(sigmas >= boundary_ratio)[0]
    low_noise_indices = torch.where(sigmas < boundary_ratio)[0]
    if high_noise_indices.numel() == 0:
        print(f"[warn] No timesteps satisfy sigma >= {boundary_ratio}; "
              "falling back to upper 50%.")
        high_noise_indices = torch.arange(0, len(sigmas) // 2, device=device)
        boundary_src += "+fallback_upper_50pct_high"
    if low_noise_indices.numel() == 0:
        print(f"[warn] No timesteps satisfy sigma < {boundary_ratio}; "
              "falling back to lower 50%.")
        low_noise_indices = torch.arange(len(sigmas) // 2, len(sigmas), device=device)
        boundary_src += "+fallback_lower_50pct_low"
    print(f"[boundary] ratio={boundary_ratio} ({boundary_src}); "
          f"high={len(high_noise_indices)} low={len(low_noise_indices)} of {len(sigmas)}")
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
          f"{cfg.num_epochs} epochs, max_steps={cfg.max_steps}")

    # --- Training loop ---
    global_step = 0
    phase_step = 0
    n_swaps = 0
    swap_overhead_total = 0.0
    final_loss = float("nan")
    grad_assert_done = False

    # per-phase running averages
    phase_loss_sum = {"high": 0.0, "low": 0.0}
    phase_loss_count = {"high": 0, "low": 0}

    t_train_start = time.perf_counter()

    done = False
    for epoch in range(cfg.num_epochs):
        if done:
            break
        for step, batch in enumerate(loader):
            # phase swap at cycle boundary (only if more work remains).
            # Caller-side `del` is required so the old expert refcount drops
            # to 0 before we call from_pretrained for the new one.
            if phase_step >= cfg.cycle_steps and global_step < cfg.max_steps:
                phase_old = phase
                new_sub = next_subfolder_for(phase)
                swap_t0 = time.perf_counter()
                del transformer
                transformer = free_then_load_expert(
                    cfg.base_model_path, new_sub, device,
                )
                torch.cuda.synchronize()
                swap_dt = time.perf_counter() - swap_t0
                phase = "low" if phase == "high" else "high"
                phase_step = 0
                n_swaps += 1
                swap_overhead_total += swap_dt
                print(f"[swap] {phase_old} -> {phase} at step {global_step} "
                      f"({swap_dt:.1f}s)")
                wandb.log({
                    "event": "phase_swap",
                    "swap_from": 0 if phase_old == "high" else 1,
                    "swap_to": 0 if phase == "high" else 1,
                    "swap_seconds": swap_dt,
                    "step": global_step,
                }, step=global_step)

            indices = high_noise_indices if phase == "high" else low_noise_indices
            active_expert_id = 0 if phase == "high" else 1

            canny = batch["canny"].to(device, non_blocking=True)
            z_real = batch["latent"].to(device, non_blocking=True)
            prompt_embeds = batch["prompt_embeds"].to(device, non_blocking=True)

            # 1. Sample noise + timestep in CURRENT phase regime
            noise = torch.randn_like(z_real)
            sel = torch.randint(0, len(indices), (1,), device=device).item()
            t_idx = indices[sel].item()
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

            # 5. Loss
            loss = F.mse_loss(v_pred.float(), v_target)

            # 6. Backward
            loss.backward()

            if not grad_assert_done:
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
            phase_step += 1
            cycle_idx = (global_step - 1) // (2 * cfg.cycle_steps)

            phase_loss_sum[phase] += float(loss.item())
            phase_loss_count[phase] += 1

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
                "active_expert": active_expert_id,
                "phase_step": phase_step - 1,
                "cycle_idx": cycle_idx,
            }, step=global_step)
            final_loss = loss.item()

            if peak_mem > cfg.memory_tripwire_gb:
                raise RuntimeError(
                    f"GPU memory {peak_mem:.2f}GB exceeded tripwire "
                    f"{cfg.memory_tripwire_gb}GB at step {global_step}."
                )

            if global_step % cfg.checkpoint_every == 0:
                ckpt_path = output_dir / f"{cfg.run_name}_step{global_step}.safetensors"
                _save_safetensors(controlnet, ckpt_path)
                print(f"[ckpt] {ckpt_path}")

            if cfg.max_steps is not None and global_step >= cfg.max_steps:
                print(f"[stop] reached --max_steps={cfg.max_steps}")
                done = True
                break

    # --- Final checkpoint ---
    final_ckpt = output_dir / f"{cfg.run_name}_final.safetensors"
    _save_safetensors(controlnet, final_ckpt)
    print(f"[ckpt] {final_ckpt}")

    wall_time_s = time.perf_counter() - t_train_start
    high_avg = (phase_loss_sum["high"] / phase_loss_count["high"]
                if phase_loss_count["high"] else float("nan"))
    low_avg = (phase_loss_sum["low"] / phase_loss_count["low"]
               if phase_loss_count["low"] else float("nan"))
    _RESULTS_STATE.update({
        "final_loss": round(final_loss, 6),
        "gpu_peak_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
        "wall_time": _format_seconds(wall_time_s),
        "wall_time_seconds": round(wall_time_s, 1),
        "global_steps_completed": global_step,
        "final_checkpoint": str(final_ckpt),
        "n_swaps": n_swaps,
        "swap_overhead_sec": round(swap_overhead_total, 1),
        "high_phase_avg_loss": round(high_avg, 6),
        "low_phase_avg_loss": round(low_avg, 6),
    })
    _write_results()

    # --- End-of-run inference smoke (TWO videos) ---
    if not cfg.skip_inference_smoke:
        # Free the active training-time expert before building the inference
        # pipeline, which loads BOTH experts.
        try:
            del transformer
            gc.collect()
            torch.cuda.empty_cache()
        except Exception:
            pass
        try:
            mp4_paths = run_inference_smoke_beta2(
                cfg=cfg, controlnet=controlnet, dataset=dataset,
                base_model_path=cfg.base_model_path, device=device,
                inf_out_dir=inf_out_dir,
                cn_guidance_ends=(0.125, 1.0),
            )
            _RESULTS_STATE["inference_mp4s"] = [str(p) for p in mp4_paths]
            for p in mp4_paths:
                try:
                    wandb.log({f"inference_video_{p.stem}": wandb.Video(str(p))},
                              step=global_step)
                except Exception as e:
                    print(f"[wandb] could not log video {p.name}: {e}")
        except Exception as e:
            print(f"[inference-smoke] failed: {e}")
            traceback.print_exc()
            _RESULTS_STATE["inference_error"] = str(e)

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
    print(f"[done] {cfg.run_name} | loss={final_loss:.4f} | "
          f"wall={_RESULTS_STATE['wall_time']} | swaps={n_swaps}")


# ---------------- inference ----------------

def run_inference_smoke_beta2(cfg, controlnet, dataset, base_model_path: str,
                              device: torch.device, inf_out_dir: Path,
                              cn_guidance_ends=(0.125, 1.0)) -> list[Path]:
    """Build the full pipeline once and write one mp4 per controlnet_guidance_end."""
    from diffusers import AutoencoderKLWan
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
    transformer_2 = CustomWanTransformer3DModel.from_pretrained(
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
    # Two experts won't both fit; offload like run_inference_beta.py does.
    pipe.enable_model_cpu_offload()
    from accelerate.hooks import remove_hook_from_module
    remove_hook_from_module(pipe.controlnet, recurse=True)
    pipe.controlnet.to("cuda")

    rec = dataset.records[0]
    cache_dir = dataset.cache_dir
    canny_u8 = torch.load(cache_dir / rec["canny_path"], map_location="cpu")
    canny_img = Image.fromarray(canny_u8.permute(1, 2, 0).numpy())
    from training.input_prompts import PROMPTS_BATCH_1, PROMPTS_BATCH_2
    prompt_text = {**PROMPTS_BATCH_1, **PROMPTS_BATCH_2}[rec["slug"]]
    print(f"[inference-smoke] face_idx={rec['face_idx']} slug={rec['slug']}")

    out_paths: list[Path] = []
    for cn_end in cn_guidance_ends:
        # Re-pin controlnet on each call: model_cpu_offload re-attaches an
        # accelerate hook on every pipeline __call__.
        remove_hook_from_module(pipe.controlnet, recurse=True)
        pipe.controlnet.to("cuda")
        generator = torch.Generator().manual_seed(cfg.seed)
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
            controlnet_guidance_start=0.0,
            controlnet_guidance_end=cn_end,
            generator=generator,
            output_type="np",
        )
        frames = out.frames[0]
        end_str = f"{cn_end:.3f}".rstrip("0").rstrip(".").replace(".", "p")
        mp4_path = inf_out_dir / f"{cfg.run_name}_final_e{end_str}.mp4"
        _save_video(frames, mp4_path, fps=8)
        print(f"[inference-smoke] wrote {mp4_path} (cn_end={cn_end})")
        out_paths.append(mp4_path)
    return out_paths


def _save_video(frames_np: np.ndarray, path: Path, fps: int = 8) -> None:
    import imageio
    arr = (np.clip(frames_np, 0.0, 1.0) * 255).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(path), list(arr), fps=fps, codec="libx264")


if __name__ == "__main__":
    main()

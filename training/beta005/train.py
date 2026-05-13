"""BETA5 training: cold-start WanControlnet on the HIGH-noise expert.

Sequel to ``train_beta3.py``: same supervisor recipe (EMA, AdamW8bit, lower
weight decay, gradient accumulation, ``zero_module`` cold init), but applied
to the high-noise expert (``transformer/`` subfolder) instead of
``transformer_2``. Adds two new capabilities over train_beta3:

* ``--micro_batch_size`` for real batching when VRAM allows. The DataLoader
  yields micro-batches of size B; per-micro-batch we sample B independent
  sigmas (one per sample) so eff_batch=accum*B reduces both σ-aliasing
  variance and per-face variance simultaneously. With B=1 the path is
  bit-equivalent to train_beta3.
* Hardware-aware attention backend: on devices with compute capability ≥ 12
  (RTX PRO 6000 Blackwell), we explicitly call ``set_attention_backend('native')``
  on the loaded transformer to avoid the ``flash_attn_func is None`` crash that
  killed beta-004's first attempt on hpc-node31. Belt-and-braces with the
  sbatch-side ``DIFFUSERS_ATTN_BACKEND=native`` export.

Drives the ``beta-005`` run only — cold-start, high-noise, no warm-start
flag at all (compare train_beta3.py which carried it for beta-003).

End-of-run inference smoke writes a single-CN video (Plan A equivalent of
beta-001's smoke), since we don't yet have a paired low-CN to slot into the
dual pipeline.
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
    _save_state_dict,
    _maybe_force_native_attention,
    _build_controlnet_from_checkpoint,
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


# ---------------- main ----------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--cache_dir", type=str, required=True)
    p.add_argument("--base_model_path", type=str, required=True)
    p.add_argument("--controlnet_config_repo", type=str, required=True,
                   help="HF repo id or local path containing a controlnet config.json. "
                        "Architecture only; cold init from from_config.")
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--inference_output_dir", type=str, default=None)
    p.add_argument("--card_path", type=str, default=None)
    p.add_argument("--wandb_project", type=str, default="wan-controlnet-beta")
    p.add_argument("--run_name", type=str, required=True)

    p.add_argument("--num_frames", type=int, default=9)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=512)

    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--num_epochs", type=int, default=1)
    p.add_argument("--max_steps", type=int, default=125,
                   help="Number of EFFECTIVE optimizer steps. Total micro-steps = "
                        "max_steps * gradient_accumulation_steps * micro_batch_size.")
    p.add_argument("--micro_batch_size", type=int, default=1,
                   help="DataLoader micro-batch (per forward/backward call). On Blackwell "
                        "(96 GB) values up to 8 fit comfortably with grad checkpointing.")
    p.add_argument("--gradient_accumulation_steps", type=int, default=32)
    p.add_argument("--ema_decay", type=float, default=0.999)
    p.add_argument("--ema_update_after_step", type=int, default=100)
    p.add_argument("--loss_ema_window", type=int, default=20,
                   help="Window length (in EFFECTIVE steps) for the loss EMA logged to wandb.")

    p.add_argument("--num_train_timesteps_for_sampling", type=int, default=1000)
    p.add_argument("--boundary_ratio_override", type=float, default=None)
    p.add_argument("--checkpoint_every", type=int, default=50)
    p.add_argument("--memory_tripwire_gb", type=float, default=90.0)

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
        _RESULTS_PATH = Path("training_cards") / "beta005" / f"{cfg.run_name}_results.json"

    init_mode = "cold"  # beta-005 is cold-only by design
    effective_batch = cfg.micro_batch_size * cfg.gradient_accumulation_steps
    _RESULTS_STATE.update({
        "status": "running",
        "date_started": _now_iso(),
        "git_sha": _git_sha(),
        "cluster_partition": os.environ.get("SLURM_JOB_PARTITION", "unknown"),
        "run_name": cfg.run_name,
        "init_mode": init_mode,
        "micro_batch_size": cfg.micro_batch_size,
        "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
        "effective_batch_size": effective_batch,
        "ema_decay": cfg.ema_decay,
        "ema_update_after_step": cfg.ema_update_after_step,
        "loss_ema_window": cfg.loss_ema_window,
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
        raise RuntimeError("BETA5 training requires CUDA; got CPU.")
    try:
        major, minor = torch.cuda.get_device_capability(0)
        gpu_name = torch.cuda.get_device_name(0)
        print(f"[gpu] {gpu_name} (compute_cap={major}.{minor})")
        _RESULTS_STATE["gpu_name"] = gpu_name
        _RESULTS_STATE["gpu_compute_cap"] = f"{major}.{minor}"
    except Exception:
        pass

    # --- Models ---
    from wan_controlnet import WanControlnet
    from wan_transformer import CustomWanTransformer3DModel
    from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

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
    controlnet = WanControlnet.from_config(config)
    cast_respecting_fp32_modules(controlnet, torch.bfloat16)
    fp32_params = [n for n, p in controlnet.named_parameters()
                   if p.dtype == torch.float32]
    assert any("norm" in n or "time_embedder" in n or "scale_shift" in n
               for n in fp32_params), \
        "Expected norm/time_embedder/scale_shift params kept in fp32"
    assert any(p.dtype == torch.bfloat16 for p in controlnet.parameters()), \
        "Expected most controlnet params to be bf16"
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

    # High-noise: sigma >= boundary_ratio (mirrors train_beta.py).
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

    # --- Optimizer ---
    import bitsandbytes as bnb
    optimizer = bnb.optim.AdamW8bit(
        controlnet.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay,
    )

    # --- EMA shadow ---
    from ema_pytorch import EMA
    ema = EMA(
        controlnet,
        beta=cfg.ema_decay,
        update_after_step=cfg.ema_update_after_step,
        update_every=1,
    )
    ema.to(device)

    # --- Data ---
    dataset = BetaPairDataset(cfg.cache_dir, num_frames=cfg.num_frames)
    _RESULTS_STATE["pair_count"] = len(dataset)
    loader = DataLoader(
        dataset, batch_size=cfg.micro_batch_size, shuffle=True,
        num_workers=cfg.num_workers,
        persistent_workers=cfg.num_workers > 0,
        pin_memory=True,
        drop_last=True,
        collate_fn=_collate_keep_meta,
    )
    steps_per_epoch = len(loader)
    print(f"[data] {len(dataset)} pairs, micro_batch={cfg.micro_batch_size}, "
          f"{steps_per_epoch} micro-steps/epoch, {cfg.num_epochs} epochs, "
          f"max_steps={cfg.max_steps} (effective)")

    # --- Training loop ---
    global_step = 0           # optimizer steps (effective)
    micro_step = 0            # forward/backward calls
    final_loss = float("nan")
    grad_assert_done = False

    accum_loss_sum = 0.0
    accum_residual_l2: list[float] = []
    accum_sigmas: list[float] = []  # all sigmas seen this effective step (B per micro-step)
    accum_t_last = 0.0

    eff_step_log: list[tuple[float, float]] = []  # (sigma_mean, loss) per effective step
    loss_ema_value: Optional[float] = None
    loss_ema_alpha = 2.0 / (cfg.loss_ema_window + 1)

    optimizer.zero_grad(set_to_none=True)
    t_train_start = time.perf_counter()

    done = False
    for epoch in range(cfg.num_epochs):
        if done:
            break
        for step, batch in enumerate(loader):
            canny = batch["canny"].to(device, non_blocking=True)
            z_real = batch["latent"].to(device, non_blocking=True)
            prompt_embeds = batch["prompt_embeds"].to(device, non_blocking=True)
            B = z_real.shape[0]

            # 1. Sample B independent sigmas in the high-noise regime.
            #    Independent draws per sample reduces σ-aliasing variance — the
            #    dominant per-step noise source per the beta-001 wandb summary.
            sel = torch.randint(0, len(high_noise_indices), (B,), device=device)
            t_idx = high_noise_indices[sel]                     # [B]
            sigma = sigmas[t_idx].to(z_real.dtype)              # [B]
            t = timesteps_full[t_idx]                           # [B]
            sigma_b = sigma.view(B, 1, 1, 1, 1)

            # 2. Build noisy latent + FM target
            noise = torch.randn_like(z_real)
            z_t = (1.0 - sigma_b) * z_real + sigma_b * noise
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

            # 5. Loss (FM in fp32). Scale by accum so summed grads = mean over eff batch.
            loss = F.mse_loss(v_pred.float(), v_target)
            loss_scaled = loss / cfg.gradient_accumulation_steps

            # 6. Backward
            loss_scaled.backward()

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

            accum_loss_sum += float(loss_scaled.item())
            accum_residual_l2.append(mean_residual_l2(controlnet_states))
            accum_sigmas.extend(sigma.detach().float().cpu().tolist())
            accum_t_last = float(t[-1].item())
            micro_step += 1

            # 7. Optimizer step + EMA update only on accumulation boundaries.
            if micro_step % cfg.gradient_accumulation_steps == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    controlnet.parameters(), cfg.grad_clip
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                ema.update()

                global_step += 1
                final_loss = accum_loss_sum
                sigma_mean = float(np.mean(accum_sigmas))
                sigma_std = float(np.std(accum_sigmas)) if len(accum_sigmas) > 1 else 0.0
                eff_step_log.append((sigma_mean, accum_loss_sum))

                if loss_ema_value is None:
                    loss_ema_value = accum_loss_sum
                else:
                    loss_ema_value = (loss_ema_alpha * accum_loss_sum
                                      + (1.0 - loss_ema_alpha) * loss_ema_value)

                peak_mem = torch.cuda.max_memory_allocated() / 1e9
                try:
                    ema_decay_current = float(ema.get_current_decay())
                except Exception:
                    ema_decay_current = float(getattr(ema, "beta", cfg.ema_decay))
                wandb.log({
                    "loss": accum_loss_sum,
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

                if peak_mem > cfg.memory_tripwire_gb:
                    raise RuntimeError(
                        f"GPU memory {peak_mem:.2f}GB exceeded tripwire "
                        f"{cfg.memory_tripwire_gb}GB at step {global_step}."
                    )

                if global_step % cfg.checkpoint_every == 0:
                    ema_ckpt = output_dir / f"{cfg.run_name}_step{global_step}.safetensors"
                    _save_state_dict(ema.ema_model.state_dict(), ema_ckpt)
                    print(f"[ckpt] {ema_ckpt}  (EMA)")

                accum_loss_sum = 0.0
                accum_residual_l2 = []
                accum_sigmas = []

                if cfg.max_steps is not None and global_step >= cfg.max_steps:
                    print(f"[stop] reached --max_steps={cfg.max_steps} (effective)")
                    done = True
                    break

    # Drop any leftover partial accumulation, same hygiene as train_beta3.
    if micro_step % cfg.gradient_accumulation_steps != 0:
        print(f"[stop] discarding {micro_step % cfg.gradient_accumulation_steps} "
              "micro-step(s) of partial accumulation at exit")
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

    # --- End-of-run inference smoke (single-CN, like beta-001's) ---
    if not cfg.skip_inference_smoke:
        try:
            del transformer
            del controlnet
            del ema
            gc.collect()
            torch.cuda.empty_cache()
        except Exception:
            pass

        try:
            mp4_path = inf_out_dir / f"{cfg.run_name}_smoke_singleCN_e1.mp4"
            run_single_cn_smoke(
                cfg=cfg, ema_checkpoint=final_ema,
                dataset=dataset, device=device, mp4_path=mp4_path,
            )
            _RESULTS_STATE["inference_mp4s"] = [str(mp4_path)]
            try:
                wandb.log({f"inference_video_{mp4_path.stem}": wandb.Video(str(mp4_path))},
                          step=global_step)
            except Exception as e:
                print(f"[wandb] could not log video: {e}")
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
          f"loss_ema={loss_ema_value:.4f} | wall={_RESULTS_STATE['wall_time']} | init={init_mode}")


# ---------------- inference smoke ----------------

def run_single_cn_smoke(cfg, ema_checkpoint: Path, dataset,
                        device: torch.device, mp4_path: Path) -> Path:
    """Single-CN inference smoke: this run's EMA controlnet at weight=1.0."""
    from diffusers import AutoencoderKLWan
    from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
    from transformers import AutoTokenizer, UMT5EncoderModel
    from wan_transformer import CustomWanTransformer3DModel
    from wan_t2v_controlnet_pipeline import WanTextToVideoControlnetPipeline
    from accelerate.hooks import remove_hook_from_module
    from PIL import Image

    print(f"[inference-smoke single-CN] loading full pipeline ...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model_path, subfolder="tokenizer")
    text_encoder = UMT5EncoderModel.from_pretrained(
        cfg.base_model_path, subfolder="text_encoder", torch_dtype=torch.bfloat16,
    ).eval()
    vae = AutoencoderKLWan.from_pretrained(
        cfg.base_model_path, subfolder="vae", torch_dtype=torch.bfloat16,
    ).eval()
    transformer = CustomWanTransformer3DModel.from_pretrained(
        cfg.base_model_path, subfolder="transformer", torch_dtype=torch.bfloat16,
    ).eval()
    transformer_2 = CustomWanTransformer3DModel.from_pretrained(
        cfg.base_model_path, subfolder="transformer_2", torch_dtype=torch.bfloat16,
    ).eval()
    _maybe_force_native_attention(transformer, "transformer (smoke)")
    _maybe_force_native_attention(transformer_2, "transformer_2 (smoke)")

    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        cfg.base_model_path, subfolder="scheduler",
    )
    boundary_ratio, _ = detect_boundary_ratio(
        cfg.base_model_path, dict(transformer.config),
        override=cfg.boundary_ratio_override,
    )
    controlnet = _build_controlnet_from_checkpoint(
        cfg.controlnet_config_repo, ema_checkpoint,
    )
    _maybe_force_native_attention(controlnet, "controlnet (smoke)")

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
    pipe.enable_model_cpu_offload()
    remove_hook_from_module(pipe.controlnet, recurse=True)
    pipe.controlnet.to("cuda")

    rec = dataset.records[0]
    cache_dir = dataset.cache_dir
    canny_u8 = torch.load(cache_dir / rec["canny_path"], map_location="cpu")
    canny_img = Image.fromarray(canny_u8.permute(1, 2, 0).numpy())
    from training.input_prompts import PROMPTS_BATCH_1, PROMPTS_BATCH_2
    prompt_text = {**PROMPTS_BATCH_1, **PROMPTS_BATCH_2}[rec["slug"]]
    print(f"[inference-smoke single-CN] face_idx={rec['face_idx']} slug={rec['slug']}")

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
        generator=generator,
        output_type="np",
    )
    frames = out.frames[0]
    _save_video(frames, mp4_path, fps=8)
    print(f"[inference-smoke single-CN] wrote {mp4_path}")
    return mp4_path


def _save_video(frames_np: np.ndarray, path: Path, fps: int = 8) -> None:
    import imageio
    arr = (np.clip(frames_np, 0.0, 1.0) * 255).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(path), list(arr), fps=fps, codec="libx264")


if __name__ == "__main__":
    main()

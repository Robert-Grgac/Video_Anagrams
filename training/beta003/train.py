"""BETA3 training: dedicated low-phase WanControlnet training (no swaps).

Sequel to ``train_beta2.py``. Trains only the low-noise expert
(``transformer_2``) in a single-phase loop, optionally warm-started from a
checkpoint. Adds gradient accumulation (effective batch = 2), EMA on
controlnet weights, and a lower default weight decay than train_beta2.

The same script drives both runs in the BETA3 family:

* ``beta-003`` — warm-start: pass ``--warm_start_checkpoint`` pointing at
  ``beta-001_final.safetensors``.
* ``beta-004`` — cold-start: omit ``--warm_start_checkpoint``.

End-of-run inference smoke writes one video for the dual-CN pipeline
(beta-001 frozen as high-CN + this run's EMA as low-CN), and — if the run
was warm-started — also writes a single-CN Plan-A video. Both at
``controlnet_weight=1.0``, ``controlnet_guidance_end=1.0``.
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


def _save_state_dict(state_dict: dict, path: Path) -> None:
    from safetensors.torch import save_file
    sd = {k: v.detach().cpu().contiguous() for k, v in state_dict.items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(sd, str(path))


# ---------------- main ----------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--cache_dir", type=str, required=True)
    p.add_argument("--base_model_path", type=str, required=True)
    p.add_argument("--controlnet_config_repo", type=str, required=True,
                   help="HF repo id or local path containing a controlnet config.json. "
                        "Weights are NOT loaded from this; warm-start uses --warm_start_checkpoint.")
    p.add_argument("--warm_start_checkpoint", type=str, default=None,
                   help="Optional .safetensors path. If set, weights are loaded into the "
                        "controlnet after from_config() and before training. If omitted, "
                        "the controlnet keeps its from_config cold-init weights.")
    p.add_argument("--dual_smoke_high_checkpoint", type=str, default=None,
                   help="Optional .safetensors path used as the high-CN in the end-of-run "
                        "dual-CN inference smoke. For beta-003 this is the same file as "
                        "--warm_start_checkpoint; for beta-004 it's beta-001_final.safetensors. "
                        "If unset, the dual-CN smoke is skipped.")
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
    p.add_argument("--max_steps", type=int, default=4000,
                   help="Number of EFFECTIVE optimizer steps. Total micro-steps = "
                        "max_steps * gradient_accumulation_steps.")
    p.add_argument("--gradient_accumulation_steps", type=int, default=2)
    p.add_argument("--ema_decay", type=float, default=0.999)
    p.add_argument("--ema_update_after_step", type=int, default=100,
                   help="Number of optimizer steps before the EMA shadow starts updating; "
                        "protects EMA from absorbing the early warm-start transient.")

    p.add_argument("--num_train_timesteps_for_sampling", type=int, default=1000)
    p.add_argument("--boundary_ratio_override", type=float, default=None)
    p.add_argument("--checkpoint_every", type=int, default=1000)
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
        _RESULTS_PATH = Path("training_cards") / "beta003" / f"{cfg.run_name}_results.json"

    init_mode = "warm" if cfg.warm_start_checkpoint else "cold"
    effective_batch = 1 * cfg.gradient_accumulation_steps  # micro-batch=1
    _RESULTS_STATE.update({
        "status": "running",
        "date_started": _now_iso(),
        "git_sha": _git_sha(),
        "cluster_partition": os.environ.get("SLURM_JOB_PARTITION", "unknown"),
        "run_name": cfg.run_name,
        "init_mode": init_mode,
        "effective_batch_size": effective_batch,
        "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
        "ema_decay": cfg.ema_decay,
        "ema_update_after_step": cfg.ema_update_after_step,
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
        raise RuntimeError("BETA3 training requires CUDA; got CPU.")

    # --- Models ---
    from wan_controlnet import WanControlnet
    from wan_transformer import CustomWanTransformer3DModel
    from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

    print(f"[load] low-noise transformer (transformer_2) from {cfg.base_model_path} ...")
    transformer = CustomWanTransformer3DModel.from_pretrained(
        cfg.base_model_path, subfolder="transformer_2", torch_dtype=torch.bfloat16,
    )
    transformer.requires_grad_(False).eval().to(device)
    transformer.enable_gradient_checkpointing()

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

    if cfg.warm_start_checkpoint:
        from safetensors.torch import load_file
        print(f"[warm-start] loading {cfg.warm_start_checkpoint} ...")
        sd = load_file(cfg.warm_start_checkpoint)
        missing, unexpected = controlnet.load_state_dict(sd, strict=False)
        assert len(missing) == 0 and len(unexpected) == 0, (
            f"warm-start key mismatch: missing={len(missing)} unexpected={len(unexpected)} "
            f"(missing first 3: {missing[:3]}, unexpected first 3: {unexpected[:3]})"
        )
        print(f"[warm-start] loaded; missing=0 unexpected=0")
        _RESULTS_STATE["warm_start_missing_keys"] = 0
        _RESULTS_STATE["warm_start_unexpected_keys"] = 0
    else:
        print("[cold-start] using fresh from_config weights (output projections zeroed)")
        # Recorded as None so autofill prints something stable rather than missing.
        _RESULTS_STATE["warm_start_missing_keys"] = "n/a"
        _RESULTS_STATE["warm_start_unexpected_keys"] = "n/a"

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

    low_noise_indices = torch.where(sigmas < boundary_ratio)[0]
    if low_noise_indices.numel() == 0:
        print(f"[warn] No timesteps satisfy sigma < {boundary_ratio}; "
              "falling back to lower 50%.")
        low_noise_indices = torch.arange(len(sigmas) // 2, len(sigmas), device=device)
        boundary_src += "+fallback_lower_50pct_low"
    print(f"[boundary] ratio={boundary_ratio} ({boundary_src}); "
          f"low={len(low_noise_indices)} of {len(sigmas)} timesteps in regime")
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
        dataset, batch_size=1, shuffle=True,
        num_workers=cfg.num_workers,
        persistent_workers=cfg.num_workers > 0,
        pin_memory=True,
        collate_fn=_collate_keep_meta,
    )
    steps_per_epoch = len(loader)
    print(f"[data] {len(dataset)} pairs, {steps_per_epoch} steps/epoch, "
          f"{cfg.num_epochs} epochs, max_steps={cfg.max_steps} (effective)")

    # --- Training loop ---
    global_step = 0           # optimizer steps (effective)
    micro_step = 0            # forward/backward calls
    final_loss = float("nan")
    grad_assert_done = False

    accum_loss_sum = 0.0      # sum of (already-scaled) micro-step losses across one effective step
    accum_residual_l2 = []
    accum_sigma = None
    accum_t = None

    low_loss_sum = 0.0
    low_loss_count = 0
    # Per-effective-step samples used for the late-10% low-sigma summary.
    eff_step_log: list[tuple[float, float]] = []  # (sigma, loss)

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

            # 1. Sample noise + timestep in low-noise regime
            noise = torch.randn_like(z_real)
            sel = torch.randint(0, len(low_noise_indices), (1,), device=device).item()
            t_idx = low_noise_indices[sel].item()
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

            # 5. Loss (FM in fp32). Scale by accum so that summed grads
            # equal the mean over the effective batch.
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
            accum_sigma = float(sigma.item())
            accum_t = float(t[0].item())
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
                eff_step_log.append((accum_sigma, accum_loss_sum))
                low_loss_sum += accum_loss_sum
                low_loss_count += 1

                peak_mem = torch.cuda.max_memory_allocated() / 1e9
                try:
                    ema_decay_current = float(ema.get_current_decay())
                except Exception:
                    ema_decay_current = float(getattr(ema, "beta", cfg.ema_decay))
                wandb.log({
                    "loss": accum_loss_sum,
                    "grad_norm": grad_norm.item() if torch.is_tensor(grad_norm) else float(grad_norm),
                    "lr": optimizer.param_groups[0]["lr"],
                    "controlnet_residual_norm": float(np.mean(accum_residual_l2)),
                    "timestep": accum_t,
                    "sigma": accum_sigma,
                    "step": global_step,
                    "epoch": epoch,
                    "gpu_mem_gb": peak_mem,
                    "ema_decay_current": ema_decay_current,
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

                if cfg.max_steps is not None and global_step >= cfg.max_steps:
                    print(f"[stop] reached --max_steps={cfg.max_steps} (effective)")
                    done = True
                    break

    # --- Discard any residual partial accumulation ---
    # If max_steps was reached mid-accumulation, the loop exits with partial
    # grads still summed in .grad. Don't let those leak into a stale optimizer
    # step or an EMA update; just zero them out.
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
    low_avg = low_loss_sum / low_loss_count if low_loss_count else float("nan")

    # late-10% mean loss at sigma < 0.1
    if eff_step_log:
        cutoff = int(0.9 * len(eff_step_log))
        late = eff_step_log[cutoff:]
        late_low_sigma = [ls for sg, ls in late if sg < 0.1]
        late_low_sigma_mean = (float(np.mean(late_low_sigma))
                               if late_low_sigma else float("nan"))
    else:
        late_low_sigma_mean = float("nan")
    _RESULTS_STATE.update({
        "final_loss": round(final_loss, 6),
        "gpu_peak_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
        "wall_time": _format_seconds(wall_time_s),
        "wall_time_seconds": round(wall_time_s, 1),
        "global_steps_completed": global_step,
        "micro_steps_completed": micro_step,
        "final_checkpoint": str(final_ema),
        "final_checkpoint_raw": str(final_raw),
        "low_phase_avg_loss": round(low_avg, 6),
        "loss_at_sigma_lt_0p1_late10pct": (
            round(late_low_sigma_mean, 6) if late_low_sigma_mean == late_low_sigma_mean
            else "nan"
        ),
    })
    _write_results()

    # --- End-of-run inference smoke ---
    if not cfg.skip_inference_smoke:
        # Free training-time refs before loading the inference pipeline,
        # which loads BOTH transformer experts.
        try:
            del transformer
            del controlnet
            del ema
            gc.collect()
            torch.cuda.empty_cache()
        except Exception:
            pass

        mp4_paths: list[Path] = []
        try:
            if cfg.warm_start_checkpoint:
                # Plan A — single-CN inference at cn_end=1.0
                mp4 = run_single_cn_smoke(
                    cfg=cfg, ema_checkpoint=final_ema,
                    dataset=dataset, device=device, inf_out_dir=inf_out_dir,
                )
                mp4_paths.append(mp4)
        except Exception as e:
            print(f"[inference-smoke single-CN] failed: {e}")
            traceback.print_exc()
            _RESULTS_STATE["inference_error_singlecn"] = str(e)

        # Free the single-CN pipeline before loading the dual-CN one.
        gc.collect()
        torch.cuda.empty_cache()

        try:
            if cfg.dual_smoke_high_checkpoint:
                mp4 = run_dual_cn_smoke(
                    cfg=cfg,
                    high_checkpoint=cfg.dual_smoke_high_checkpoint,
                    low_checkpoint=final_ema,
                    dataset=dataset, device=device, inf_out_dir=inf_out_dir,
                )
                mp4_paths.append(mp4)
        except Exception as e:
            print(f"[inference-smoke dual-CN] failed: {e}")
            traceback.print_exc()
            _RESULTS_STATE["inference_error_dualcn"] = str(e)

        _RESULTS_STATE["inference_mp4s"] = [str(p) for p in mp4_paths]
        for p in mp4_paths:
            try:
                wandb.log({f"inference_video_{p.stem}": wandb.Video(str(p))},
                          step=global_step)
            except Exception as e:
                print(f"[wandb] could not log video {p.name}: {e}")

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
          f"wall={_RESULTS_STATE['wall_time']} | init={init_mode}")


# ---------------- inference smokes ----------------

def _load_dataset_canny(dataset, cfg):
    """Pull face_idx=0's canny + long prompt from the dataset cache.

    Returns (PIL.Image canny, prompt_text). Mirrors run_inference_beta.py.
    """
    from PIL import Image
    rec = dataset.records[0]
    cache_dir = dataset.cache_dir
    canny_u8 = torch.load(cache_dir / rec["canny_path"], map_location="cpu")
    canny_img = Image.fromarray(canny_u8.permute(1, 2, 0).numpy())
    from training.input_prompts import PROMPTS_BATCH_1, PROMPTS_BATCH_2
    prompt_text = {**PROMPTS_BATCH_1, **PROMPTS_BATCH_2}[rec["slug"]]
    print(f"[inference-smoke] face_idx={rec['face_idx']} slug={rec['slug']}")
    return canny_img, prompt_text


def _build_controlnet_from_checkpoint(controlnet_config_repo: str,
                                      checkpoint_path: Path) -> "WanControlnet":
    from safetensors.torch import load_file
    from wan_controlnet import WanControlnet
    config = WanControlnet.load_config(controlnet_config_repo)
    cn = WanControlnet.from_config(config)
    cast_respecting_fp32_modules(cn, torch.bfloat16)
    sd = load_file(str(checkpoint_path))
    missing, unexpected = cn.load_state_dict(sd, strict=False)
    if missing:
        print(f"[load-ckpt] WARN missing keys: {len(missing)} (e.g. {missing[:2]})")
    if unexpected:
        print(f"[load-ckpt] WARN unexpected keys: {len(unexpected)} (e.g. {unexpected[:2]})")
    cn.eval()
    return cn


def run_single_cn_smoke(cfg, ema_checkpoint: Path, dataset,
                        device: torch.device, inf_out_dir: Path) -> Path:
    """Plan A inference smoke: single-CN pipeline with this run's EMA weights."""
    from diffusers import AutoencoderKLWan
    from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
    from transformers import AutoTokenizer, UMT5EncoderModel
    from wan_transformer import CustomWanTransformer3DModel
    from wan_t2v_controlnet_pipeline import WanTextToVideoControlnetPipeline
    from accelerate.hooks import remove_hook_from_module

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

    canny_img, prompt_text = _load_dataset_canny(dataset, cfg)

    # Re-pin: model_cpu_offload re-attaches an accelerate hook on every __call__.
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
        controlnet_guidance_end=1.0,
        generator=generator,
        output_type="np",
    )
    frames = out.frames[0]
    mp4_path = inf_out_dir / f"{cfg.run_name}_smoke_planA_e1.mp4"
    _save_video(frames, mp4_path, fps=8)
    print(f"[inference-smoke single-CN] wrote {mp4_path}")
    return mp4_path


def run_dual_cn_smoke(cfg, high_checkpoint: str, low_checkpoint: Path,
                      dataset, device: torch.device, inf_out_dir: Path) -> Path:
    """Dual-CN inference smoke: beta-001 frozen high + this run's EMA low."""
    from diffusers import AutoencoderKLWan
    from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
    from transformers import AutoTokenizer, UMT5EncoderModel
    from wan_transformer import CustomWanTransformer3DModel
    from wan_t2v_controlnet_pipeline_dual import (
        WanTextToVideoDualControlnetPipeline,
    )
    from accelerate.hooks import remove_hook_from_module

    print(f"[inference-smoke dual-CN] loading full pipeline ...")
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
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        cfg.base_model_path, subfolder="scheduler",
    )
    boundary_ratio, _ = detect_boundary_ratio(
        cfg.base_model_path, dict(transformer.config),
        override=cfg.boundary_ratio_override,
    )
    controlnet_high = _build_controlnet_from_checkpoint(
        cfg.controlnet_config_repo, Path(high_checkpoint),
    )
    controlnet_low = _build_controlnet_from_checkpoint(
        cfg.controlnet_config_repo, low_checkpoint,
    )

    pipe = WanTextToVideoDualControlnetPipeline(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        transformer=transformer,
        transformer_2=transformer_2,
        vae=vae,
        controlnet_high=controlnet_high,
        controlnet_low=controlnet_low,
        scheduler=scheduler,
        boundary_ratio=boundary_ratio,
    )
    pipe.enable_model_cpu_offload()
    # Pin both controlnets to GPU; same offload→Conv3D mismatch as the single
    # pipeline, so strip the hook on each controlnet.
    for cn in (pipe.controlnet_high, pipe.controlnet_low):
        remove_hook_from_module(cn, recurse=True)
        cn.to("cuda")

    canny_img, prompt_text = _load_dataset_canny(dataset, cfg)

    for cn in (pipe.controlnet_high, pipe.controlnet_low):
        remove_hook_from_module(cn, recurse=True)
        cn.to("cuda")
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
        controlnet_guidance_end=1.0,
        generator=generator,
        output_type="np",
    )
    frames = out.frames[0]
    suffix = "dualCN" if cfg.warm_start_checkpoint else "planB"
    mp4_path = inf_out_dir / f"{cfg.run_name}_smoke_{suffix}_e1.mp4"
    _save_video(frames, mp4_path, fps=8)
    print(f"[inference-smoke dual-CN] wrote {mp4_path}")
    return mp4_path


def _save_video(frames_np: np.ndarray, path: Path, fps: int = 8) -> None:
    import imageio
    arr = (np.clip(frames_np, 0.0, 1.0) * 255).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(path), list(arr), fps=fps, codec="libx264")


if __name__ == "__main__":
    main()

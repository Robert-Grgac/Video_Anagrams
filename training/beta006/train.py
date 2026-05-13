"""BETA6 training: cold-start WanControlnet on the LOW-noise expert,
A40-pure-accumulation variant.

Sibling of ``train_beta5.py`` (which trains the high-noise expert with
batching on Blackwell). Both runs use the same supervisor recipe:
``eff_batch=32``, EMA(0.999) with ``update_after_step=100``,
AdamW8bit(lr=1e-4, wd=1e-4), grad_clip=1.0, cold-init (``zero_module``).
The two scripts together replace beta-001/beta-004 as the high+low CN pair
trained under one consistent recipe.

Differences from train_beta3.py:

* No ``--warm_start_checkpoint`` flag (cold-only by design — beta-003 lives
  in the legacy script).
* ``--micro_batch_size`` flag for parity with train_beta5; on A40 we keep
  ``micro_batch=1`` and lift eff_batch via accumulation alone.
* Logs ``loss_ema`` (EMA over effective-step losses) and ``sigma_std``.
* Hardware-aware ``set_attention_backend('native')`` guard for Blackwell;
  no-op on A40 (compute_cap < 12).

End-of-run inference smoke writes a dual-CN video pairing this run's EMA
(low-CN) with whatever ``--dual_smoke_high_checkpoint`` points at — the
canonical companion is the new beta-005 EMA, not the legacy beta-001.
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
from training.utils import (
    _maybe_force_native_attention,
    _build_controlnet_from_checkpoint,
    _save_state_dict,
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
    p.add_argument("--controlnet_config_repo", type=str, required=True)
    p.add_argument("--dual_smoke_high_checkpoint", type=str, default=None,
                   help="Optional .safetensors used as the high-CN in the end-of-run "
                        "dual-CN smoke. If unset, the dual-CN smoke is skipped.")
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
    p.add_argument("--micro_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=32)
    p.add_argument("--ema_decay", type=float, default=0.999)
    p.add_argument("--ema_update_after_step", type=int, default=100)
    p.add_argument("--loss_ema_window", type=int, default=20)

    # Architecture overrides — defaults preserve baseline beta-006 behavior.
    p.add_argument("--num_cn_layers", type=int, default=None,
                   help="Override the controlnet config's num_layers (=6 by default in the "
                        "HED config). When set, the controlnet is built with this many blocks. "
                        "Constraint: stride*(num_layers-1) <= 39 to keep all residuals usable.")
    p.add_argument("--controlnet_stride", type=int, default=3,
                   help="Stride at which CN residuals inject into the 40-block transformer. "
                        "Injection rule: residual i lands at transformer block i*stride.")

    # Self-distillation against the EMA model (Mean-Teacher-style consistency loss).
    # Disabled by default; enabling adds an extra no_grad forward through the EMA CN +
    # transformer per micro-step (~+30-40% wall-clock).
    p.add_argument("--use_self_distillation", action="store_true",
                   help="Add lambda_consistency * MSE(v_pred_live, v_pred_ema) to the FM loss "
                        "before backward. Requires meaningful EMA divergence — see ema_decay / "
                        "ema_update_after_step.")
    p.add_argument("--lambda_consistency", type=float, default=0.5,
                   help="Weight on the consistency term when --use_self_distillation is set.")

    p.add_argument("--num_train_timesteps_for_sampling", type=int, default=1000)
    p.add_argument("--boundary_ratio_override", type=float, default=None)
    p.add_argument("--checkpoint_every", type=int, default=50)
    p.add_argument("--memory_tripwire_gb", type=float, default=43.0)

    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip_inference_smoke", action="store_true")
    return p.parse_args()


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
        _RESULTS_PATH = Path("training_cards") / "beta006" / f"{cfg.run_name}_results.json"

    init_mode = "cold"
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
        "num_cn_layers_override": cfg.num_cn_layers,
        "controlnet_stride": cfg.controlnet_stride,
        "use_self_distillation": cfg.use_self_distillation,
        "lambda_consistency": cfg.lambda_consistency if cfg.use_self_distillation else None,
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
        raise RuntimeError("BETA6 training requires CUDA; got CPU.")
    try:
        major, minor = torch.cuda.get_device_capability(0)
        gpu_name = torch.cuda.get_device_name(0)
        print(f"[gpu] {gpu_name} (compute_cap={major}.{minor})")
        _RESULTS_STATE["gpu_name"] = gpu_name
        _RESULTS_STATE["gpu_compute_cap"] = f"{major}.{minor}"
    except Exception:
        pass

    from wan_controlnet import WanControlnet
    from wan_transformer import CustomWanTransformer3DModel
    from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

    print(f"[load] low-noise transformer (transformer_2) from {cfg.base_model_path} ...")
    transformer = CustomWanTransformer3DModel.from_pretrained(
        cfg.base_model_path, subfolder="transformer_2", torch_dtype=torch.bfloat16,
    )
    transformer.requires_grad_(False).eval().to(device)
    transformer.enable_gradient_checkpointing()
    _maybe_force_native_attention(transformer, "transformer_2")

    boundary_ratio, boundary_src = detect_boundary_ratio(
        cfg.base_model_path, dict(transformer.config),
        override=cfg.boundary_ratio_override,
    )

    print(f"[load] controlnet config from {cfg.controlnet_config_repo} (architecture only) ...")
    config = WanControlnet.load_config(cfg.controlnet_config_repo)
    if cfg.num_cn_layers is not None:
        old_L = config.get("num_layers", "?")
        config["num_layers"] = cfg.num_cn_layers
        # Sanity: with a 40-block transformer, residual i lands at block i*stride;
        # any residual whose target block is >= 40 is silently discarded by
        # wan_transformer.py:90. Warn (don't error — user might have a reason).
        max_used_block = (cfg.num_cn_layers - 1) * cfg.controlnet_stride
        if max_used_block >= 40:
            wasted = sum(
                1 for i in range(cfg.num_cn_layers)
                if i * cfg.controlnet_stride >= 40
            )
            print(f"[arch] WARN num_layers={cfg.num_cn_layers} × stride={cfg.controlnet_stride} "
                  f"would waste {wasted} CN layer(s) (target block >= 40). "
                  f"Compute is paid; output is dropped.", file=sys.stderr)
        print(f"[arch] num_layers override: {old_L} -> {cfg.num_cn_layers} "
              f"(stride={cfg.controlnet_stride}, max target block={max_used_block})")
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

    global_step = 0
    micro_step = 0
    final_loss = float("nan")
    grad_assert_done = False

    accum_loss_sum = 0.0
    accum_loss_fm_sum = 0.0
    accum_loss_consistency_sum = 0.0
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
        for step, batch in enumerate(loader):
            canny = batch["canny"].to(device, non_blocking=True)
            z_real = batch["latent"].to(device, non_blocking=True)
            prompt_embeds = batch["prompt_embeds"].to(device, non_blocking=True)
            B = z_real.shape[0]

            sel = torch.randint(0, len(low_noise_indices), (B,), device=device)
            t_idx = low_noise_indices[sel]
            sigma = sigmas[t_idx].to(z_real.dtype)
            t = timesteps_full[t_idx]
            sigma_b = sigma.view(B, 1, 1, 1, 1)

            noise = torch.randn_like(z_real)
            z_t = (1.0 - sigma_b) * z_real + sigma_b * noise
            v_target = (noise - z_real).float()

            # Self-distillation teacher forward: compute v_pred_ema BEFORE the
            # live forward so the EMA path's intermediates can be released
            # before live activations start piling up. no_grad → no checkpoint
            # recompute, no stored activations.
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
                        cn_for_tx_ema = [
                            s.to(dtype=transformer.dtype) for s in cn_states_ema
                        ]
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
                controlnet_states_for_tx = [
                    s.to(dtype=transformer.dtype) for s in controlnet_states
                ]
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

            loss_fm = F.mse_loss(v_pred.float(), v_target)
            if cfg.use_self_distillation and v_pred_ema is not None:
                loss_consistency = F.mse_loss(v_pred.float(), v_pred_ema)
                loss = loss_fm + cfg.lambda_consistency * loss_consistency
            else:
                loss_consistency = torch.zeros((), device=v_pred.device)
                loss = loss_fm
            loss_scaled = loss / cfg.gradient_accumulation_steps
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
            accum_loss_fm_sum += float(loss_fm.item()) / cfg.gradient_accumulation_steps
            accum_loss_consistency_sum += (
                float(loss_consistency.item()) / cfg.gradient_accumulation_steps
            )
            accum_residual_l2.append(mean_residual_l2(controlnet_states))
            accum_sigmas.extend(sigma.detach().float().cpu().tolist())
            accum_t_last = float(t[-1].item())
            micro_step += 1

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
                log_payload = {
                    "loss": accum_loss_sum,
                    "loss_fm": accum_loss_fm_sum,
                    "loss_consistency": accum_loss_consistency_sum,
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
                }
                wandb.log(log_payload, step=global_step)

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
                accum_loss_fm_sum = 0.0
                accum_loss_consistency_sum = 0.0
                accum_residual_l2 = []
                accum_sigmas = []

                if cfg.max_steps is not None and global_step >= cfg.max_steps:
                    print(f"[stop] reached --max_steps={cfg.max_steps} (effective)")
                    done = True
                    break

    if micro_step % cfg.gradient_accumulation_steps != 0:
        print(f"[stop] discarding {micro_step % cfg.gradient_accumulation_steps} "
              "micro-step(s) of partial accumulation at exit")
        optimizer.zero_grad(set_to_none=True)

    final_ema = output_dir / f"{cfg.run_name}_final.safetensors"
    final_raw = output_dir / f"{cfg.run_name}_final_raw.safetensors"
    _save_state_dict(ema.ema_model.state_dict(), final_ema)
    _save_safetensors(controlnet, final_raw)
    print(f"[ckpt] {final_ema}  (EMA, canonical)")
    print(f"[ckpt] {final_raw}  (raw, debug)")

    wall_time_s = time.perf_counter() - t_train_start
    low_avg = float(np.mean([ls for _sg, ls in eff_step_log])) if eff_step_log else float("nan")
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
        "loss_ema_final": (round(float(loss_ema_value), 6)
                           if loss_ema_value is not None else "nan"),
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

    if not cfg.skip_inference_smoke:
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
    loss_ema_str = f"{loss_ema_value:.4f}" if loss_ema_value is not None else "nan"
    print(f"[done] {cfg.run_name} | loss={final_loss:.4f} | "
          f"loss_ema={loss_ema_str} | wall={_RESULTS_STATE['wall_time']} | init={init_mode}")


def run_dual_cn_smoke(cfg, high_checkpoint: str, low_checkpoint: Path,
                      dataset, device: torch.device, inf_out_dir: Path) -> Path:
    """Dual-CN inference smoke: arbitrary high-CN + this run's EMA low-CN."""
    from diffusers import AutoencoderKLWan
    from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
    from transformers import AutoTokenizer, UMT5EncoderModel
    from wan_transformer import CustomWanTransformer3DModel
    from wan_t2v_controlnet_pipeline_dual import WanTextToVideoDualControlnetPipeline
    from accelerate.hooks import remove_hook_from_module
    from PIL import Image

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
    _maybe_force_native_attention(transformer, "transformer (smoke)")
    _maybe_force_native_attention(transformer_2, "transformer_2 (smoke)")

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
    _maybe_force_native_attention(controlnet_high, "controlnet_high (smoke)")
    _maybe_force_native_attention(controlnet_low, "controlnet_low (smoke)")

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
    for cn in (pipe.controlnet_high, pipe.controlnet_low):
        remove_hook_from_module(cn, recurse=True)
        cn.to("cuda")

    rec = dataset.records[0]
    cache_dir = dataset.cache_dir
    canny_u8 = torch.load(cache_dir / rec["canny_path"], map_location="cpu")
    canny_img = Image.fromarray(canny_u8.permute(1, 2, 0).numpy())
    from training.input_prompts import PROMPTS_BATCH_1, PROMPTS_BATCH_2
    prompt_text = {**PROMPTS_BATCH_1, **PROMPTS_BATCH_2}[rec["slug"]]
    print(f"[inference-smoke dual-CN] face_idx={rec['face_idx']} slug={rec['slug']}")

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
    mp4_path = inf_out_dir / f"{cfg.run_name}_smoke_dualCN_e1.mp4"
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

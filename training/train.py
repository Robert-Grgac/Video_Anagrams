"""ControlNet training script.

Trains the ``WanControlnet`` (cold start from the HED config, weights
zero-init in the output projections) against the high-noise expert of
Wan 2.2 T2V-A14B. Reads the recipe from ``training.config.TrainConfig``;
only paths and the optional wandb identity come from the CLI.

Outputs the final EMA-smoothed weights as ``<output_dir>/controlnet.safetensors``.
No intermediate checkpoints, no in-training inference, no eval.
"""
from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from training.config import TrainConfig
from training.dataset import PairDataset, CONTROL_SUBDIR
from utils.utils import (
    cast_respecting_fp32_modules,
    detect_boundary_ratio,
    mean_residual_l2,
    _collate_keep_meta,
    _save_state_dict,
    _format_seconds,
    _maybe_force_native_attention,
)

# T5 padding length expected by the Wan transformer's cross-attention. MUST
# match precompute_training.py; mismatched lengths silently corrupt the
# cross-attention output at inference (no shape error, just noise).
WAN_T5_MAX_SEQ_LEN = 226

# The silhouette mask doubles as the D1 spatial loss weight; identical subdir.
FACE_MASK_SUBDIR = CONTROL_SUBDIR


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--cache_dir", type=str,
                   default=os.environ.get("WAN_BETA_CACHE", "./cache/training"))
    p.add_argument("--base_model_path", type=str,
                   default=os.environ.get("WAN_MODEL", "./models/wan2.2"))
    p.add_argument("--controlnet_config_repo", type=str,
                   default=os.environ.get("HED_CONFIG", "./models/hed_config"))
    p.add_argument("--output_dir", type=str,
                   default=os.environ.get("WAN_BETA_CKPT", "./models/controlnet"))
    p.add_argument("--wandb_project", type=str, default=None)
    p.add_argument("--wandb_run_name", type=str, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = TrainConfig()

    if (args.wandb_project is None) != (args.wandb_run_name is None):
        raise SystemExit(
            "--wandb_project and --wandb_run_name must be passed together "
            "(or neither — wandb is off by default)."
        )
    wandb_enabled = args.wandb_project is not None

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    final_ckpt_path = output_dir / "controlnet.safetensors"

    if wandb_enabled:
        import wandb
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config={**vars(cfg), **vars(args)},
        )
        print(f"[wandb] enabled: project={args.wandb_project} "
              f"name={args.wandb_run_name} url={wandb_run.get_url()}")
    else:
        wandb = None  # type: ignore[assignment]
        print("[wandb] disabled (no --wandb_project / --wandb_run_name)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Training requires CUDA; got CPU.")
    major, minor = torch.cuda.get_device_capability(0)
    print(f"[gpu] {torch.cuda.get_device_name(0)} (compute_cap={major}.{minor})")

    from accelerate import Accelerator
    accelerator = Accelerator(gradient_accumulation_steps=cfg.gradient_accumulation_steps)
    print(f"[accel] gradient_accumulation_steps={cfg.gradient_accumulation_steps}")

    # ---- Models --------------------------------------------------------------
    from pipeline.wan_controlnet import WanControlnet
    from pipeline.wan_transformer import CustomWanTransformer3DModel
    from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

    print(f"[load] high-noise transformer (transformer) from {args.base_model_path} ...")
    transformer = CustomWanTransformer3DModel.from_pretrained(
        args.base_model_path, subfolder="transformer", torch_dtype=torch.bfloat16,
    )
    transformer.requires_grad_(False).eval().to(device)
    transformer.enable_gradient_checkpointing()
    _maybe_force_native_attention(transformer, "transformer")

    boundary_ratio, boundary_src = detect_boundary_ratio(
        args.base_model_path, dict(transformer.config),
        override=cfg.boundary_ratio_override,
    )

    print(f"[load] controlnet config from {args.controlnet_config_repo} (architecture only) ...")
    config = WanControlnet.load_config(args.controlnet_config_repo)
    if cfg.num_cn_layers is not None:
        old_L = config.get("num_layers", "?")
        config["num_layers"] = cfg.num_cn_layers
        print(f"[arch] num_layers override: {old_L} -> {cfg.num_cn_layers}")
    controlnet = WanControlnet.from_config(config)
    cast_respecting_fp32_modules(controlnet, torch.bfloat16)
    print("[cold-start] using fresh from_config weights (output projections zeroed)")
    controlnet.enable_gradient_checkpointing()
    controlnet.train().to(device)
    _maybe_force_native_attention(controlnet, "controlnet")
    print(f"[controlnet] trainable params: "
          f"{sum(p.numel() for p in controlnet.parameters()):,}")

    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.base_model_path, subfolder="scheduler",
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
          f"high={len(high_noise_indices)} of {len(sigmas)} timesteps")

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

    # ---- Dataset (full 10 000 face×prompt pairs) -----------------------------
    full_dataset = PairDataset(args.cache_dir, num_frames=cfg.num_frames)
    total_n = len(full_dataset)
    if total_n != 10000:
        raise RuntimeError(
            f"Expected 10000 records (100 faces × 100 prompts); got {total_n}. "
            "Re-run precompute_training.py against the full input set."
        )
    print(f"[data] train: {total_n} pairs (1 epoch = full coverage)")

    # ---- D1 spatial face-weight masks ----------------------------------------
    # Avg-pool the 8-channel-downsampled silhouette to latent resolution; pooled
    # values in [0,1] avoid the hard-edge aliasing of a thresholded downsample.
    mask_dir = Path(args.cache_dir) / FACE_MASK_SUBDIR
    if not mask_dir.exists():
        raise FileNotFoundError(
            f"Face-mask dir {mask_dir} does not exist. "
            "Run precompute_training.py first."
        )
    spatial_factor = 8  # Wan VAE spatial downsample
    h_lat = cfg.height // spatial_factor
    w_lat = cfg.width // spatial_factor
    unique_face_idxs = sorted({r["face_idx"] for r in full_dataset.records})
    face_masks_latent: dict[int, torch.Tensor] = {}
    for fi in unique_face_idxs:
        mask_path = mask_dir / f"face_{fi}.pt"
        if not mask_path.exists():
            raise FileNotFoundError(f"D1 mask missing for face_idx={fi}: {mask_path}")
        raw = torch.load(mask_path, map_location="cpu", weights_only=True)
        m_bin = (raw[0] > 0).float()
        m_lat = F.avg_pool2d(
            m_bin.unsqueeze(0).unsqueeze(0),
            kernel_size=spatial_factor, stride=spatial_factor,
        ).squeeze(0).squeeze(0)
        face_masks_latent[int(fi)] = m_lat.contiguous()
    coverage = float(torch.stack(list(face_masks_latent.values())).mean().item())
    print(f"[d1] cached {len(face_masks_latent)} face masks at {h_lat}x{w_lat}; "
          f"mean coverage={coverage:.3f}; alpha={cfg.face_weight_alpha}")

    train_loader = DataLoader(
        full_dataset, batch_size=cfg.micro_batch_size, shuffle=True,
        num_workers=cfg.num_workers,
        persistent_workers=cfg.num_workers > 0,
        pin_memory=True, drop_last=True,
        collate_fn=_collate_keep_meta,
    )
    micro_steps_per_epoch = len(train_loader)
    eff_per_epoch = micro_steps_per_epoch // cfg.gradient_accumulation_steps
    max_steps = eff_per_epoch * cfg.num_epochs
    print(f"[data] {micro_steps_per_epoch} micro/epoch, {eff_per_epoch} "
          f"effective/epoch, total max_steps={max_steps}")

    # ---- Training loop -------------------------------------------------------
    global_step = 0
    micro_step = 0
    final_loss = float("nan")
    grad_assert_done = False

    accum_losses: list[torch.Tensor] = []
    accum_losses_fm: list[torch.Tensor] = []
    accum_losses_consist: list[torch.Tensor] = []
    accum_residual_l2: list[float] = []
    accum_sigmas: list[float] = []

    loss_ema_value: Optional[float] = None
    loss_ema_alpha = 2.0 / (cfg.loss_ema_window + 1)

    optimizer.zero_grad(set_to_none=True)
    t_train_start = time.perf_counter()

    pbar = tqdm(total=max_steps, desc="train", unit="step", dynamic_ncols=True)
    for epoch in range(cfg.num_epochs):
        for step, batch in enumerate(train_loader):
            with accelerator.accumulate(controlnet):
                control = batch["control"].to(device, non_blocking=True)
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

                v_pred_ema = None
                if cfg.use_self_distillation:
                    with torch.no_grad():
                        cn_states_ema = ema.ema_model(
                            hidden_states=z_t,
                            timestep=t,
                            encoder_hidden_states=prompt_embeds,
                            controlnet_states=control,
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
                    controlnet_states=control,
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

                # D1 spatial weighting on the FM term only.
                face_idxs_b = batch["face_idx"]
                masks_b = torch.stack(
                    [face_masks_latent[int(fi)] for fi in face_idxs_b], dim=0,
                ).to(v_pred.device, dtype=torch.float32)
                masks_b = masks_b.unsqueeze(1).unsqueeze(1)
                weight_map = 1.0 + cfg.face_weight_alpha * masks_b
                diff2 = (v_pred.float() - v_target) ** 2
                loss_fm = (weight_map * diff2).mean()
                if cfg.use_self_distillation and v_pred_ema is not None:
                    loss_consistency = F.mse_loss(v_pred.float(), v_pred_ema)
                    loss = loss_fm + cfg.lambda_consistency * loss_consistency
                else:
                    loss_consistency = torch.zeros((), device=v_pred.device)
                    loss = loss_fm

                accelerator.backward(loss)

                accum_losses.append(loss.detach())
                accum_losses_fm.append(loss_fm.detach())
                accum_losses_consist.append(loss_consistency.detach())
                accum_residual_l2.append(mean_residual_l2(controlnet_states))
                accum_sigmas.extend(sigma.detach().float().cpu().tolist())
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

            if accelerator.sync_gradients:
                torch.nn.utils.clip_grad_norm_(controlnet.parameters(), cfg.grad_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                ema.update()
                global_step += 1

                accum_loss_mean = float(torch.stack(accum_losses).mean().item())
                accum_loss_fm_mean = float(torch.stack(accum_losses_fm).mean().item())
                accum_loss_consist_mean = float(torch.stack(accum_losses_consist).mean().item())
                final_loss = accum_loss_mean
                sigma_mean = float(np.mean(accum_sigmas))

                if loss_ema_value is None:
                    loss_ema_value = accum_loss_mean
                else:
                    loss_ema_value = (loss_ema_alpha * accum_loss_mean
                                     + (1.0 - loss_ema_alpha) * loss_ema_value)

                if wandb_enabled:
                    wandb.log({
                        "sigma": sigma_mean,
                        "loss_fm": accum_loss_fm_mean,
                        "loss_consistency": accum_loss_consist_mean,
                        "loss": accum_loss_mean,
                        "loss_ema": loss_ema_value,
                        "controlnet_residual_norm": float(np.mean(accum_residual_l2)),
                    }, step=global_step)

                pbar.update(1)
                pbar.set_postfix(
                    loss=f"{accum_loss_mean:.4f}",
                    loss_ema=f"{loss_ema_value:.4f}",
                )

                accum_losses.clear()
                accum_losses_fm.clear()
                accum_losses_consist.clear()
                accum_residual_l2.clear()
                accum_sigmas.clear()

                peak_mem = torch.cuda.max_memory_allocated() / 1e9
                if peak_mem > cfg.memory_tripwire_gb:
                    raise RuntimeError(
                        f"GPU memory {peak_mem:.2f}GB exceeded tripwire "
                        f"{cfg.memory_tripwire_gb}GB at step {global_step}."
                    )

    pbar.close()

    if micro_step % cfg.gradient_accumulation_steps != 0:
        leftover = micro_step % cfg.gradient_accumulation_steps
        print(f"[stop] discarding {leftover} micro-step(s) of partial accumulation at exit")
        optimizer.zero_grad(set_to_none=True)

    _save_state_dict(ema.ema_model.state_dict(), final_ckpt_path)
    print(f"[ckpt] {final_ckpt_path}  (EMA)")

    wall_time_s = time.perf_counter() - t_train_start
    loss_ema_str = f"{loss_ema_value:.4f}" if loss_ema_value is not None else "nan"
    print(f"[done] loss={final_loss:.4f} | loss_ema={loss_ema_str} | "
          f"wall={_format_seconds(wall_time_s)} | steps={global_step}")

    if wandb_enabled:
        wandb.finish()

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

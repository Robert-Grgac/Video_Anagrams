"""Smoke test for ``train_beta6.py``.

A40 / pure-accumulation sibling of ``smoke_test_beta5.py``. Loads the
LOW-noise transformer (``transformer_2``) and a cold-init controlnet, then
runs a few effective steps with ``micro_batch=1`` and the configured
accumulation. Same asserts as smoke_test_beta5; only the expert and the
sigma regime differ.

Default eff_steps×accum keeps the smoke under ~2 minutes on A40 while still
covering: attention dispatch, grad-flow, EMA cadence, and memory tripwire
on the chosen accumulation depth.
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from training.dataset_beta import BetaPairDataset
from training.utils import (
    cast_respecting_fp32_modules,
    detect_boundary_ratio,
    mean_residual_l2,
    _collate_keep_meta,
)
from training.utils import _maybe_force_native_attention


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--cache_dir", type=str, required=True)
    p.add_argument("--base_model_path", type=str, required=True)
    p.add_argument("--controlnet_config_repo", type=str, required=True)
    p.add_argument("--num_frames", type=int, default=9)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=512)

    p.add_argument("--effective_steps", type=int, default=2)
    p.add_argument("--micro_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8,
                   help="Smaller than the real run (32) to keep the smoke fast; the loop "
                        "logic is identical regardless of this value.")
    p.add_argument("--ema_decay", type=float, default=0.999)
    p.add_argument("--ema_update_after_step", type=int, default=100)

    # Architecture overrides — defaults match HED config + train_beta6 hardcode.
    p.add_argument("--num_cn_layers", type=int, default=None)
    p.add_argument("--controlnet_stride", type=int, default=3)

    # Self-distillation knobs (mirrors train_beta6).
    p.add_argument("--use_self_distillation", action="store_true")
    p.add_argument("--lambda_consistency", type=float, default=0.5)

    p.add_argument("--mem_tripwire_gb", type=float, default=43.0)
    p.add_argument("--residual_l2_ceiling", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def assert_finite_optimizer_state(optimizer, where: str) -> None:
    for p, st in optimizer.state.items():
        for k, v in st.items():
            if torch.is_tensor(v) and torch.is_floating_point(v):
                assert torch.isfinite(v).all(), \
                    f"non-finite optimizer state '{k}' {where}"


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)

    cache_dir = Path(args.cache_dir)
    if not (cache_dir / "manifest.json").exists():
        print(f"[smoke] FAIL: no manifest.json in {cache_dir}", file=sys.stderr)
        return 2

    from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
    from wan_transformer import CustomWanTransformer3DModel
    from wan_controlnet import WanControlnet
    import bitsandbytes as bnb
    from ema_pytorch import EMA

    device = torch.device("cuda")
    try:
        major, minor = torch.cuda.get_device_capability(0)
        print(f"[smoke] GPU={torch.cuda.get_device_name(0)} compute_cap={major}.{minor}")
    except Exception:
        major = 0

    print(f"[smoke] loading low-noise transformer (transformer_2) ...")
    transformer = CustomWanTransformer3DModel.from_pretrained(
        args.base_model_path, subfolder="transformer_2", torch_dtype=torch.bfloat16,
    )
    transformer.requires_grad_(False).eval().to(device)
    transformer.enable_gradient_checkpointing()
    _maybe_force_native_attention(transformer, "transformer_2")

    boundary_ratio, _src = detect_boundary_ratio(
        args.base_model_path, dict(transformer.config),
    )
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.base_model_path, subfolder="scheduler",
    )
    scheduler.set_timesteps(1000, device=device)
    sigmas = scheduler.sigmas[:-1].to(device)
    timesteps_full = scheduler.timesteps.to(device)

    low_noise_indices = torch.where(sigmas < boundary_ratio)[0]
    if low_noise_indices.numel() == 0:
        low_noise_indices = torch.arange(len(sigmas) // 2, len(sigmas), device=device)
    print(f"[smoke] boundary={boundary_ratio} low={len(low_noise_indices)} of {len(sigmas)}")

    print(f"[smoke] building cold-init controlnet "
          f"(num_layers={args.num_cn_layers or 'default'}, stride={args.controlnet_stride}) ...")
    config = WanControlnet.load_config(args.controlnet_config_repo)
    if args.num_cn_layers is not None:
        config["num_layers"] = args.num_cn_layers
    cn = WanControlnet.from_config(config)
    cast_respecting_fp32_modules(cn, torch.bfloat16)
    fp32_params = [n for n, p in cn.named_parameters() if p.dtype == torch.float32]
    assert any("norm" in n or "time_embedder" in n or "scale_shift" in n
               for n in fp32_params), "Expected fp32 norm/time_embedder/scale_shift"
    assert any(p.dtype == torch.bfloat16 for p in cn.parameters()), \
        "Expected most controlnet params to be bf16"
    cn.enable_gradient_checkpointing()
    cn.train().to(device)
    _maybe_force_native_attention(cn, "controlnet")

    optimizer = bnb.optim.AdamW8bit(cn.parameters(), lr=1e-4, weight_decay=1e-4)
    ema = EMA(
        cn,
        beta=args.ema_decay,
        update_after_step=args.ema_update_after_step,
        update_every=1,
    )
    ema.to(device)

    dataset = BetaPairDataset(cache_dir, num_frames=args.num_frames)
    loader = DataLoader(
        dataset, batch_size=args.micro_batch_size, shuffle=True, num_workers=0,
        collate_fn=_collate_keep_meta, drop_last=True,
    )
    loader_iter = iter(loader)

    target_eff = args.effective_steps
    target_micro = target_eff * args.gradient_accumulation_steps
    print(f"[smoke] running {target_eff} eff-steps × accum={args.gradient_accumulation_steps} "
          f"× micro_batch={args.micro_batch_size} = {target_eff * args.gradient_accumulation_steps * args.micro_batch_size} "
          f"sample-views (eff_batch={args.micro_batch_size * args.gradient_accumulation_steps})")

    optimizer.zero_grad(set_to_none=True)
    grad_assert_done = False
    micro_step = 0
    global_step = 0
    eff_step_times: list[float] = []
    eff_t0 = None

    while global_step < target_eff:
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            batch = next(loader_iter)

        if micro_step % args.gradient_accumulation_steps == 0:
            torch.cuda.synchronize()
            eff_t0 = time.perf_counter()

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

        # SD teacher forward (no_grad) before live forward, mirroring train_beta6.
        v_pred_ema = None
        if args.use_self_distillation:
            with torch.no_grad():
                cn_states_ema = ema.ema_model(
                    hidden_states=z_t, timestep=t,
                    encoder_hidden_states=prompt_embeds,
                    controlnet_states=canny, return_dict=False,
                )[0]
                cn_for_tx_ema = [s.to(dtype=transformer.dtype) for s in cn_states_ema] \
                    if isinstance(cn_states_ema, (list, tuple)) else cn_states_ema.to(dtype=transformer.dtype)
                v_pred_ema = transformer(
                    hidden_states=z_t, timestep=t,
                    encoder_hidden_states=prompt_embeds,
                    controlnet_states=cn_for_tx_ema,
                    controlnet_weight=1.0, controlnet_stride=args.controlnet_stride,
                    return_dict=False,
                )[0].float()
                del cn_states_ema, cn_for_tx_ema

        cn_states = cn(
            hidden_states=z_t, timestep=t,
            encoder_hidden_states=prompt_embeds,
            controlnet_states=canny, return_dict=False,
        )[0]
        cn_for_tx = [s.to(dtype=transformer.dtype) for s in cn_states] \
            if isinstance(cn_states, (list, tuple)) else cn_states.to(dtype=transformer.dtype)
        v_pred = transformer(
            hidden_states=z_t, timestep=t,
            encoder_hidden_states=prompt_embeds,
            controlnet_states=cn_for_tx,
            controlnet_weight=1.0, controlnet_stride=args.controlnet_stride,
            return_dict=False,
        )[0]

        loss_fm = F.mse_loss(v_pred.float(), v_target)
        if args.use_self_distillation and v_pred_ema is not None:
            loss_consistency = F.mse_loss(v_pred.float(), v_pred_ema)
            loss = loss_fm + args.lambda_consistency * loss_consistency
        else:
            loss = loss_fm
        loss_scaled = loss / args.gradient_accumulation_steps
        loss_scaled.backward()

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
                for p in cn.parameters()
            )
            assert cn_with_grad, "No controlnet param has nonzero grad after first micro-step"
            grad_assert_done = True
            print("[smoke] grad-flow check passed at micro-step 1")

        peak_mem = torch.cuda.max_memory_allocated() / 1e9
        residual_l2 = mean_residual_l2(cn_states)
        assert torch.isfinite(loss).item(), f"NaN/Inf loss at micro-step {micro_step + 1}"
        assert peak_mem < args.mem_tripwire_gb, (
            f"GPU peak {peak_mem:.2f}GB >= ceiling {args.mem_tripwire_gb}GB"
        )
        assert 0.0 <= residual_l2 < args.residual_l2_ceiling, (
            f"cold residual L2 {residual_l2:.2e} out of [0, {args.residual_l2_ceiling})"
        )

        micro_step += 1

        if micro_step % args.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(cn.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            ema.update()
            global_step += 1

            torch.cuda.synchronize()
            eff_step_times.append(time.perf_counter() - eff_t0)

            assert_finite_optimizer_state(
                optimizer, where=f"after eff-step {global_step}"
            )
            ema_step_count = int(ema.step.item())
            assert ema_step_count == global_step, (
                f"EMA step counter drift: ema.step={ema_step_count} != "
                f"global_step={global_step} (grad-accum miswired?)"
            )
            print(f"[smoke] eff-step {global_step:02d}/{target_eff} "
                  f"(micro={micro_step:02d}/{target_micro}) | "
                  f"loss={loss.item():.4f} | dt={eff_step_times[-1]:.2f}s | "
                  f"peak_mem={peak_mem:.2f}GB | residual_l2={residual_l2:.2e}")

    print("")
    if eff_step_times:
        print(f"[smoke] eff-steps={len(eff_step_times)} "
              f"mean={statistics.mean(eff_step_times):.2f}s "
              f"median={statistics.median(eff_step_times):.2f}s")
    print("[smoke] OK")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except AssertionError as e:
        print(f"[smoke] FAIL: {e}", file=sys.stderr)
        sys.exit(1)

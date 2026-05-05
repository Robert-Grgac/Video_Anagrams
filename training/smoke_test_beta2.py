"""Smoke test for the BETA2 phase-alternating training loop.

Reuses the existing precompute cache (no re-precompute). Runs cycle_steps=5,
max_steps=15 so the loop exercises at least two swaps (5 high -> 5 low ->
5 high). Asserts:

* loss is finite at every step;
* mean controlnet residual L2 is finite and inside (0, 50] from step 2 onward
  (output projections wake up after step 1);
* GPU peak < 40 GB throughout;
* every swap completes without exception and yields a CustomWanTransformer3DModel;
* optimizer state remains finite after every swap.

Prints per-phase per-step time and exits non-zero on any assertion failure.
No wandb, no checkpointing, no end-of-run inference.
"""
from __future__ import annotations

import argparse
import gc
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from training.dataset_beta import BetaPairDataset
from training.train_beta import (
    cast_respecting_fp32_modules,
    detect_boundary_ratio,
    mean_residual_l2,
    _collate_keep_meta,
)
from training.train_beta2 import (
    load_expert,
    free_then_load_expert,
    next_subfolder_for,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--cache_dir", type=str, required=True,
                   help="Existing precompute cache (manifest.json + canny/ + latent/ + prompt/).")
    p.add_argument("--base_model_path", type=str, required=True)
    p.add_argument("--controlnet_config_repo", type=str, required=True)
    p.add_argument("--num_frames", type=int, default=9)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=512)

    # Smoke-specific knobs (defaults match the plan: 2 swaps in 15 steps).
    p.add_argument("--cycle_steps", type=int, default=5)
    p.add_argument("--max_steps", type=int, default=15)
    p.add_argument("--start_phase", type=str, default="high",
                   choices=["high", "low"])

    p.add_argument("--mem_tripwire_gb", type=float, default=40.0)
    p.add_argument("--residual_l2_ceiling", type=float, default=50.0)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def assert_finite_optimizer_state(optimizer, where: str) -> None:
    """Best-effort finite-check of all floating-point optimizer state tensors."""
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
        print(f"[smoke] FAIL: no manifest.json in {cache_dir}; "
              "expected an existing precompute cache.", file=sys.stderr)
        return 2

    from wan_controlnet import WanControlnet
    from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
    from wan_transformer import CustomWanTransformer3DModel
    import bitsandbytes as bnb

    device = torch.device("cuda")

    phase = args.start_phase
    start_subfolder = "transformer" if phase == "high" else "transformer_2"
    print(f"[smoke] loading start expert: {start_subfolder} (phase={phase}) ...")
    transformer = load_expert(args.base_model_path, start_subfolder, device)

    boundary_ratio, _src = detect_boundary_ratio(
        args.base_model_path, dict(transformer.config),
    )

    print(f"[smoke] cold-init controlnet from {args.controlnet_config_repo} ...")
    config = WanControlnet.load_config(args.controlnet_config_repo)
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

    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.base_model_path, subfolder="scheduler",
    )
    scheduler.set_timesteps(1000, device=device)
    sigmas = scheduler.sigmas[:-1].to(device)
    timesteps_full = scheduler.timesteps.to(device)

    high_noise_indices = torch.where(sigmas >= boundary_ratio)[0]
    low_noise_indices = torch.where(sigmas < boundary_ratio)[0]
    if high_noise_indices.numel() == 0:
        high_noise_indices = torch.arange(0, len(sigmas) // 2, device=device)
    if low_noise_indices.numel() == 0:
        low_noise_indices = torch.arange(len(sigmas) // 2, len(sigmas), device=device)
    print(f"[smoke] boundary={boundary_ratio} "
          f"high={len(high_noise_indices)} low={len(low_noise_indices)}")

    optimizer = bnb.optim.AdamW8bit(controlnet.parameters(), lr=1e-4,
                                    weight_decay=0.01)

    dataset = BetaPairDataset(cache_dir, num_frames=args.num_frames)
    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0,
                        collate_fn=_collate_keep_meta)
    loader_iter = iter(loader)

    phase_step_times: dict[str, list[float]] = {"high": [], "low": []}
    swap_dts: list[float] = []
    n_swaps = 0
    phase_step = 0

    for step in range(args.max_steps):
        if phase_step >= args.cycle_steps and step < args.max_steps:
            phase_old = phase
            new_sub = next_subfolder_for(phase)
            swap_t0 = time.perf_counter()
            try:
                # Caller-side del: drop the only remaining strong ref to the
                # old expert so its CUDA memory is released before from_pretrained.
                del transformer
                transformer = free_then_load_expert(
                    args.base_model_path, new_sub, device,
                )
                torch.cuda.synchronize()
            except Exception as e:
                print(f"[smoke] FAIL: swap raised {type(e).__name__}: {e}",
                      file=sys.stderr)
                raise
            swap_dt = time.perf_counter() - swap_t0
            phase = "low" if phase == "high" else "high"
            assert isinstance(transformer, CustomWanTransformer3DModel), \
                f"after swap, transformer is {type(transformer).__name__}"
            assert_finite_optimizer_state(optimizer, where=f"after swap #{n_swaps + 1}")
            phase_step = 0
            n_swaps += 1
            swap_dts.append(swap_dt)
            print(f"[smoke] [swap #{n_swaps}] {phase_old} -> {phase} "
                  f"({swap_dt:.1f}s)  expected_subfolder={new_sub}")

        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            batch = next(loader_iter)

        canny = batch["canny"].to(device, non_blocking=True)
        z_real = batch["latent"].to(device, non_blocking=True)
        prompt_embeds = batch["prompt_embeds"].to(device, non_blocking=True)

        indices = high_noise_indices if phase == "high" else low_noise_indices
        sel = torch.randint(0, len(indices), (1,), device=device).item()
        t_idx = indices[sel].item()
        sigma = sigmas[t_idx].to(z_real.dtype)
        t = timesteps_full[t_idx].expand(z_real.shape[0])

        torch.cuda.synchronize()
        t0 = time.perf_counter()

        noise = torch.randn_like(z_real)
        z_t = (1.0 - sigma) * z_real + sigma * noise
        v_target = (noise - z_real).float()

        cn_states = controlnet(
            hidden_states=z_t, timestep=t,
            encoder_hidden_states=prompt_embeds,
            controlnet_states=canny, return_dict=False,
        )[0]
        cn_for_tx = [s.to(dtype=transformer.dtype) for s in cn_states]
        v_pred = transformer(
            hidden_states=z_t, timestep=t,
            encoder_hidden_states=prompt_embeds,
            controlnet_states=cn_for_tx,
            controlnet_weight=1.0, controlnet_stride=3,
            return_dict=False,
        )[0]

        loss = F.mse_loss(v_pred.float(), v_target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(controlnet.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        torch.cuda.synchronize()
        dt = time.perf_counter() - t0

        peak_mem = torch.cuda.max_memory_allocated() / 1e9
        residual_l2 = mean_residual_l2(cn_states)
        print(f"[smoke] step {step+1:02d}/{args.max_steps} | "
              f"phase={phase} ps={phase_step:02d} | "
              f"loss={loss.item():.4f} | dt={dt:.2f}s | "
              f"peak_mem={peak_mem:.2f}GB | residual_l2={residual_l2:.2e}")

        assert torch.isfinite(loss).item(), \
            f"NaN/Inf loss at step {step+1}"
        assert peak_mem < args.mem_tripwire_gb, \
            f"GPU peak {peak_mem:.2f}GB >= ceiling {args.mem_tripwire_gb}GB"
        # Output projections are zero-initialized; residuals are exactly 0 at
        # step 1 and become nonzero from step 2 onward. The ceiling catches
        # the divergence mode the diagnostic sweep showed at low sigma.
        if step >= 1:
            assert residual_l2 > 0.0, \
                f"residual L2 unexpectedly 0 at step {step+1} (>=2)"
        assert residual_l2 < args.residual_l2_ceiling, \
            f"residual L2 {residual_l2:.2e} exceeds ceiling {args.residual_l2_ceiling}"

        phase_step_times[phase].append(dt)
        phase_step += 1

    # Final optimizer-state sanity (post all swaps + final phase steps)
    assert_finite_optimizer_state(optimizer, where="end of run")

    print("")
    print(f"[smoke] swaps observed: {n_swaps} (expected >= 2)")
    assert n_swaps >= 2, \
        f"expected >= 2 swaps with cycle_steps={args.cycle_steps} max_steps={args.max_steps}, got {n_swaps}"
    if swap_dts:
        print(f"[smoke] swap times: "
              f"{[round(d, 1) for d in swap_dts]} (mean {statistics.mean(swap_dts):.1f}s)")
    for ph in ("high", "low"):
        ts = phase_step_times[ph]
        if ts:
            print(f"[smoke] phase={ph} steps={len(ts)} "
                  f"mean={statistics.mean(ts):.2f}s "
                  f"median={statistics.median(ts):.2f}s")
        else:
            print(f"[smoke] phase={ph} steps=0  (none observed)")

    print("[smoke] OK")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except AssertionError as e:
        print(f"[smoke] FAIL: {e}", file=sys.stderr)
        sys.exit(1)

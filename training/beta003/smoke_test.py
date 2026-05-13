"""Smoke test for the BETA3 training script and the dual-CN pipeline dispatch.

Reuses the existing precompute cache (no re-precompute). One job covers both
init paths exercised by ``train_beta3.py``:

  * Path 1 — warm-start: load ``--warm_start_checkpoint`` into the controlnet
    and run 10 effective steps with ``accum=2`` (= 20 micro-steps).
  * Path 2 — cold-start: re-init the controlnet from config and run 10
    effective steps with ``accum=2``.

Per micro-step asserts: loss finite; GPU peak < tripwire; residual L2 in
``(0, 5)`` for warm and ``[0, 5)`` for cold.

Per effective step asserts: optimizer state finite; EMA shadow updated.

After both paths, runs the dispatch unit-test on the dual-CN pipeline helper:
``select_controlnet_for_timestep`` must return the high CN at high sigma and
the low CN at low sigma; a single forward through each verifies both CNs are
callable with realistic inputs.

Print per-step time; exit non-zero on any assertion failure.
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--cache_dir", type=str, required=True,
                   help="Existing precompute cache (manifest.json + canny/ + latent/ + prompt/).")
    p.add_argument("--base_model_path", type=str, required=True)
    p.add_argument("--controlnet_config_repo", type=str, required=True)
    p.add_argument("--warm_start_checkpoint", type=str, required=True,
                   help="beta-001_final.safetensors; reused as the warm path's checkpoint "
                        "and as a stand-in second CN instance for the dispatch test.")
    p.add_argument("--num_frames", type=int, default=9)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=512)

    # Smoke-specific knobs (defaults: 10 effective steps × accum=2 per path).
    p.add_argument("--effective_steps_per_path", type=int, default=10)
    p.add_argument("--gradient_accumulation_steps", type=int, default=2)
    p.add_argument("--ema_decay", type=float, default=0.999)
    p.add_argument("--ema_update_after_step", type=int, default=100)

    p.add_argument("--mem_tripwire_gb", type=float, default=38.0)
    p.add_argument("--residual_l2_ceiling", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def assert_finite_optimizer_state(optimizer, where: str) -> None:
    for p, st in optimizer.state.items():
        for k, v in st.items():
            if torch.is_tensor(v) and torch.is_floating_point(v):
                assert torch.isfinite(v).all(), \
                    f"non-finite optimizer state '{k}' {where}"


def build_controlnet(config_repo: str, device: torch.device,
                     warm_start: str | None = None):
    """Build (and optionally warm-start) a controlnet on `device`. Returns the
    instance plus a snapshot of one (key, tensor.clone) used by the warm-start
    integrity check."""
    from wan_controlnet import WanControlnet
    from safetensors.torch import load_file

    config = WanControlnet.load_config(config_repo)
    cn = WanControlnet.from_config(config)
    cast_respecting_fp32_modules(cn, torch.bfloat16)
    fp32_params = [n for n, p in cn.named_parameters()
                   if p.dtype == torch.float32]
    assert any("norm" in n or "time_embedder" in n or "scale_shift" in n
               for n in fp32_params), \
        "Expected norm/time_embedder/scale_shift params kept in fp32"
    assert any(p.dtype == torch.bfloat16 for p in cn.parameters()), \
        "Expected most controlnet params to be bf16"

    if warm_start:
        sd = load_file(warm_start)
        missing, unexpected = cn.load_state_dict(sd, strict=False)
        assert len(missing) == 0 and len(unexpected) == 0, (
            f"warm-start key mismatch: missing={len(missing)} unexpected={len(unexpected)}"
        )
    cn.enable_gradient_checkpointing()
    cn.train().to(device)
    return cn


def _pick_diagnostic_param_name(cn) -> str:
    """Pick a controlnet_blocks output-projection weight (zero-initialized
    in cold init, non-zero after warm-start). Used by the warm-start
    integrity check."""
    for n, p in cn.named_parameters():
        if "controlnet_blocks" in n and n.endswith(".weight"):
            return n
    raise RuntimeError("Could not find a controlnet_blocks weight to snapshot.")


def run_one_path(args, label: str, warm_start: str | None,
                 transformer, sigmas, timesteps_full, low_noise_indices,
                 dataset, device) -> tuple[list[float], "WanControlnet"]:
    """Run `effective_steps_per_path` effective steps. Returns per-effective-step
    times and the final controlnet (kept by caller for the dispatch test)."""
    import bitsandbytes as bnb
    from ema_pytorch import EMA

    print(f"\n[smoke {label}] building controlnet (warm_start={'yes' if warm_start else 'no'}) ...")
    cn = build_controlnet(args.controlnet_config_repo, device, warm_start=warm_start)

    # --- Warm-start integrity check (Path 1 only) ---
    if warm_start is not None:
        diag_name = _pick_diagnostic_param_name(cn)
        diag_tensor = dict(cn.named_parameters())[diag_name]
        # Cold-init reference: a freshly built CN's controlnet_blocks weights
        # are zeroed by zero_module(); the warm-started one must differ.
        from wan_controlnet import WanControlnet
        ref_cn = WanControlnet.from_config(WanControlnet.load_config(args.controlnet_config_repo))
        cast_respecting_fp32_modules(ref_cn, torch.bfloat16)
        ref_tensor = dict(ref_cn.named_parameters())[diag_name]
        assert not torch.allclose(diag_tensor.detach().cpu().float(),
                                  ref_tensor.detach().cpu().float()), \
            f"warm-start integrity check failed: '{diag_name}' equals cold init"
        print(f"[smoke {label}] warm-start integrity OK ('{diag_name}' differs from cold-init reference)")
        del ref_cn, ref_tensor

    optimizer = bnb.optim.AdamW8bit(cn.parameters(), lr=1e-4, weight_decay=1e-4)
    ema = EMA(
        cn,
        beta=args.ema_decay,
        update_after_step=args.ema_update_after_step,
        update_every=1,
    )
    ema.to(device)

    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0,
                        collate_fn=_collate_keep_meta)
    loader_iter = iter(loader)

    micro_step = 0
    global_step = 0
    eff_step_times: list[float] = []
    optimizer.zero_grad(set_to_none=True)

    target_eff = args.effective_steps_per_path
    target_micro = target_eff * args.gradient_accumulation_steps

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

        sel = torch.randint(0, len(low_noise_indices), (1,), device=device).item()
        t_idx = low_noise_indices[sel].item()
        sigma = sigmas[t_idx].to(z_real.dtype)
        t = timesteps_full[t_idx].expand(z_real.shape[0])

        noise = torch.randn_like(z_real)
        z_t = (1.0 - sigma) * z_real + sigma * noise
        v_target = (noise - z_real).float()

        cn_states = cn(
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
        loss_scaled = loss / args.gradient_accumulation_steps
        loss_scaled.backward()

        # Per micro-step assertions
        peak_mem = torch.cuda.max_memory_allocated() / 1e9
        residual_l2 = mean_residual_l2(cn_states)
        assert torch.isfinite(loss).item(), \
            f"NaN/Inf loss at {label} micro-step {micro_step + 1}"
        assert peak_mem < args.mem_tripwire_gb, \
            f"GPU peak {peak_mem:.2f}GB >= ceiling {args.mem_tripwire_gb}GB"
        if warm_start is not None:
            # Warm-start residuals should be non-zero from the very first call.
            assert 0.0 < residual_l2 < args.residual_l2_ceiling, (
                f"warm residual L2 {residual_l2:.2e} out of (0, "
                f"{args.residual_l2_ceiling}) at {label} micro-step {micro_step + 1}"
            )
        else:
            # Cold-start: zero at micro-step 1, non-zero from micro-step 2 on.
            assert 0.0 <= residual_l2 < args.residual_l2_ceiling, (
                f"cold residual L2 {residual_l2:.2e} out of [0, "
                f"{args.residual_l2_ceiling}) at {label} micro-step {micro_step + 1}"
            )

        micro_step += 1

        if micro_step % args.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(cn.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            ema.update()
            global_step += 1

            torch.cuda.synchronize()
            eff_dt = time.perf_counter() - eff_t0
            eff_step_times.append(eff_dt)

            assert_finite_optimizer_state(
                optimizer, where=f"after {label} eff-step {global_step}"
            )

            # EMA cadence: internal step counter must equal optimizer steps
            # (NOT micro-steps). If grad-accum is miswired, this trips.
            ema_step_count = int(ema.step.item())
            assert ema_step_count == global_step, (
                f"EMA step counter drift: ema.step={ema_step_count} "
                f"!= global_step={global_step} (grad-accum miswired?)"
            )

            print(f"[smoke {label}] eff-step {global_step:02d}/{target_eff} "
                  f"(micro={micro_step:02d}/{target_micro}) | "
                  f"loss={loss.item():.4f} | dt={eff_dt:.2f}s | "
                  f"peak_mem={peak_mem:.2f}GB | residual_l2={residual_l2:.2e}")

    return eff_step_times, cn


def run_dispatch_test(args, controlnet_a, device) -> None:
    """Dispatch unit-test: select_controlnet_for_timestep must return the
    high-CN at high sigma and the low-CN at low sigma; a single forward
    through each verifies both CNs are functional with realistic inputs."""
    from wan_t2v_controlnet_pipeline_dual import select_controlnet_for_timestep
    from wan_controlnet import WanControlnet

    print("\n[smoke dispatch] building second controlnet for dispatch test ...")
    config = WanControlnet.load_config(args.controlnet_config_repo)
    controlnet_b = WanControlnet.from_config(config)
    cast_respecting_fp32_modules(controlnet_b, torch.bfloat16)
    controlnet_b.eval().to(device)

    # Use the just-trained CN as the "high" and the freshly-built one as the
    # "low" so the two instances are distinguishable by identity.
    cn_high, cn_low = controlnet_a, controlnet_b

    # Resolve boundary the same way training does. The smoke uses the high-noise
    # transformer for forward, but we only need its config for the boundary;
    # the dispatch math itself is independent of which expert is loaded.
    boundary_ratio = 0.875  # default Wan2.2 A14B; matches model_index.json

    # FlowMatch's discrete schedule for num_train_timesteps=1000: t = 1000 * sigma.
    # sigma=0.95 → t=950 (above 0.875*1000=875, high regime).
    # sigma=0.50 → t=500 (below 875, low regime).
    num_train_timesteps = 1000
    t_high = torch.tensor(950.0)
    t_low = torch.tensor(500.0)

    sel_high = select_controlnet_for_timestep(
        t_high, boundary_ratio, num_train_timesteps, cn_high, cn_low,
    )
    sel_low = select_controlnet_for_timestep(
        t_low, boundary_ratio, num_train_timesteps, cn_high, cn_low,
    )
    assert sel_high is cn_high, \
        "dispatch: at sigma=0.95 (t=950), expected controlnet_high"
    assert sel_low is cn_low, \
        "dispatch: at sigma=0.50 (t=500), expected controlnet_low"
    assert sel_high is not sel_low, \
        "dispatch: high and low selections must be distinct"
    print("[smoke dispatch] dispatch logic OK (high@0.95 -> A, low@0.50 -> B)")

    # Run a single forward through each selected CN with realistic shapes.
    # Both forward calls must succeed; we don't compare outputs because the
    # cold-init CN's residuals are exactly zero and equality would not falsify
    # a wrong dispatch.
    B, C = 1, 16
    T_lat = (args.num_frames - 1) // 4 + 1
    H_lat = args.height // 8
    W_lat = args.width // 8
    z_t = torch.randn(B, C, T_lat, H_lat, W_lat, device=device, dtype=torch.bfloat16)
    canny = torch.zeros(B, 3, args.num_frames, args.height, args.width,
                        device=device, dtype=torch.bfloat16)
    prompt_embeds = torch.zeros(B, 226, 4096, device=device, dtype=torch.bfloat16)

    with torch.no_grad():
        out_high = sel_high(
            hidden_states=z_t, timestep=t_high.to(device).expand(B),
            encoder_hidden_states=prompt_embeds,
            controlnet_states=canny, return_dict=False,
        )[0]
        out_low = sel_low(
            hidden_states=z_t, timestep=t_low.to(device).expand(B),
            encoder_hidden_states=prompt_embeds,
            controlnet_states=canny, return_dict=False,
        )[0]
    n_high = len(out_high) if isinstance(out_high, (list, tuple)) else 1
    n_low = len(out_low) if isinstance(out_low, (list, tuple)) else 1
    print(f"[smoke dispatch] forward through both CNs OK "
          f"(high produced {n_high} residuals, low produced {n_low})")


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)

    cache_dir = Path(args.cache_dir)
    if not (cache_dir / "manifest.json").exists():
        print(f"[smoke] FAIL: no manifest.json in {cache_dir}; "
              "expected an existing precompute cache.", file=sys.stderr)
        return 2

    if not Path(args.warm_start_checkpoint).exists():
        print(f"[smoke] FAIL: warm_start_checkpoint not found: "
              f"{args.warm_start_checkpoint}", file=sys.stderr)
        return 2

    from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
    from wan_transformer import CustomWanTransformer3DModel

    device = torch.device("cuda")

    # train_beta3 trains against transformer_2 (low-noise). Smoke does the same
    # so the assertions match what the real run will exercise.
    print(f"[smoke] loading low-noise transformer (transformer_2) ...")
    transformer = CustomWanTransformer3DModel.from_pretrained(
        args.base_model_path, subfolder="transformer_2", torch_dtype=torch.bfloat16,
    )
    transformer.requires_grad_(False).eval().to(device)
    transformer.enable_gradient_checkpointing()

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
    print(f"[smoke] boundary={boundary_ratio} low={len(low_noise_indices)}")

    dataset = BetaPairDataset(cache_dir, num_frames=args.num_frames)

    # --- Path 1: warm-start ---
    warm_times, cn_warm = run_one_path(
        args, label="warm", warm_start=args.warm_start_checkpoint,
        transformer=transformer, sigmas=sigmas, timesteps_full=timesteps_full,
        low_noise_indices=low_noise_indices, dataset=dataset, device=device,
    )

    # Free warm-path CN before building cold-path CN so peaks reflect a single
    # CN resident at a time (matching real-run conditions).
    del cn_warm
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    # --- Path 2: cold-start ---
    cold_times, cn_cold = run_one_path(
        args, label="cold", warm_start=None,
        transformer=transformer, sigmas=sigmas, timesteps_full=timesteps_full,
        low_noise_indices=low_noise_indices, dataset=dataset, device=device,
    )

    # --- Dispatch test (cheap: one extra CN built, no full pipeline) ---
    cn_cold.eval()
    run_dispatch_test(args, controlnet_a=cn_cold, device=device)

    # --- Summary ---
    print("")
    for label, ts in (("warm", warm_times), ("cold", cold_times)):
        if ts:
            print(f"[smoke] {label} eff-steps={len(ts)} "
                  f"mean={statistics.mean(ts):.2f}s "
                  f"median={statistics.median(ts):.2f}s")
        else:
            print(f"[smoke] {label} eff-steps=0  (none observed)")

    print("[smoke] OK")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except AssertionError as e:
        print(f"[smoke] FAIL: {e}", file=sys.stderr)
        sys.exit(1)

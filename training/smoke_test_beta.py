"""End-to-end smoke test on a 5-pair subset.

Runs `precompute_beta` with `--limit 5` then `train_beta` for ~20 steps,
measures per-step wall-time, and writes the budget projection to
`training_cards/{run_id}_smoke_results.json` so the card can be filled
before launching the real 24h job.
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from training import precompute_beta as precompute_mod
from training.dataset_beta import BetaPairDataset
from training.train_beta import (
    cast_respecting_fp32_modules,
    detect_boundary_ratio,
    mean_residual_l2,
    _collate_keep_meta,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--base_model_path", type=str, required=True)
    p.add_argument("--controlnet_config_repo", type=str, required=True)
    p.add_argument("--input_faces_dir", type=str, required=True,
                   help="Directory of input face PNGs (e.g., $HOME/data/wan-beta/input_faces).")
    p.add_argument("--targets_dir", type=str, required=True,
                   help="Directory of PTDiffusion target JPGs (e.g., $HOME/data/wan-beta/targets).")
    p.add_argument("--cache_dir", type=str, required=True,
                   help="Cache root (e.g., $HOME/cache/wan-beta). Smoke cache is written to <cache_dir>/_smoke.")
    p.add_argument("--num_frames", type=int, default=9)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--limit", type=int, default=5)
    p.add_argument("--num_train_steps", type=int, default=20)
    p.add_argument("--warmup_steps", type=int, default=3,
                   help="Skip first N steps when computing per-step wall-time.")
    p.add_argument("--max_per_step_seconds", type=float, default=8.5)
    p.add_argument("--mem_tripwire_gb", type=float, default=44.0)
    p.add_argument("--card_run_id", type=str, default="beta-001")
    return p.parse_args()


def run_precompute(args: argparse.Namespace, smoke_cache: Path) -> None:
    cli = [
        "--input_faces_dir", args.input_faces_dir,
        "--targets_dir", args.targets_dir,
        "--output_dir", str(smoke_cache),
        "--base_model_path", args.base_model_path,
        "--height", str(args.height),
        "--width", str(args.width),
        "--num_frames", str(args.num_frames),
        "--limit", str(args.limit),
    ]
    sys.argv = ["precompute_beta"] + cli
    precompute_mod.main()


def assert_cache_files_exist(smoke_cache: Path) -> None:
    manifest = json.loads((smoke_cache / "manifest.json").read_text())
    assert manifest, "smoke manifest is empty"
    for rec in manifest:
        for key in ("canny_path", "latent_path", "prompt_path"):
            p = smoke_cache / rec[key]
            assert p.exists(), f"missing cache file: {p}"
            t = torch.load(p, map_location="cpu")
            assert t.numel() > 0, f"empty cache file: {p}"


def run_training_steps(args: argparse.Namespace, smoke_cache: Path) -> dict:
    """Build models manually and run N steps. Returns timing summary."""
    from wan_transformer import CustomWanTransformer3DModel
    from wan_controlnet import WanControlnet
    from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
    import bitsandbytes as bnb

    device = torch.device("cuda")

    print("[smoke] loading high-noise transformer ...")
    transformer = CustomWanTransformer3DModel.from_pretrained(
        args.base_model_path, subfolder="transformer", torch_dtype=torch.bfloat16,
    )
    transformer.requires_grad_(False).eval().to(device)
    # Frozen, but cn_for_tx (input) needs grad → activations would be retained
    # for backward without checkpointing. Recompute on demand to fit in A40 VRAM.
    transformer.enable_gradient_checkpointing()

    print("[smoke] cold-init controlnet from config ...")
    config = WanControlnet.load_config(args.controlnet_config_repo)
    controlnet = WanControlnet.from_config(config)
    cast_respecting_fp32_modules(controlnet, torch.bfloat16)

    # dtype asserts (after cast)
    fp32_params = [n for n, p in controlnet.named_parameters()
                   if p.dtype == torch.float32]
    assert any("norm" in n or "time_embedder" in n or "scale_shift" in n
               for n in fp32_params), \
        "Expected norm/time_embedder/scale_shift params kept in fp32 after cast"
    assert any(p.dtype == torch.bfloat16 for p in controlnet.parameters()), \
        "Expected most controlnet params to be bf16 after cast"

    controlnet.enable_gradient_checkpointing()
    controlnet.train().to(device)

    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.base_model_path, subfolder="scheduler",
    )
    scheduler.set_timesteps(1000, device=device)
    sigmas = scheduler.sigmas[:-1].to(device)
    timesteps_full = scheduler.timesteps.to(device)
    boundary_ratio, _ = detect_boundary_ratio(args.base_model_path,
                                              dict(transformer.config))
    high_noise_indices = torch.where(sigmas >= boundary_ratio)[0]
    if high_noise_indices.numel() == 0:
        high_noise_indices = torch.arange(0, len(sigmas) // 2, device=device)
    print(f"[smoke] boundary_ratio={boundary_ratio} "
          f"high_noise_steps={len(high_noise_indices)}")

    optimizer = bnb.optim.AdamW8bit(controlnet.parameters(), lr=1e-4,
                                    weight_decay=0.01)

    dataset = BetaPairDataset(smoke_cache, num_frames=args.num_frames)
    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0,
                        collate_fn=_collate_keep_meta)
    loader_iter = iter(loader)

    step_times: list[float] = []
    grad_assert_done = False
    for step in range(args.num_train_steps):
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            batch = next(loader_iter)

        canny = batch["canny"].to(device, non_blocking=True)
        z_real = batch["latent"].to(device, non_blocking=True)
        prompt_embeds = batch["prompt_embeds"].to(device, non_blocking=True)

        sel = torch.randint(0, len(high_noise_indices), (1,), device=device).item()
        t_idx = high_noise_indices[sel].item()
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

        if not grad_assert_done:
            tx_with_grad = [
                n for n, p in transformer.named_parameters()
                if p.grad is not None and p.grad.abs().sum() > 0
            ]
            assert not tx_with_grad, \
                f"Transformer should have no grads but found {len(tx_with_grad)}"
            assert any(
                p.grad is not None and p.grad.abs().sum() > 0
                for p in controlnet.parameters()
            ), "No controlnet param has nonzero grad after step 1"
            grad_assert_done = True

        torch.nn.utils.clip_grad_norm_(controlnet.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        torch.cuda.synchronize()
        dt = time.perf_counter() - t0

        peak_mem = torch.cuda.max_memory_allocated() / 1e9
        print(f"[smoke] step {step+1:02d}/{args.num_train_steps} | "
              f"loss={loss.item():.4f} | dt={dt:.2f}s | "
              f"peak_mem={peak_mem:.2f}GB | residual_l2={mean_residual_l2(cn_states):.2e}")

        if not torch.isfinite(loss):
            raise RuntimeError(f"NaN loss at step {step+1}")
        if peak_mem >= args.mem_tripwire_gb:
            raise RuntimeError(
                f"GPU memory {peak_mem:.2f}GB >= tripwire "
                f"{args.mem_tripwire_gb}GB at step {step+1}"
            )
        if step >= args.warmup_steps:
            step_times.append(dt)

    if not step_times:
        raise RuntimeError("Not enough steps after warmup to estimate timing.")

    median = statistics.median(step_times)
    p90 = sorted(step_times)[max(0, int(0.9 * len(step_times)) - 1)]
    mean = statistics.mean(step_times)
    std = statistics.pstdev(step_times) if len(step_times) > 1 else 0.0

    proj_total_steps = 10000
    proj_median = median * proj_total_steps
    proj_p90 = p90 * proj_total_steps

    return {
        "step_times": [round(t, 4) for t in step_times],
        "median": round(median, 4),
        "p90": round(p90, 4),
        "mean": round(mean, 4),
        "std": round(std, 4),
        "projected_wall_time_median_seconds": round(proj_median, 1),
        "projected_wall_time_p90_seconds": round(proj_p90, 1),
        "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
    }


def fmt_hms(s: float) -> str:
    s = int(s)
    h, r = divmod(s, 3600)
    m, sec = divmod(r, 60)
    return f"{h:02d}:{m:02d}:{sec:02d}"


def main() -> int:
    args = parse_args()
    smoke_cache = Path(args.cache_dir) / "_smoke"
    smoke_cache.mkdir(parents=True, exist_ok=True)

    print(f"[smoke] precompute -> {smoke_cache}")
    run_precompute(args, smoke_cache)
    assert_cache_files_exist(smoke_cache)

    print(f"[smoke] running {args.num_train_steps} training steps ...")
    timing = run_training_steps(args, smoke_cache)

    median = timing["median"]
    p90 = timing["p90"]
    proj_median_s = timing["projected_wall_time_median_seconds"]
    proj_p90_s = timing["projected_wall_time_p90_seconds"]

    print("")
    print("SMOKE TEST PASSED" if median <= args.max_per_step_seconds else "SMOKE TEST FAILED")
    print(f"per-step wall-time (median): {median:.2f}s")
    print(f"per-step wall-time (p90):    {p90:.2f}s")
    print(f"projected 10000 steps @ median: {fmt_hms(proj_median_s)}")
    print(f"projected 10000 steps @ p90:    {fmt_hms(proj_p90_s)}")
    print(f"peak GPU memory: {timing['peak_mem_gb']:.2f} GB")

    smoke_results_path = Path("training_cards") / f"{args.card_run_id}_smoke_results.json"
    smoke_results_path.parent.mkdir(parents=True, exist_ok=True)
    smoke_results_path.write_text(json.dumps({
        "smoke_step_median": f"{median:.2f}",
        "smoke_step_p90": f"{p90:.2f}",
        "smoke_projected_wall_time": fmt_hms(proj_median_s),
        "smoke_projected_wall_time_p90": fmt_hms(proj_p90_s),
        "smoke_peak_mem_gb": timing["peak_mem_gb"],
        "smoke_step_times": timing["step_times"],
    }, indent=2))
    print(f"[smoke] wrote {smoke_results_path}")

    if median > args.max_per_step_seconds:
        print(f"[FAIL] median per-step {median:.2f}s > {args.max_per_step_seconds}s "
              "would push the real run over 24h. Consider T=5 or 2x A40 DDP.")
        return 1
    if p90 > args.max_per_step_seconds:
        print(f"[WARN] p90 per-step {p90:.2f}s > {args.max_per_step_seconds}s "
              "(median ok). Variance suggests borderline; user decides.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

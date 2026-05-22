"""Smoke test for beta-009 training.

Verifies what beta-009 adds on top of beta-008:

1. **z_face encoding path:** load (3, H, W) uint8 from the chosen
   ``--face_target_subdir`` (raw_face or silhouette), VAE-encode it, get
   back a normalized latent of the same shape as ``z_real``. Asserted by
   comparing shapes.
2. **Additive face-region loss:** ``λ_face · face_region_MSE(v_pred, v_face_target)``
   with ``v_face_target = noise − z_face``. Asserted to be finite and
   non-negative.
3. **Total loss decomposition:** ``loss == loss_fm + λ_face · loss_face +
   λ_cons · loss_consistency`` (self-distillation optional).
4. End-to-end one inference call on a held-out eval sample (unchanged from
   beta-008's smoke).
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from training.dataset_beta import BetaPairDataset
from training.utils import (
    cast_respecting_fp32_modules,
    detect_boundary_ratio,
    mean_residual_l2,
    _collate_keep_meta,
)
from training.utils import _maybe_force_native_attention
from training.beta009.train import (
    _frames_target_mse,
    _frames_target_ssim,
    _load_target_image,
    _save_video,
    _build_eval_periodic_splits,
    _compute_cn_end_high_noise,
    encode_face_targets_to_latent,
    WAN_T5_MAX_SEQ_LEN,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--cache_dir", type=str, required=True)
    p.add_argument("--targets_dir", type=str, required=True)
    p.add_argument("--base_model_path", type=str, required=True)
    p.add_argument("--controlnet_config_repo", type=str, required=True)
    p.add_argument("--smoke_output_dir", type=str, default="/tmp/wan_beta9_smoke")

    p.add_argument("--control_subdir", type=str, default="silhouette")
    p.add_argument("--face_mask_subdir", type=str, default="silhouette")
    p.add_argument("--face_target_subdir", type=str, default="raw_face",
                   choices=("raw_face", "silhouette"))
    p.add_argument("--lambda_face", type=float, default=5.0)

    p.add_argument("--num_frames", type=int, default=9)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=512)

    p.add_argument("--effective_steps", type=int, default=2)
    p.add_argument("--micro_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--ema_decay", type=float, default=0.99)
    p.add_argument("--ema_update_after_step", type=int, default=1)

    p.add_argument("--num_cn_layers", type=int, default=None)
    p.add_argument("--controlnet_stride", type=int, default=3)

    p.add_argument("--use_self_distillation", action="store_true")
    p.add_argument("--lambda_consistency", type=float, default=0.5)

    p.add_argument("--eval_size", type=int, default=100)
    p.add_argument("--inference_steps", type=int, default=5)
    p.add_argument("--inference_guidance_scale", type=float, default=5.0)
    p.add_argument("--inference_controlnet_weight", type=float, default=1.0)
    p.add_argument("--inference_controlnet_end", type=float, default=None)
    p.add_argument("--negative_prompt", type=str, default="bad quality, worst quality")

    p.add_argument("--mem_tripwire_gb", type=float, default=90.0)
    p.add_argument("--residual_l2_ceiling", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--cpu_offload_inference", action="store_true",
        help="Keep transformer_2 on CPU during training and enable "
             "pipe.enable_model_cpu_offload() before the inference smoke. "
             "Lets the dual-expert pipeline fit on a 45-48 GB GPU (A40/L40S).",
    )
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
    np.random.seed(args.seed)

    cache_dir = Path(args.cache_dir)
    if not (cache_dir / "manifest.json").exists():
        print(f"[smoke] FAIL: no manifest.json in {cache_dir}", file=sys.stderr)
        return 2
    targets_dir = Path(args.targets_dir)
    if not targets_dir.exists():
        print(f"[smoke] FAIL: targets_dir does not exist: {targets_dir}",
              file=sys.stderr)
        return 2
    face_image_dir = cache_dir / args.face_target_subdir
    if not face_image_dir.exists():
        print(f"[smoke] FAIL: face_target_subdir does not exist: {face_image_dir}",
              file=sys.stderr)
        return 2
    smoke_out = Path(args.smoke_output_dir)
    smoke_out.mkdir(parents=True, exist_ok=True)

    from accelerate import Accelerator
    from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
    from diffusers import AutoencoderKLWan
    from transformers import AutoTokenizer, UMT5EncoderModel
    from wan_transformer import CustomWanTransformer3DModel
    from wan_controlnet import WanControlnet
    from wan_t2v_controlnet_pipeline import WanTextToVideoControlnetPipeline, prompt_clean
    import bitsandbytes as bnb
    from ema_pytorch import EMA
    from PIL import Image

    device = torch.device("cuda")
    try:
        major, minor = torch.cuda.get_device_capability(0)
        print(f"[smoke] GPU={torch.cuda.get_device_name(0)} compute_cap={major}.{minor}")
    except Exception:
        major = 0

    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)
    print(f"[smoke] Accelerator(gradient_accumulation_steps="
          f"{args.gradient_accumulation_steps})")

    # ---- Models ----
    print(f"[smoke] loading high-noise transformer ...")
    transformer = CustomWanTransformer3DModel.from_pretrained(
        args.base_model_path, subfolder="transformer", torch_dtype=torch.bfloat16,
    )
    transformer.requires_grad_(False).eval().to(device)
    transformer.enable_gradient_checkpointing()
    _maybe_force_native_attention(transformer, "transformer")

    boundary_ratio, _src = detect_boundary_ratio(
        args.base_model_path, dict(transformer.config),
    )
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.base_model_path, subfolder="scheduler",
    )
    scheduler.set_timesteps(1000, device=device)
    sigmas = scheduler.sigmas[:-1].to(device)
    timesteps_full = scheduler.timesteps.to(device)
    high_noise_indices = torch.where(sigmas >= boundary_ratio)[0]
    if high_noise_indices.numel() == 0:
        high_noise_indices = torch.arange(0, len(sigmas) // 2, device=device)
    print(f"[smoke] boundary={boundary_ratio} high={len(high_noise_indices)} of {len(sigmas)}")

    print(f"[smoke] building cold-init controlnet (stride={args.controlnet_stride}) ...")
    config = WanControlnet.load_config(args.controlnet_config_repo)
    if args.num_cn_layers is not None:
        config["num_layers"] = args.num_cn_layers
    cn = WanControlnet.from_config(config)
    cast_respecting_fp32_modules(cn, torch.bfloat16)
    cn.enable_gradient_checkpointing()
    cn.train().to(device)
    _maybe_force_native_attention(cn, "controlnet")

    optimizer = bnb.optim.AdamW8bit(cn.parameters(), lr=5e-5, weight_decay=1e-4)
    ema = EMA(cn, beta=args.ema_decay,
              update_after_step=args.ema_update_after_step, update_every=1)
    ema.to(device)

    # ---- Data: face/prompt-stratified split ----
    full_dataset = BetaPairDataset(cache_dir, num_frames=args.num_frames,
                                   control_subdir=args.control_subdir)
    total_n = len(full_dataset)
    if total_n != 10000:
        print(f"[smoke] FAIL: expected 10000 records; got {total_n}", file=sys.stderr)
        return 2
    train_indices, eval_indices, periodic_indices = _build_eval_periodic_splits(
        full_dataset.records
    )
    train_dataset = Subset(full_dataset, train_indices)
    print(f"[smoke] split: train={len(train_dataset)} eval={len(eval_indices)} "
          f"periodic={len(periodic_indices)}")

    loader = DataLoader(
        train_dataset, batch_size=args.micro_batch_size, shuffle=True, num_workers=0,
        collate_fn=_collate_keep_meta, drop_last=True,
    )
    loader_iter = iter(loader)

    # ---- Pre-cache face mask (where face loss applies) ----
    mask_dir = cache_dir / args.face_mask_subdir
    if not mask_dir.exists():
        print(f"[smoke] FAIL: mask dir missing: {mask_dir}", file=sys.stderr)
        return 2
    spatial_factor = 8
    h_lat = args.height // spatial_factor
    w_lat = args.width // spatial_factor
    unique_face_idxs = sorted({r["face_idx"] for r in full_dataset.records})
    face_masks_latent: dict[int, torch.Tensor] = {}
    for fi in unique_face_idxs:
        mp = mask_dir / f"face_{fi}.pt"
        raw = torch.load(mp, map_location="cpu", weights_only=True)
        m_bin = (raw[0] > 0).float()
        m_lat = F.avg_pool2d(
            m_bin.unsqueeze(0).unsqueeze(0),
            kernel_size=spatial_factor, stride=spatial_factor,
        ).squeeze(0).squeeze(0).contiguous()
        face_masks_latent[int(fi)] = m_lat
    coverage_mean = float(torch.stack(list(face_masks_latent.values())).mean().item())
    print(f"[smoke] {len(face_masks_latent)} masks at {h_lat}×{w_lat} "
          f"coverage={coverage_mean:.3f}")

    # ---- Pre-encode neg prompt + drop text_encoder ----
    print(f"[smoke] loading text_encoder + encoding neg prompt + dropping ...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, subfolder="tokenizer")
    text_encoder = UMT5EncoderModel.from_pretrained(
        args.base_model_path, subfolder="text_encoder", torch_dtype=torch.bfloat16,
    ).eval().to(device)
    with torch.no_grad():
        ti = tokenizer(
            [prompt_clean(args.negative_prompt)],
            padding="max_length", max_length=WAN_T5_MAX_SEQ_LEN, truncation=True,
            add_special_tokens=True, return_attention_mask=True, return_tensors="pt",
        )
        ids = ti.input_ids.to(device)
        mask = ti.attention_mask.to(device)
        lens = mask.gt(0).sum(dim=1).long()
        neg = text_encoder(ids, mask).last_hidden_state.to(torch.bfloat16)
        neg = [u[:v] for u, v in zip(neg, lens)]
        neg_embeds = torch.stack(
            [torch.cat([u, u.new_zeros(WAN_T5_MAX_SEQ_LEN - u.size(0), u.size(1))])
             for u in neg], dim=0
        )
    del text_encoder, tokenizer, ti, ids, mask, lens, neg
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    print(f"[smoke] neg_embeds shape={tuple(neg_embeds.shape)}")

    # ---- VAE + transformer_2 ----
    print(f"[smoke] loading vae + transformer_2 ...")
    vae = AutoencoderKLWan.from_pretrained(
        args.base_model_path, subfolder="vae", torch_dtype=torch.bfloat16,
    ).eval().to(device)
    transformer_2 = CustomWanTransformer3DModel.from_pretrained(
        args.base_model_path, subfolder="transformer_2", torch_dtype=torch.bfloat16,
    )
    transformer_2.requires_grad_(False).eval()
    if not args.cpu_offload_inference:
        transformer_2 = transformer_2.to(device)
    else:
        print(f"[smoke] cpu_offload_inference: leaving transformer_2 on CPU "
              f"until inference")
    _maybe_force_native_attention(transformer_2, "transformer_2")

    # ---- z_face encoding (the new path) ----
    # Encode only the 10 faces we'll actually use during the smoke (small subset
    # of unique faces) to keep startup quick. The full train.py encodes all 100.
    smoke_face_idxs = unique_face_idxs[:10]
    print(f"[smoke] encoding {len(smoke_face_idxs)} z_face latents from "
          f"{face_image_dir} ({args.face_target_subdir}) ...")
    t_enc0 = time.perf_counter()
    z_faces_latent = encode_face_targets_to_latent(
        face_image_dir=face_image_dir,
        unique_face_idxs=smoke_face_idxs,
        vae=vae, num_frames=args.num_frames,
        height=args.height, width=args.width, device=device,
    )
    t_enc = time.perf_counter() - t_enc0
    sample_z = next(iter(z_faces_latent.values()))
    print(f"[smoke] z_face: {len(z_faces_latent)} latents in {t_enc:.1f}s; "
          f"shape={tuple(sample_z.shape)} dtype={sample_z.dtype} "
          f"device={sample_z.device}")

    # ---- Pre-stage 1 eval sample for the inference smoke ----
    rec = full_dataset.records[eval_indices[0]]
    canny_path = cache_dir / args.control_subdir / Path(rec["canny_path"]).name
    canny_u8 = torch.load(canny_path, map_location="cpu", weights_only=True)
    canny_img = Image.fromarray(canny_u8.permute(1, 2, 0).numpy())
    prompt_embed = torch.load(cache_dir / rec["prompt_path"], map_location="cpu",
                              weights_only=True).to(torch.bfloat16)
    if prompt_embed.dim() == 2:
        prompt_embed = prompt_embed.unsqueeze(0)
    prompt_embed = prompt_embed.to(device)
    target_hwc = _load_target_image(targets_dir, rec["face_idx"], rec["slug"],
                                    args.height, args.width)
    print(f"[smoke] pre-staged eval sample: face_idx={rec['face_idx']} slug={rec['slug']}")

    # ---- Pipeline ----
    eval_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.base_model_path, subfolder="scheduler",
    )
    pipe = WanTextToVideoControlnetPipeline(
        tokenizer=None, text_encoder=None,
        transformer=transformer, transformer_2=transformer_2, vae=vae,
        controlnet=cn, scheduler=eval_scheduler,
        boundary_ratio=boundary_ratio,
    )
    print(f"[smoke] pipeline built")

    # ---- Mini training loop (manual-gate accumulate) ----
    target_eff = args.effective_steps
    target_micro = target_eff * args.gradient_accumulation_steps
    print(f"[smoke] running {target_eff} eff-steps × accum="
          f"{args.gradient_accumulation_steps} = {target_micro} micro-steps")

    optimizer.zero_grad(set_to_none=True)
    grad_assert_done = False
    z_face_shape_asserted = False
    loss_decomp_asserted = False
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

        # Reject micro-batches whose face_idx isn't in the smoke z_face subset
        # — otherwise we'd KeyError on the smoke's truncated dict. Real train.py
        # encodes all 100 so this never happens there.
        if any(int(fi) not in z_faces_latent for fi in batch["face_idx"]):
            continue

        if micro_step % args.gradient_accumulation_steps == 0:
            torch.cuda.synchronize()
            eff_t0 = time.perf_counter()

        with accelerator.accumulate(cn):
            canny = batch["canny"].to(device, non_blocking=True)
            z_real = batch["latent"].to(device, non_blocking=True)
            prompt_embeds = batch["prompt_embeds"].to(device, non_blocking=True)
            B = z_real.shape[0]
            face_idxs_b = batch["face_idx"]

            sel = torch.randint(0, len(high_noise_indices), (B,), device=device)
            t_idx = high_noise_indices[sel]
            sigma = sigmas[t_idx].to(z_real.dtype)
            t = timesteps_full[t_idx]
            sigma_b = sigma.view(B, 1, 1, 1, 1)

            noise = torch.randn_like(z_real)
            z_t = (1.0 - sigma_b) * z_real + sigma_b * noise
            v_target = (noise - z_real).float()

            z_face_b = torch.stack(
                [z_faces_latent[int(fi)] for fi in face_idxs_b], dim=0,
            ).to(device=z_real.device, dtype=z_real.dtype)
            if not z_face_shape_asserted:
                assert z_face_b.shape == z_real.shape, (
                    f"z_face shape {tuple(z_face_b.shape)} != z_real shape "
                    f"{tuple(z_real.shape)} — VAE temporal/spatial compression "
                    "must produce the same latent shape from a replicated still."
                )
                z_face_shape_asserted = True
                print(f"[smoke] z_face shape matches z_real: {tuple(z_real.shape)}")
            v_face_target = (noise - z_face_b).float()

            v_pred_ema = None
            if args.use_self_distillation:
                with torch.no_grad():
                    cn_states_ema = ema.ema_model(
                        hidden_states=z_t, timestep=t,
                        encoder_hidden_states=prompt_embeds,
                        controlnet_states=canny, return_dict=False,
                    )[0]
                    cn_for_tx_ema = [s.to(dtype=transformer.dtype) for s in cn_states_ema] \
                        if isinstance(cn_states_ema, (list, tuple)) \
                        else cn_states_ema.to(dtype=transformer.dtype)
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
                if isinstance(cn_states, (list, tuple)) \
                else cn_states.to(dtype=transformer.dtype)
            v_pred = transformer(
                hidden_states=z_t, timestep=t,
                encoder_hidden_states=prompt_embeds,
                controlnet_states=cn_for_tx,
                controlnet_weight=1.0, controlnet_stride=args.controlnet_stride,
                return_dict=False,
            )[0]
            v_pred_f = v_pred.float()

            loss_fm = F.mse_loss(v_pred_f, v_target)
            masks_b = torch.stack(
                [face_masks_latent[int(fi)] for fi in face_idxs_b], dim=0,
            ).to(device=v_pred_f.device, dtype=torch.float32)
            masks_b = masks_b.unsqueeze(1).unsqueeze(1)
            diff2_face = (v_pred_f - v_face_target) ** 2
            denom_face = (masks_b.sum() * v_pred_f.shape[1] *
                          v_pred_f.shape[2]).clamp_min(1.0)
            loss_face = (masks_b * diff2_face).sum() / denom_face
            if args.use_self_distillation and v_pred_ema is not None:
                loss_consistency = F.mse_loss(v_pred_f, v_pred_ema)
            else:
                loss_consistency = torch.zeros((), device=v_pred_f.device)
            loss = (loss_fm
                    + args.lambda_face * loss_face
                    + args.lambda_consistency * loss_consistency)

            if not loss_decomp_asserted:
                # Verify the loss is the literal sum (no hidden weighting),
                # so a future change to the decomposition shows up here.
                expected = (loss_fm
                            + args.lambda_face * loss_face
                            + args.lambda_consistency * loss_consistency)
                assert torch.allclose(loss, expected), \
                    "loss != loss_fm + λ_face·loss_face + λ_cons·loss_consistency"
                assert torch.isfinite(loss_face).item(), "loss_face is non-finite"
                assert loss_face.item() >= 0, f"loss_face negative: {loss_face.item()}"
                loss_decomp_asserted = True
                print(f"[smoke] loss decomp ok: loss_fm={loss_fm.item():.4f} "
                      f"loss_face={loss_face.item():.4f} "
                      f"loss_consistency={loss_consistency.item():.4f} "
                      f"-> total={loss.item():.4f}")

            accelerator.backward(loss)

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
                assert cn_with_grad, "No controlnet param has nonzero grad after micro-step 1"
                grad_assert_done = True
                print("[smoke] grad-flow check passed at micro-step 1")

            peak_mem = torch.cuda.max_memory_allocated() / 1e9
            residual_l2 = mean_residual_l2(cn_states)
            assert torch.isfinite(loss).item(), f"NaN/Inf loss at micro-step {micro_step + 1}"
            assert peak_mem < args.mem_tripwire_gb, (
                f"GPU peak {peak_mem:.2f}GB >= ceiling {args.mem_tripwire_gb}GB"
            )
            assert 0.0 <= residual_l2 < args.residual_l2_ceiling
            micro_step += 1

        if accelerator.sync_gradients:
            torch.nn.utils.clip_grad_norm_(cn.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            ema.update()
            global_step += 1

            torch.cuda.synchronize()
            eff_step_times.append(time.perf_counter() - eff_t0)
            assert_finite_optimizer_state(optimizer, where=f"after eff-step {global_step}")
            ema_step_count = int(ema.step.item())
            assert ema_step_count == global_step
            print(f"[smoke] eff-step {global_step:02d}/{target_eff} "
                  f"(micro={micro_step:02d}/{target_micro}) | "
                  f"loss={loss.item():.4f} | dt={eff_step_times[-1]:.2f}s | "
                  f"peak_mem={peak_mem:.2f}GB | residual_l2={residual_l2:.2e}")

    # ---- Inference smoke ----
    if args.cpu_offload_inference:
        mem_before = torch.cuda.memory_allocated() / 1e9
        print(f"[smoke] enabling pipe.enable_model_cpu_offload() "
              f"(GPU resident before: {mem_before:.2f}GB)")
        pipe.enable_model_cpu_offload()
        # Re-pin the ControlNet to GPU. Under model_cpu_offload, the pipeline's
        # input prep reads pipe.controlnet.device ("cpu") and the resulting
        # tensor doesn't get migrated by accelerate's pre-forward hook, causing
        # a CPU/GPU dtype mismatch in the first Conv3D. Mirrors the fix in
        # training/run_inference_beta.py.
        from accelerate.hooks import remove_hook_from_module
        remove_hook_from_module(pipe.controlnet, recurse=True)
        pipe.controlnet.to(device)
        torch.cuda.empty_cache()
        mem_after = torch.cuda.memory_allocated() / 1e9
        print(f"[smoke] after offload + CN pin: {mem_after:.2f}GB resident")

    if args.inference_controlnet_end is None:
        cn_end_fraction, first_low_idx = _compute_cn_end_high_noise(
            args.base_model_path, args.inference_steps, boundary_ratio, device,
        )
        print(f"[smoke] cn_end (dynamic): σ < {boundary_ratio} first at "
              f"step {first_low_idx}/{args.inference_steps} → cn_end={cn_end_fraction:.4f}")
    else:
        cn_end_fraction = float(args.inference_controlnet_end)

    print(f"[smoke] running 1 inference call (steps={args.inference_steps}) ...")
    cn.eval()
    t0 = time.perf_counter()
    with torch.no_grad():
        generator = torch.Generator().manual_seed(args.seed)
        out = pipe(
            controlnet_frames=[canny_img] * args.num_frames,
            prompt_embeds=prompt_embed,
            negative_prompt_embeds=neg_embeds,
            height=args.height, width=args.width,
            num_frames=args.num_frames,
            num_inference_steps=args.inference_steps,
            guidance_scale=args.inference_guidance_scale,
            controlnet_weight=args.inference_controlnet_weight,
            controlnet_stride=args.controlnet_stride,
            controlnet_guidance_start=0.0,
            controlnet_guidance_end=cn_end_fraction,
            generator=generator,
            output_type="np",
        )
    inf_dt = time.perf_counter() - t0
    frames = out.frames[0]
    assert frames.ndim == 4 and frames.shape[0] == args.num_frames and frames.shape[-1] == 3
    assert np.isfinite(frames).all(), "non-finite values in rendered frames"

    mse = _frames_target_mse(frames, target_hwc)
    ssim_val = _frames_target_ssim(frames, target_hwc, device=str(device))
    assert np.isfinite(mse) and np.isfinite(ssim_val)
    assert -1.0 <= ssim_val <= 1.0
    print(f"[smoke] inference: dt={inf_dt:.1f}s mse={mse:.5f} ssim={ssim_val:.4f}")

    mp4 = smoke_out / f"beta9_smoke_face{rec['face_idx']}_{rec['slug']}.mp4"
    _save_video(frames, mp4, fps=8)
    print(f"[smoke] wrote {mp4}")

    print("")
    if eff_step_times:
        print(f"[smoke] eff-steps={len(eff_step_times)} "
              f"mean={statistics.mean(eff_step_times):.2f}s "
              f"median={statistics.median(eff_step_times):.2f}s")
    print(f"[smoke] final peak GPU mem: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
    print(f"[smoke] face_target={args.face_target_subdir} λ_face={args.lambda_face}")
    print("[smoke] OK")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except AssertionError as e:
        print(f"[smoke] FAIL: {e}", file=sys.stderr)
        sys.exit(1)

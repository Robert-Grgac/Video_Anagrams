---
run_id: beta-003
---

# Training Card — beta-003

## 1. Goal
Test whether warm-starting from `beta-001_final.safetensors` and dedicating the full compute budget to the low-noise expert (no shared training with the high-noise expert, with supervisor's optimizer changes — EMA, grad accum, lower WD, fewer total steps) lets a single shared CN serve both regimes — i.e. fixes the capacity-contention failure mode identified in `beta-002`.

## 2. Hypothesis & success criteria
- **Confirms (Plan A — single-CN):** Plan A inference at `controlnet_weight=1.0, controlnet_guidance_end=1.0` shows visible interior facial features (eyes/nose/mouth in the face region) AND Plan A at `controlnet_weight=1.0, controlnet_guidance_end=0.125` does not regress below `beta-001`'s silhouette quality. The single-CN architecture is then sufficient and capacity-contention was the entire issue in beta-002.
- **Confirms (Dual-CN free regression check):** if Plan A regresses at `cn_end=0.125` but the dual-CN inference (beta-001 frozen high + beta-003 EMA low) preserves the silhouette AND has features at `cn_end=1.0`, drift was the difference; adopt dual-CN inference for this checkpoint.
- **Rejects:** all dual-CN AND single-CN variants still produce brown noise at `cn_end ≥ 0.5, weight=1.0`. Capacity contention was not the only obstacle; pause CN-side iteration and reassess.
- **Quantitative bar:** `loss_at_sigma_lt_0p1_late10pct` < 0.30 (vs beta-002's 0.715); `low_phase_avg_loss` over the last 10% of training < 0.20 (vs beta-002's 0.244 low-phase late-10%). Failure threshold: > 0.50 / > 0.24 respectively.

## 3. Setup

### Base model
- Repo: `Wan-AI/Wan2.2-T2V-A14B-Diffusers`
- Expert(s) trained against: **low-noise** (`transformer_2`) only — no phase alternation
- Other components (frozen): `transformer` (high-noise, used at inference only), VAE, T5

### ControlNet
- Architecture: `WanControlnet` (unchanged); single CN trained, used in both Plan A (single-CN) and Plan B (dual-CN with frozen beta-001 high) inference variants
- Initialization: **Warm start from `$WAN_BETA_CKPT/beta-001_final.safetensors`.** Weights are loaded into a fresh `WanControlnet` instance after `from_config()`; the original beta-001 file on disk is never overwritten. Strict-load asserts 0 missing / 0 unexpected keys.
- Input channels: 3 (Canny edges, RGB-replicated)
- Trainable parameter count: `352593072`
- Gradient checkpointing: ON (and on the active transformer)

### Data
- Source faces: `data/input_faces/` (100 PNGs, 528×528 → resized 512×512)
- Source targets: `data/targets/` (10000 PTDiffusion JPGs at 512×512, named `face_{idx}_{slug}.jpg`)
- Prompt dictionary: `training/input_prompts.py` (`PROMPTS_BATCH_1 | PROMPTS_BATCH_2`)
- Pair count after validation: `10000`
- Canny preprocessing: `cv2.Canny(gray, 100, 200)`, stacked to 3 channels
- Resolution: 512×512
- Frame count `T`: 9 (replicated still → static video)
- Cache directory: `$WORK/wan-beta/cache` (reused from beta-001/beta-002, no re-precompute)

### Optimization
- Loss: flow-matching MSE (`v_target = noise - z_real`)
- Optimizer: `bitsandbytes.optim.AdamW8bit`
- LR: 1e-4, **weight decay 1e-4** (down from beta-002's 1e-2 per supervisor)
- LR schedule: constant (no warmup; single-pass run)
- Micro-batch size: 1; **gradient accumulation steps: 2** ⇒ effective batch `2`
- Grad clipping: 1.0
- Mixed precision: bf16 (fp32 kept for `_keep_in_fp32_modules`)
- **EMA on controlnet weights:** `ema_pytorch.EMA(decay=0.999, update_after_step=100, update_every=1)`. EMA updates on **optimizer** steps, not micro-steps. Final canonical inference checkpoint = `ema.ema_model.state_dict()`; raw weights also saved separately as `_final_raw.safetensors` for safety / debugging.
- Total **effective** steps: 4000 (= 8000 micro-steps)
- Timestep sampling: low-noise regime only — `sigma < boundary_ratio`, uniform within (rule used: `model_index.json.boundary_ratio`; boundary sigma: `0.875`)

### Hardware
- GPUs: 1× NVIDIA A40 (45GB)
- Memory tripwire: 43 GB (raised vs beta-002 to leave room for the fp32 EMA shadow ≈ 1.4 GB)
- Estimated wall-time: 4000 × 2 × ~3.8 s + ~100 s EMA + ~600 s inference smoke ≈ 8.7 h
- Actual wall-time: `0:07:41`
- Cluster / partition: `dmb`

### Logging & checkpointing
- wandb project: `wan-controlnet-beta`, run name: `beta-003`
- wandb URL: `https://wandb.ai/robert-grgac2-university-of-twente/wan-controlnet-beta/runs/lsqzczvy`
- Per-effective-step metrics: loss, grad_norm, lr, controlnet_residual_norm, timestep, sigma, gpu_mem_gb, ema_decay_current
- Init-time fields: `init_mode = warm`, `0` missing keys, `0` unexpected keys
- Periodic checkpoints (EMA): `$WORK/wan-beta/checkpoints/beta-003_step{1000,2000,3000}.safetensors`
- Final checkpoints:
  - `$WORK/wan-beta/checkpoints/beta-003_final.safetensors` — **EMA**, canonical for inference
  - `$WORK/wan-beta/checkpoints/beta-003_final_raw.safetensors` — raw weights, debug
- End-of-run inference smoke (one face, both at `controlnet_weight=1.0, controlnet_guidance_end=1.0`):
  - `beta-003_smoke_planA_e1.mp4` — single-CN pipeline with beta-003 EMA (Plan A)
  - `beta-003_smoke_dualCN_e1.mp4` — dual-CN pipeline with beta-001 high + beta-003 EMA low (regression check)

### Run metadata (auto)
- Status: `failed`
- Started: `2026-05-05T23:52:11+00:00`
- Finished: `2026-05-05T23:59:52+00:00`
- Git SHA: `8f4a4a1`

## 4. Results
*(machine fields auto-filled; qualitative fields human-filled after looking at wandb + inference videos)*
- Final loss: `<AUTO:final_loss — MISSING>`
- Low-phase avg loss: `<AUTO:low_phase_avg_loss — MISSING>`
- Loss at sigma < 0.1, late 10%: `<AUTO:loss_at_sigma_lt_0p1_late10pct — MISSING>`
- GPU peak memory: `<AUTO:gpu_peak_mem_gb — MISSING>` GB
- Effective batch size: `2`
- Init mode: `warm`
- Loss curve descended? (Y/N, monotonic / sawtooth / plateau, EMA vs raw divergence note):
- Plan A inference (single-CN, beta-003 EMA), full sweep grid `weights ∈ {1.0, 2.5} × ends ∈ {0.125, 0.5, 0.875, 1.0}`:
  - `cn_end=0.125, w=1.0` silhouette quality vs beta-001:
  - `cn_end=0.5, w=1.0` (beta-002 brown-noise threshold):
  - `cn_end=1.0, w=1.0` interior features visible?:
  - Other cells worth noting:
- Dual-CN inference (beta-001 high + beta-003 EMA low), full sweep grid:
  - `cn_end=0.125, w=1.0` vs Plan A — drift recovered?:
  - `cn_end=0.5, w=1.0`:
  - `cn_end=1.0, w=1.0` interior features visible?:
- Verdict (Plan A sufficient / adopt dual-CN deployment / try beta-004 cold-start / reject):
- Next action:

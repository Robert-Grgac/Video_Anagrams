---
run_id: beta-004
---

# Training Card — beta-004

## 1. Goal
Test whether a fresh (cold-start) CN, trained dedicated to the low-noise expert with supervisor's optimizer changes (EMA, grad accum, lower WD, fewer total steps) and **no contamination from prior high-phase training**, can produce usable low-phase residuals — paired with the frozen `beta-001_final.safetensors` as the high-CN at inference time. Sibling test to `beta-003` (which warm-starts the same training procedure from beta-001); the two together separate "warm-init drift" from "capacity contention" as causes of beta-002's low-phase failure.

## 2. Hypothesis & success criteria
- **Confirms (Plan B):** dual-CN inference (beta-001 frozen high + beta-004 EMA low) at `controlnet_weight=1.0, controlnet_guidance_end=1.0` shows visible interior facial features (eyes/nose/mouth in the face region). The high-phase silhouette at `cn_end=0.125` is preserved by construction (frozen beta-001). Capacity contention specifically — *not* cold-init per se — was the bottleneck in beta-002.
- **Rejects:** brown noise at `cn_end ≥ 0.5, weight=1.0` matching beta-002's failure pattern. Cold-start cannot escape low-phase regression even with dedicated capacity; the regime is fundamentally unfixable with replicated stills + this loss formulation.
- **Quantitative bar:** `loss_at_sigma_lt_0p1_late10pct` < 0.30 (vs beta-002's 0.715); `low_phase_avg_loss` over the last 10% of training < 0.20 (vs beta-002's 0.244 low-phase late-10%). Failure threshold: > 0.50 / > 0.24 respectively. Cold-start may show a still-descending curve at step 4000; if so, follow up by extending to ≥ 10k steps.

## 3. Setup

### Base model
- Repo: `Wan-AI/Wan2.2-T2V-A14B-Diffusers`
- Expert(s) trained against: **low-noise** (`transformer_2`) only — no phase alternation
- Other components (frozen): `transformer` (high-noise, used at inference only via the dual pipeline), VAE, T5

### ControlNet
- Architecture: `WanControlnet` (unchanged); single CN trained, used as the low-noise CN in the dual-CN inference variant (high-noise CN at inference is frozen `beta-001_final.safetensors`)
- Initialization: **Cold start.** Architecture config loaded from `TheDenk/wan2.2-t2v-a14b-controlnet-hed-v1` (`config.json` only); weights freshly initialized. Output projections zeroed by `zero_module()` → residuals = 0 at step 0.
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
- Actual wall-time: `0:06:54`
- Cluster / partition: `dmb`

### Logging & checkpointing
- wandb project: `wan-controlnet-beta`, run name: `beta-004`
- wandb URL: `https://wandb.ai/robert-grgac2-university-of-twente/wan-controlnet-beta/runs/b53xdr8b`
- Per-effective-step metrics: loss, grad_norm, lr, controlnet_residual_norm, timestep, sigma, gpu_mem_gb, ema_decay_current
- Init-time fields: `init_mode = cold` (no warm-start arg passed)
- Periodic checkpoints (EMA): `$WORK/wan-beta/checkpoints/beta-004_step{1000,2000,3000}.safetensors`
- Final checkpoints:
  - `$WORK/wan-beta/checkpoints/beta-004_final.safetensors` — **EMA**, canonical for inference
  - `$WORK/wan-beta/checkpoints/beta-004_final_raw.safetensors` — raw weights, debug
- End-of-run inference smoke (one face, `controlnet_weight=1.0, controlnet_guidance_end=1.0`):
  - `beta-004_smoke_planB_e1.mp4` — dual-CN pipeline with beta-001 high + beta-004 EMA low (Plan B canonical)

### Run metadata (auto)
- Status: `failed`
- Started: `2026-05-06T00:00:31+00:00`
- Finished: `2026-05-06T00:07:25+00:00`
- Git SHA: `8f4a4a1`

## 4. Results
*(machine fields auto-filled; qualitative fields human-filled after looking at wandb + inference videos)*
- Final loss: `<AUTO:final_loss — MISSING>`
- Low-phase avg loss: `<AUTO:low_phase_avg_loss — MISSING>`
- Loss at sigma < 0.1, late 10%: `<AUTO:loss_at_sigma_lt_0p1_late10pct — MISSING>`
- GPU peak memory: `<AUTO:gpu_peak_mem_gb — MISSING>` GB
- Effective batch size: `2`
- Init mode: `cold`
- Loss curve descended? (Y/N, monotonic / sawtooth / plateau, still-descending-at-end note, EMA vs raw divergence note):
- Dual-CN inference (beta-001 high + beta-004 EMA low), full sweep grid `weights ∈ {1.0, 2.5} × ends ∈ {0.125, 0.5, 0.875, 1.0}`:
  - `cn_end=0.125, w=1.0` (high-phase guaranteed by frozen beta-001):
  - `cn_end=0.5, w=1.0` (beta-002 brown-noise threshold):
  - `cn_end=1.0, w=1.0` interior features visible?:
  - Other cells worth noting:
- Comparison vs beta-003 dual-CN at the same grid (cold-start vs warm-start same checkpoint family):
- Verdict (Plan B viable / extend training / fall back to beta-003 / reject low-phase entirely):
- Next action:

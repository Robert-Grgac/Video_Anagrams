---
run_id: beta-002
---

# Training Card — beta-002

## 1. Goal
Verify whether also training the low-noise expert (`transformer_2`), via a phase-alternating loop that swaps the active expert every K=500 steps, lets interior facial features (eyes, nose, mouth) survive into the rendered painting style — i.e. fixes the silhouette-but-no-features failure mode observed in `beta-001`.

## 2. Hypothesis & success criteria
- **Confirms:** at `controlnet_weight ≤ 1.5` and `controlnet_guidance_end = 1.0` the inference video shows recognizable eyes/nose/mouth in the face region, while the abandoned-factory-style background remains coherent. The `controlnet_guidance_end = 0.125` baseline video at `controlnet_weight = 1.0` is at least as good at silhouette as `beta-001`'s.
- **Rejects:** features remain absent at every `controlnet_weight ≤ 2.5` for either guidance setting, OR the silhouette quality regresses below `beta-001` at `controlnet_guidance_end = 0.125` (capacity-contention failure: a single shared controlnet can't serve both experts; next ablation is parameter-isolated dual controlnet).
- **Quantitative bar:** `high_phase_avg_loss` over the last 10% of training is no worse than 110% of `beta-001`'s late-10% loss (≈ 0.152 → ceiling ≈ 0.167). Above that ceiling, high-noise specialization has measurably degraded.

## 3. Setup

### Base model
- Repo: `Wan-AI/Wan2.2-T2V-A14B-Diffusers`
- Expert(s) trained against: **both** — `transformer` (high-noise) and `transformer_2` (low-noise), phase-alternating with K=500 steps per phase, starting with high-noise
- Other components (frozen): VAE, T5

### ControlNet
- Architecture: `WanControlnet` (unchanged); single controlnet shared across both experts (matches inference pipeline)
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
- Cache directory: `$WORK/wan-beta/cache` (reused from beta-001, no re-precompute)

### Optimization
- Loss: flow-matching MSE (`v_target = noise - z_real`)
- Optimizer: `bitsandbytes.optim.AdamW8bit` (state persisted across expert swaps)
- LR: 1e-4, weight decay 0.01
- LR schedule: constant (no warmup; single-pass run)
- Batch size: 1, no grad accumulation
- Grad clipping: 1.0
- Mixed precision: bf16 (fp32 kept for `_keep_in_fp32_modules`)
- Total steps: 10000 (20 phases × 500 = 10 cycles ⇒ 5000 steps per expert)
- Phase-conditional timestep sampling: `sigma >= boundary` in high-noise phase, `sigma < boundary` in low-noise phase, uniform within each (rule used: `model_index.json.boundary_ratio`; boundary sigma: `0.875`)
- Swap mechanism: `del transformer; gc.collect(); torch.cuda.empty_cache(); from_pretrained(other_subfolder, ...)` — no transformer state saved, just reloaded from local cache

### Hardware
- GPUs: 1× NVIDIA A40 (45GB)
- Estimated wall-time: 10000 × ~3.8s + 19 × ~45s ≈ 10.6h + ~14 min swap overhead
- Actual wall-time: `11:06:27`
- Cluster / partition: `dmb`

### Logging & checkpointing
- wandb project: `wan-controlnet-beta`, run name: `beta-002`
- wandb URL: `https://wandb.ai/robert-grgac2-university-of-twente/wan-controlnet-beta/runs/uc0dagje`
- Per-step metrics: loss, grad_norm, lr, controlnet_residual_norm, timestep, sigma, gpu_mem_gb, **active_expert** (0=high, 1=low), **phase_step**, **cycle_idx**
- Per-swap event: `{event: phase_swap, swap_from, swap_to, swap_seconds, step}`
- Periodic checkpoints: `$WORK/wan-beta/checkpoints/beta-002_step{2000,4000,...}.safetensors`
- Final checkpoint: `$WORK/wan-beta/checkpoints/beta-002_final.safetensors`
- End-of-run inference: 1 (face, prompt) pair, **two videos** at `controlnet_weight=1.0`:
  - `beta-002_final_e0p125.mp4` — `controlnet_guidance_end=0.125` (sanity vs beta-001)
  - `beta-002_final_e1.mp4`     — `controlnet_guidance_end=1.0` (the new capability)

### Run metadata (auto)
- Status: `completed`
- Started: `2026-05-04T21:13:14+00:00`
- Finished: `2026-05-05T08:48:46+00:00`
- Git SHA: `2063cbf`

## 4. Results
*(machine fields auto-filled; qualitative fields human-filled after looking at wandb + inference videos)*
- Final loss: `0.154749`
- High-phase avg loss: `0.189664`
- Low-phase avg loss: `0.256114`
- Number of phase swaps: `19`
- Total swap overhead (seconds): `1291.1`
- GPU peak memory: `32.59` GB
- Loss curve descended? (Y/N, with note on per-phase trend and any sawtooth at swaps):
- Inference @ `cn_end=0.125, weight=1.0` — silhouette quality vs beta-001:
- Inference @ `cn_end=1.0, weight=1.0` — interior features visible? prompt content coherent?:
- Verdict (proceed to full pipeline / try longer K / try parameter-isolated dual controlnet / reject approach):
- Next action:

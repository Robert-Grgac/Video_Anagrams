---
run_id: beta-001
---

# Training Card — beta-001

## 1. Goal
Verify that a cold-init `WanControlnet` learns *any* signal at all when trained on (Canny-of-face → PTDiffusion-still) static-video pairs against the Wan 2.2 A14B high-noise expert.

## 2. Hypothesis & success criteria
- **Confirms:** flow-matching loss shows visible descent after the expected ~1k–3k step zero-init plateau, AND the end-of-run inference sample shows recognizable face structure peeking through the prompt-driven texture (qualitative; ≥ 1 of 1 sample is a low bar but the whole BETA is a single run).
- **Rejects:** loss is flat for the entire run (>> 3k steps without descent) and the inference video shows no structural correlation with the input face. In that case the next step is `beta-002` with warm-start from `TheDenk/wan2.2-t2v-a14b-controlnet-hed-v1` to disambiguate "approach broken" from "cold-init residuals never woke up".
- **Quantitative bar:** final FM loss ≤ 90% of the median of the first 100 steps. Below this threshold we treat the curve as visually descending; above is "flat".

## 3. Setup

### Base model
- Repo: `Wan-AI/Wan2.2-T2V-A14B-Diffusers`
- Expert(s) trained against: high-noise (`transformer`)
- Other components (frozen): `transformer_2`, VAE, T5

### ControlNet
- Architecture: `WanControlnet` (unchanged)
- Initialization: **Cold start.** Architecture config loaded from `TheDenk/wan2.2-t2v-a14b-controlnet-hed-v1` (`config.json` only); weights freshly initialized. Output projections zeroed by `zero_module()` → residuals = 0 at step 0.
- Input channels: 3 (Canny edges, RGB-replicated)
- Trainable parameter count: `352593072`
- Gradient checkpointing: ON

### Data
- Source faces: `data/input_faces/` (100 PNGs, 528×528 → resized 512×512)
- Source targets: `data/targets/` (10000 PTDiffusion JPGs at 512×512, named `face_{idx}_{slug}.jpg`)
- Prompt dictionary: `training/input_prompts.py` (`PROMPTS_BATCH_1 | PROMPTS_BATCH_2`)
- Pair count after validation: `10000`
- Canny preprocessing: `cv2.Canny(gray, 100, 200)`, stacked to 3 channels
- Resolution: 512×512
- Frame count `T`: 9 (replicated still → static video)
- Cache directory: `$WORK/wan-beta/cache`, `4.138` GB on disk
- VAE round-trip MSE (gate): `0.0036418151576071978`

### Optimization
- Loss: flow-matching MSE (`v_target = noise - z_real`)
- Optimizer: `bitsandbytes.optim.AdamW8bit`
- LR: 1e-4, weight decay 0.01
- LR schedule: constant (no warmup; single-pass run)
- Batch size: 1, no grad accumulation
- Grad clipping: 1.0
- Mixed precision: bf16 (fp32 kept for norms / `time_embedder` / `scale_shift_table` per `_keep_in_fp32_modules`)
- Timestep sampling: high-noise regime only (rule used: `model_index.json.boundary_ratio`; boundary sigma: `0.875`)

### Hardware
- GPUs: 1× NVIDIA A40 (45GB)
- Smoke-test per-step wall-time (median / p90): `3.80`s / `3.81`s
- Estimated wall-time (10000 steps @ smoke median): `10:33:30`
- Actual wall-time: `10:42:27`
- Cluster / partition: `dmb`

### Logging & checkpointing
- wandb project: `wan-controlnet-beta`, run name: `beta-001`
- wandb URL: `https://wandb.ai/robert-grgac2-university-of-twente/wan-controlnet-beta/runs/l64hi0jk`
- Per-step metrics: loss, grad_norm, lr, controlnet_residual_norm, timestep, sigma, gpu_mem_gb
- Periodic checkpoints: `$WORK/wan-beta/checkpoints/beta-001_step{2000,4000,...}.safetensors`
- Final checkpoint: `$WORK/wan-beta/checkpoints/beta-001_final.safetensors`
- End-of-run inference: 1 (face, prompt) pair → `$WORK/wan-beta/outputs/beta-001_final.mp4`

### Run metadata (auto)
- Status: `completed`
- Started: `2026-05-03T23:20:10+00:00`
- Finished: `2026-05-04T10:32:06+00:00`
- Git SHA: `4f4bb25`

## 4. Results
*(machine fields auto-filled; qualitative fields human-filled after looking at wandb + inference video)*
- Final loss: `0.08255`
- GPU peak memory: `32.59` GB
- Loss curve descended? (Y/N, with note on when descent began):
- Inference sample observations (does the face structure show through? does prompt content appear?):
- Verdict (proceed to full pipeline / try warm-start next / reject approach):
- Next action:

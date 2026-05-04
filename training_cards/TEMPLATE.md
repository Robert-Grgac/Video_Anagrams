---
run_id: <RUN_ID>
---

# Training Card — <RUN_ID>

## 1. Goal
*(one sentence — what specific question does this run answer?)*

## 2. Hypothesis & success criteria
- **Confirms:**
- **Rejects:**
- **Quantitative bar:**

## 3. Setup

### Base model
- Repo: `Wan-AI/Wan2.2-T2V-A14B-Diffusers`
- Expert(s) trained against: high-noise (`transformer`)
- Other components (frozen): `transformer_2`, VAE, T5

### ControlNet
- Architecture: `WanControlnet` (unchanged)
- Initialization: *(cold start / warm start from <repo>)*
- Input channels: 3 (Canny edges, RGB-replicated)
- Trainable parameter count: `<AUTO:trainable_params>`
- Gradient checkpointing: ON

### Data
- Source faces: `data/input_faces/`
- Source targets: `data/targets/`
- Prompt dictionary: `training/input_prompts.py` (`PROMPTS_BATCH_1 | PROMPTS_BATCH_2`)
- Pair count after validation: `<AUTO:pair_count>`
- Canny preprocessing: `cv2.Canny(gray, 100, 200)`, stacked to 3 channels
- Resolution: 512×512
- Frame count `T`: 9 (replicated still → static video)
- Cache directory: `$WORK/wan-beta/cache`, `<AUTO:cache_disk_gb>` GB on disk
- VAE round-trip MSE (gate): `<AUTO:smoke_latent_roundtrip_mse>`

### Optimization
- Loss: flow-matching MSE (`v_target = noise - z_real`)
- Optimizer: `bitsandbytes.optim.AdamW8bit`
- LR: 1e-4, weight decay 0.01
- LR schedule: constant (no warmup; single-pass)
- Batch size: 1, no grad accumulation
- Grad clipping: 1.0
- Mixed precision: bf16 (fp32 kept for `_keep_in_fp32_modules`)
- Timestep sampling: high-noise regime only (rule used: `<AUTO:high_noise_rule>`; boundary sigma: `<AUTO:boundary_sigma>`)

### Hardware
- GPUs: 1× NVIDIA A40 (45GB)
- Smoke-test per-step wall-time (median / p90): `<AUTO:smoke_step_median>`s / `<AUTO:smoke_step_p90>`s
- Estimated wall-time (10000 steps @ smoke median): `<AUTO:smoke_projected_wall_time>`
- Actual wall-time: `<AUTO:wall_time>`
- Cluster / partition: `<AUTO:cluster_partition>`

### Logging & checkpointing
- wandb project: `wan-controlnet-beta`, run name: `<RUN_ID>`
- wandb URL: `<AUTO:wandb_url>`
- Per-step metrics: loss, grad_norm, lr, controlnet_residual_norm, timestep, sigma, gpu_mem_gb
- Periodic checkpoints: `$WORK/wan-beta/checkpoints/<RUN_ID>_step{2000,4000,...}.safetensors`
- Final checkpoint: `$WORK/wan-beta/checkpoints/<RUN_ID>_final.safetensors`
- End-of-run inference: 1 (face, prompt) pair → `$WORK/wan-beta/outputs/<RUN_ID>_final.mp4`

### Run metadata (auto)
- Status: `<AUTO:status>`
- Started: `<AUTO:date_started>`
- Finished: `<AUTO:date_finished>`
- Git SHA: `<AUTO:git_sha>`

## 4. Results
*(machine fields auto-filled; qualitative fields human-filled after looking at wandb + inference video)*
- Final loss: `<AUTO:final_loss>`
- GPU peak memory: `<AUTO:gpu_peak_mem_gb>` GB
- Loss curve descended? (Y/N, with note on when descent began):
- Inference sample observations (does the face structure show through? does prompt content appear?):
- Verdict (proceed to full pipeline / try warm-start next / reject approach):
- Next action:

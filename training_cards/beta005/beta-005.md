---
run_id: beta-005
---

# Training Card — beta-005

## 1. Goal
Re-train the high-noise expert's ControlNet from scratch under the supervisor's modern recipe (EMA, AdamW8bit with low WD, cold-init `zero_module`, much larger effective batch) so we have a high-CN trained under the same protocol as beta-004's low-CN — replacing the legacy beta-001 as the canonical high-CN for any future dual-CN inference. Sibling test to `beta-006` (same recipe applied to the low-noise expert on A40).

## 2. Hypothesis & success criteria
- **Confirms:** silhouette is visible in the end-of-run single-CN smoke at `controlnet_weight=1.0` (beta-001 needed weight≥2.5 — see `training_cards/beta001/beta-001.md`); larger eff_batch + EMA should let the trained residuals carry more signal at unit weight. Loss curve descends monotonically once smoothed (`loss_ema` window 20).
- **Rejects:** loss plateaus immediately or rises; smoke video shows no recognizable face contour at weight=1.0; `controlnet_residual_norm` fails to grow above ~0.3 by the end (beta-001 ended at 0.57 with 80× more updates per σ-unit, so a lower endpoint is acceptable here).
- **Quantitative bar:** `loss_ema_final ≤ 0.18` (beta-001's σ=0.875–0.8875 bin mean was 0.150 — the cleanest reference because beta-001 sampled this band heavily). Failure threshold: `loss_ema_final > 0.25` AND no descent visible after smoothing.

## 3. Setup

### Base model
- Repo: `Wan-AI/Wan2.2-T2V-A14B-Diffusers`
- Expert(s) trained against: high-noise (`transformer`)
- Other components (frozen): `transformer_2`, VAE, T5

### ControlNet
- Architecture: `WanControlnet` (unchanged)
- Initialization: **Cold start.** Architecture config from `TheDenk/wan2.2-t2v-a14b-controlnet-hed-v1` (`config.json` only); fresh `from_config` weights with `zero_module()`-zeroed output projections.
- Input channels: 3 (Canny edges, RGB-replicated)
- Trainable parameter count: `352593072`
- Gradient checkpointing: ON (and on the active transformer)

### Data
- Source faces: `data/input_faces/` (100 PNGs, 528×528 → 512×512)
- Source targets: `data/targets/` (10000 PTDiffusion JPGs at 512×512)
- Prompt dictionary: `training/input_prompts.py` (`PROMPTS_BATCH_1 | PROMPTS_BATCH_2`)
- Pair count after validation: `10000`
- Canny preprocessing: `cv2.Canny(gray, 100, 200)`, stacked to 3 channels
- Resolution: 512×512
- Frame count `T`: 9 (replicated still → static video)
- Cache directory: `$WORK/wan-beta/cache` (reused from beta-001/002/004; no re-precompute)

### Optimization
- Loss: flow-matching MSE (`v_target = noise - z_real`)
- Optimizer: `bitsandbytes.optim.AdamW8bit`
- LR: 1e-4, **weight decay 1e-4** (matches beta-004; down from beta-001's 1e-2)
- LR schedule: constant (no warmup; single-pass run)
- **Effective batch:** `32` = micro_batch=`8` × gradient_accumulation_steps=`4`. Per-micro-batch we sample 8 independent σ in the high-noise band so the eff_batch averages over both faces and σ.
- **Total optimizer steps:** `125` (= 4000 micro-steps; mentor's "4000/32" budget).
- Grad clipping: 1.0
- Mixed precision: bf16 (fp32 kept for `_keep_in_fp32_modules`)
- **EMA on controlnet weights:** `ema_pytorch.EMA(decay=0.999, update_after_step=100, update_every=1)`. EMA updates on optimizer steps only. Final canonical inference checkpoint = `ema.ema_model.state_dict()`; raw weights saved separately as `_final_raw.safetensors`.
- Timestep sampling: high-noise regime only — `sigma >= boundary_ratio`, uniform within (rule used: `model_index.json.boundary_ratio`; boundary sigma: `0.875`)

### Hardware
- GPU: 1× **NVIDIA RTX PRO 6000 Blackwell Server Edition (96 GB, compute_cap 12.0)** on `hpc-node31`
- Attention backend: **NATIVE SDPA** forced via both `DIFFUSERS_ATTN_BACKEND=native` and in-script `set_attention_backend("native")`. Reason: flash-attn imports cleanly on `wan22-bw` but `flash_attn_func` is None at runtime on SM 12.0 — this killed beta-004's first attempt (see `logs/wan-beta4-train-492350.err`).
- Memory tripwire: 90 GB
- Conda env: `wan22-bw` (no flash-attn)
- Smoke-test per-step wall-time (median / p90): `<AUTO:smoke_step_median — MISSING>`s / `<AUTO:smoke_step_p90 — MISSING>`s
- Estimated wall-time (125 effective steps): `<AUTO:smoke_projected_wall_time — MISSING>`
- Actual wall-time: `01:19:09`
- Cluster / partition: `dmb`

### Logging & checkpointing
- wandb project: `wan-controlnet-beta`, run name: `beta-005`
- wandb URL: `https://wandb.ai/robert-grgac2-university-of-twente/wan-controlnet-beta/runs/l6xpx8nj`
- Per-effective-step metrics: `loss`, **`loss_ema`** (window=20), `grad_norm`, `lr`, `controlnet_residual_norm`, `timestep`, `sigma` (mean over the 32 samples), `sigma_std`, `gpu_mem_gb`, `ema_decay_current`, `samples_seen`
- Periodic checkpoints (EMA): `$WORK/wan-beta/checkpoints/beta-005_step{50,100}.safetensors`
- Final checkpoints:
  - `$WORK/wan-beta/checkpoints/beta-005_final.safetensors` — **EMA**, canonical for inference
  - `$WORK/wan-beta/checkpoints/beta-005_final_raw.safetensors` — raw weights, debug
- End-of-run inference smoke (one face, single-CN pipeline, `controlnet_weight=1.0`):
  - `beta-005_smoke_singleCN_e1.mp4`

### Run metadata (auto)
- Status: `completed`
- Started: `2026-05-07T00:44:33+00:00`
- Finished: `2026-05-07T02:18:08+00:00`
- Git SHA: `8f4a4a1`

## 4. Results
*(machine fields auto-filled; qualitative fields human-filled after looking at wandb + inference video)*
- Final loss (last effective step, raw): `0.179671`
- Final loss EMA (window 20): `0.187789`
- High-phase avg loss (over all eff steps): `0.210092`
- GPU peak memory: `51.57` GB (90 GB tripwire)
- Effective batch size: `32` (= micro_batch × accum)
- Init mode: `cold`
- GPU: `NVIDIA RTX PRO 6000 Blackwell Server Edition` (compute_cap `12.0`)
- Loss curve descended? (Y/N, `loss_ema` shape — monotonic / sawtooth / plateau / still-descending-at-end):
- Single-CN smoke at `weight=1.0`:
  - silhouette visible? prompt content visible? compare against beta-001's smoke at the same weight (which showed nothing at weight=1.0):
- Comparison vs beta-001's `loss_by_sigma_bin` for sigma ∈ [0.875, 1.0]:
- Verdict (replace beta-001 as canonical high-CN / extend to more steps / fall back to beta-001):
- Next action:
- **Pre-flagged risk:** 125 optimizer steps from cold-init is fewer Adam updates than beta-001 (10000) or beta-004 (4000). The eff_batch=32 buys cleaner gradients (~5.7× lower per-step variance) but only ~75 productive updates after the EMA wake-up. If `loss_ema` is still descending at step 125, the obvious extension is `max_steps=250` or `500`.

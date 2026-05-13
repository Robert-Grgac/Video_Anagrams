---
run_id: beta-006
---

# Training Card — beta-006

## 1. Goal
Re-train the low-noise expert's ControlNet from scratch under the same supervisor recipe as beta-005, but on A40 with pure gradient accumulation (no batching, VRAM-locked). Sibling test to beta-005 (high-noise on Blackwell with batching). The pair gives us the first end-to-end "modern recipe" CN for both experts and replaces the beta-001 ⊕ beta-004 mix that was inconsistent across regimes.

## 2. Hypothesis & success criteria
- **Confirms:** `loss_ema` descends monotonically over the 125 effective steps; the dual-CN smoke (beta-005 high + beta-006 EMA low at `cn_end=1.0, weight=1.0`) shows interior facial features in the face region — the failure mode beta-002 demonstrated and beta-004 partially mitigated. Specifically, the σ < 0.1 bin loss should remain bounded (< 0.50 mean), confirming the low-σ regime didn't catastrophically diverge with the larger eff_batch.
- **Rejects:** brown-noise dual-CN output at `cn_end ≥ 0.5, weight=1.0` matching beta-002's failure pattern, OR `loss_ema_final > beta-004's loss_ema_final * 1.1` (i.e. larger eff_batch made things worse, not better).
- **Quantitative bar:** `low_phase_avg_loss < 0.22` (beta-004 was 0.244; this is a modest 10% improvement bar appropriate for fewer total Adam updates). Failure threshold: `> 0.30`. Pre-flagged risk: 125 optimizer steps is much fewer than beta-004's 4000; if `loss_ema` is still descending, the natural extension is `max_steps=250–500`.

## 3. Setup

### Base model
- Repo: `Wan-AI/Wan2.2-T2V-A14B-Diffusers`
- Expert(s) trained against: **low-noise** (`transformer_2`)
- Other components (frozen): `transformer` (used at dual-CN smoke only), VAE, T5

### ControlNet
- Architecture: `WanControlnet` (unchanged); single CN trained, slotted as the low-CN at dual-CN inference.
- Initialization: **Cold start.** Architecture config from `TheDenk/wan2.2-t2v-a14b-controlnet-hed-v1`; fresh `from_config` weights with `zero_module()`-zeroed output projections. **No warm-start** (compare beta-003 in `train_beta3.py`, which is the warm-start variant of this recipe).
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
- Cache directory: `$WORK/wan-beta/cache` (reused; no re-precompute)

### Optimization
- Loss: flow-matching MSE (`v_target = noise - z_real`)
- Optimizer: `bitsandbytes.optim.AdamW8bit`
- LR: 1e-4, weight decay 1e-4 (identical to beta-004 / beta-005)
- LR schedule: constant (no warmup; single-pass run)
- **Effective batch:** `32` = micro_batch=`1` × gradient_accumulation_steps=`32`. Per-micro-step we sample 1 σ in the low-noise band; over the accumulation, 32 independent σ are averaged into each Adam step.
- **Total optimizer steps:** `125` (= 4000 micro-steps; mentor's "4000/32" budget — same as beta-005 for fair comparison).
- Grad clipping: 1.0
- Mixed precision: bf16 (fp32 kept for `_keep_in_fp32_modules`)
- **EMA on controlnet weights:** `ema_pytorch.EMA(decay=0.999, update_after_step=100, update_every=1)`. EMA updates on optimizer steps only. Final canonical inference checkpoint = `ema.ema_model.state_dict()`; raw weights saved separately as `_final_raw.safetensors`.
- Timestep sampling: low-noise regime only — `sigma < boundary_ratio`, uniform within (rule used: `model_index.json.boundary_ratio`; boundary sigma: `0.875`)

### Hardware
- GPU: 1× **NVIDIA A40 (45 GB, compute_cap 8.6)** on `ctit092`
- Attention backend: **flash-attn** via `DIFFUSERS_ATTN_BACKEND=flash`. The Blackwell guard inside the training script is a no-op at compute_cap 8.6.
- Memory tripwire: 43 GB
- Conda env: `wan22`
- Smoke-test per-step wall-time (median / p90): `<AUTO:smoke_step_median — MISSING>`s / `<AUTO:smoke_step_p90 — MISSING>`s
- Estimated wall-time (125 effective steps × 32 micro-steps × ~3.8 s): ~4.2 h
- Actual wall-time: `04:14:51`
- Cluster / partition: `dmb`

### Logging & checkpointing
- wandb project: `wan-controlnet-beta`, run name: `beta-006`
- wandb URL: `https://wandb.ai/robert-grgac2-university-of-twente/wan-controlnet-beta/runs/owr1eefx`
- Per-effective-step metrics: `loss`, **`loss_ema`** (window=20), `grad_norm`, `lr`, `controlnet_residual_norm`, `timestep`, `sigma` (mean over the 32 samples), `sigma_std`, `gpu_mem_gb`, `ema_decay_current`, `samples_seen`
- Periodic checkpoints (EMA): `$WORK/wan-beta/checkpoints/beta-006_step{50,100}.safetensors`
- Final checkpoints:
  - `$WORK/wan-beta/checkpoints/beta-006_final.safetensors` — **EMA**, canonical for inference
  - `$WORK/wan-beta/checkpoints/beta-006_final_raw.safetensors` — raw weights, debug
- End-of-run inference smoke (one face, dual-CN pipeline, `controlnet_weight=1.0, controlnet_guidance_end=1.0`):
  - `beta-006_smoke_dualCN_e1.mp4` — high-CN = beta-005 if available else beta-001 (legacy fallback)

### Run metadata (auto)
- Status: `completed`
- Started: `2026-05-07T00:44:37+00:00`
- Finished: `2026-05-07T05:19:39+00:00`
- Git SHA: `8f4a4a1`

## 4. Results
*(machine fields auto-filled; qualitative fields human-filled after looking at wandb + inference video)*
- Final loss (last effective step, raw): `0.219183`
- Final loss EMA (window 20): `0.241856`
- Low-phase avg loss (over all eff steps): `0.29635`
- Loss at sigma < 0.1, late 10%: `nan`
- GPU peak memory: `34.01` GB (43 GB tripwire)
- Effective batch size: `32`
- Init mode: `cold`
- GPU: `NVIDIA A40` (compute_cap `8.6`)
- Loss curve descended? (Y/N, `loss_ema` shape — monotonic / sawtooth / plateau / still-descending-at-end, EMA vs raw divergence note):
- Dual-CN smoke at `weight=1.0, cn_end=1.0`:
  - interior features visible? brown-noise present? compare against beta-004's `cn_end=1.0` failure:
- Comparison vs beta-004's `loss_by_sigma_bin` (especially the σ < 0.1 floor at 0.480):
- Comparison vs beta-005 (sibling: same recipe, opposite expert, batching variant):
- Verdict (Plan B viable with modern recipe / extend training / fall back / reject low-phase):
- Next action:
- **Pre-flagged risk:** identical eff_batch=32 to beta-005 means a head-to-head sample-efficiency comparison: same total compute, same recipe, only the σ regime differs. If beta-006 lags beta-005 markedly even on `ratio_last_over_first`, it's evidence the low-σ regime is intrinsically harder to learn from canny conditioning (predicted in beta-004's σ < 0.1 floor analysis).

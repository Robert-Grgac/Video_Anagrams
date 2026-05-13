---
run_id: beta-007
---

> **⚠️ Superseded by [beta-007_v2](../beta007_v2/beta-007_v2.md).**
> The in-training periodic + final eval videos for this run were *not* a fair measurement of the trained CN. The precompute cache padded T5 prompt embeddings to **512** but the Wan transformer was pretrained on **226** — the 286 extra zero-padded tokens silently dilute cross-attention, producing noise outputs at eval. Training itself was internally consistent (everything at 512), which is why loss descended normally; but the CN learned to compensate for poisoned conditioning that the standalone inference pipeline does not produce.
> Diagnosis path: a `controlnet_weight` sweep on `beta-007_final.safetensors` via the standalone pipeline produced coherent prompt scenes (no face structure), while a one-shot test passing the 512-length cached prompt + a fresh 512-length negative reproduced the in-training noise → bug confirmed.
> Wandb run, checkpoints, eval JSON, and inference videos for this run are preserved as-is for reference. The rerun is documented in [beta-007_v2](../beta007_v2/beta-007_v2.md).

# Training Card — beta-007

## 1. Goal
**Rerun of the original beta-007** (see `training_cards/beta007/beta-007.md` history in git for the discarded first attempt). Same training recipe — Accelerate-driven gradient accumulation, EMA(decay=0.99, after=10), self-distillation, lr=5e-5, 309 effective steps over 9900 train records on the high-noise expert — with two protocol bugs fixed and two analysis upgrades:

**Bug fixes** (the original run's outputs were unusable because of these):
1. **CN injected into the high-noise expert only.** The `WanTextToVideoControlnetPipeline` gates CN computation by step-fraction `i/N`, *not* by σ. Setting `controlnet_guidance_end=1.0` (original beta-007) caused the CN to inject residuals into `transformer_2` (low-noise expert) for the ~88% of inference steps where σ < boundary_ratio — but `transformer_2` was never trained with the CN, so the residuals corrupted the late-σ denoising into pure noise. **Fix:** `train_beta7.py` now computes `controlnet_guidance_end` dynamically from the scheduler's σ trajectory (`_compute_cn_end_high_noise`) — passing the step fraction at which σ first drops below `boundary_ratio`, so the CN runs only while `transformer` (high-noise) is the active expert.
2. **Periodic eval uses TRAINING-set samples** (overfit check), not held-out eval samples. The original run watched 10 held-out samples and asked "is it generalizing?", which was the wrong question at 309 cold-start steps. We instead watch 10 samples the model *does* train on and ask "is it fitting *anything*?". The held-out 100-sample eval is retained for the final-eval pass only.

**New additions:**
3. **SSIM logged alongside pixel MSE** for every periodic and final eval sample. Pixel MSE in [0,1] is poorly calibrated (a noise-vs-image MSE of ~0.116 looks "low" but is actually noise-floor); SSIM gives a perceptually-aligned second axis with a wide gap between "noise" (~0.0) and "structurally similar" (>0.3). Computed via `pytorch-msssim` on GPU.
4. **Face/prompt-stratified Latin-pair splits** for both periodic and final eval. The original run's "last 100 manifest records" eval was dominated by face_idx=9 (the manifest is face-sorted) — useless for variety. Now: every sample inside each eval set has a distinct face AND a distinct slug.

## 2. Hypothesis & success criteria
- **Confirms:**
  - Periodic-eval `eval/mse_avg` *descends* across the 30 periodic rounds (step 10 → step 300) — the original beta-007 saw it *rise* from 0.113 → 0.134, which we now attribute to the CN-into-low-noise bug, not to the training itself.
  - Periodic-eval `eval/ssim_avg` *rises* across rounds. A descending MSE without rising SSIM would indicate luminance fitting without structural fitting.
  - `loss_ema_final ≤ 0.20` (matches the original run's `0.180` — the training loss itself was reasonable; only inference was poisoned).
  - Final-eval `eval_final/mse_avg < eval/mse_avg` of the first periodic round (step 10), by ≥ 20%. Soft target: `eval_final/ssim_avg ≥ 0.10` (noise scores ~0.0).
- **Rejects:**
  - Periodic-eval MSE is flat or rising → CN still not fitting *trained* samples after 309 steps; cold-start budget is too small for this recipe (matches CLAUDE.md's "1k–3k step wake-up" expectation).
  - Periodic and final eval MSE/SSIM are similar → no train/eval gap → either the model isn't learning, or 309 steps is too few to overfit even on samples it sees.
  - Inference videos still pure noise despite the cn_end fix → the failure mode is something other than the CN-into-transformer_2 injection (e.g., latent-norm bug, scheduler mismatch).
- **Quantitative bar:** `loss_ema_final ≤ 0.20`; `eval/mse_avg` at step 300 < `eval/mse_avg` at step 10; `eval/ssim_avg` at step 300 > step 10 by ≥ 0.05.

## 3. Setup

### Base model
- Repo: `Wan-AI/Wan2.2-T2V-A14B-Diffusers`
- Expert(s) trained against: **high-noise** (`transformer`)
- Other components (frozen): `transformer_2`, VAE, T5 (loaded once for neg-prompt encoding, then dropped)

### ControlNet
- Architecture: `WanControlnet` (unchanged); single CN trained.
- Initialization: **Cold start.** Architecture config from `TheDenk/wan2.2-t2v-a14b-controlnet-hed-v1`; fresh `from_config` weights with `zero_module()`-zeroed output projections. **No `--num_cn_layers` override** (uses HED config default), stride=3.
- Input channels: 3 (Canny edges, RGB-replicated)
- Trainable parameter count: `352593072`
- Gradient checkpointing: ON (and on the active transformer)

### Data
- Source faces: `data/wan-beta/input_faces/` (100 PNGs, 528×528 → 512×512)
- Source targets: `data/wan-beta/targets/` (10000 PTDiffusion JPGs at 512×512)
- Prompt dictionary: `training/input_prompts.py` (`PROMPTS_BATCH_1 | PROMPTS_BATCH_2`, 100 distinct slugs)
- Total manifest records: 10000 = 100 faces × 100 prompts, every (face_idx, slug) pair appears exactly once.
- **Train / eval / periodic-eval split — face × prompt stratified (Latin pairings):**
  - **Eval set (100 records, held out):** identity matching — `sorted_faces[i]` paired with `sorted_slugs[i]` for `i = 0..99`. Every face appears once; every slug appears once.
  - **Periodic-eval set (10 records, subset of TRAIN):** shift-by-50 matching on the first 10 sorted faces — `sorted_faces[i]` paired with `sorted_slugs[(i+50) % 100]` for `i = 0..9`. 10 distinct faces × 10 distinct slugs, disjoint from the eval set. Used to watch whether the CN overfits on samples it trains on.
  - **Train set (9900 records):** every record whose (face_idx, slug) is not in the eval set. The 10 periodic-eval pairs *are* in the train set on purpose.
  - Constructed by `_build_eval_periodic_splits()` in `train_beta7.py`. Deterministic, no overlap, asserted at start.
- Canny preprocessing: `cv2.Canny(gray, 100, 200)`, stacked to 3 channels
- Resolution: 512×512
- Frame count `T`: 9 (replicated still → static video)
- Pair count after validation (train): `9900` (expected 9900)
- Eval count: `100` (expected 100)
- Periodic-eval count: `10` (expected 10)
- Cache directory: `$HOME/cache/wan-beta` (reused; no re-precompute — the manifest is identical, only the in-script split logic changes)

### Optimization
- Loss: flow-matching MSE (`v_target = noise - z_real`) + self-distillation consistency term
- Optimizer: `bitsandbytes.optim.AdamW8bit`
- LR: **5e-5**; weight decay 1e-4
- LR schedule: constant (no warmup; single-pass run)
- **Effective batch:** `32` = micro_batch=`1` × gradient_accumulation_steps=`32`. Per-micro-step we sample 1 σ in the high-noise band; over the accumulation, 32 independent σ are averaged into each Adam step.
- **Gradient accumulation backend (manual-gate pattern):** `accelerate.Accelerator(gradient_accumulation_steps=32)` provides the boundary signal via `accelerator.sync_gradients`; `accelerator.backward(loss)` does the 1/N loss scaling. `accelerator.prepare(...)` is NOT called — `optimizer.step()`, `zero_grad()`, clip, EMA, logging, eval, checkpoint are all gated by `if accelerator.sync_gradients:` outside the `accumulate` context.
- **Total optimizer steps:** `309` (1 epoch over 9900 train records ÷ 32)
- Grad clipping: 1.0 (only applied on `accelerator.sync_gradients`)
- Mixed precision: bf16 (fp32 kept for `_keep_in_fp32_modules`)
- **EMA on controlnet weights:** `ema_pytorch.EMA(decay=0.99, update_after_step=10, update_every=1)` (L6-style).
- **Self-distillation:** ON. `lambda_consistency=0.5`. Adds `0.5 · MSE(v_pred_live, v_pred_ema)` to the FM loss.
- Timestep sampling: high-noise regime only — `sigma >= boundary_ratio`, uniform within (rule used: `model_index.json.boundary_ratio`; boundary sigma: `0.875`)

### Eval protocol (in-training + post-training)
- **Periodic eval:** every **10 effective optimizer steps**, run inference on the **fixed 10 training-set samples** (overfit-check; same 10 every round so per-sample MSE/SSIM trajectories are directly comparable). Uses the **LIVE controlnet** (not EMA).
- **Final eval:** after training, swap the EMA checkpoint into the pipeline and run inference on all **100 held-out samples** (face/prompt-stratified).
- **Inference settings:** 50 denoising steps, `guidance_scale=5.0`, `controlnet_weight=1.0`, `controlnet_stride=3`, `controlnet_guidance_start=0.0`, `controlnet_guidance_end = <dynamically computed from σ trajectory>`, `seed=42`, fps=8.
- **CN-end fraction (dynamic):** `_compute_cn_end_high_noise()` runs `scheduler.set_timesteps(50)` and finds the smallest step index `i*` where `sigmas[i*] < boundary_ratio`. The fraction `i*/50` is passed as `controlnet_guidance_end`. Resulting `cn_end_fraction` recorded as `0.14`. Effect: the pipeline's step-fraction gate exactly mirrors the σ-based expert switch — CN computed only while `transformer` is the active expert; `transformer_2` never sees a residual.
- **Metrics per sample:**
  - **Pixel MSE:** `mean((frames - target_jpg)^2)` over `T × H × W × 3`, both in `[0, 1]`. Target JPG resized to 512×512, broadcast to `T=9`.
  - **SSIM:** `pytorch_msssim.ssim(frames, target_broadcast, data_range=1.0, size_average=True)`, computed on GPU. Single number per sample in roughly [-1, 1] (1 = identical, 0 = unrelated, < 0 = anti-correlated).
- **Storage:** wandb scalars (`eval/mse_avg`, `eval/ssim_avg`, `eval/mse_sample_{0..9}`, `eval/ssim_sample_{0..9}`; `eval_final/mse_avg`, `eval_final/ssim_avg` + a `wandb.Table` with both at end). Sibling JSON `training_cards/beta007/beta-007_eval.json` written incrementally so wall-time crash leaves a usable artifact. Each per-sample dict now carries both `mse` and `ssim` fields.
- **Inference videos saved:** all 30 × 10 = 300 periodic + 100 final → `outputs/wan-beta/beta-007/{periodic,final}/...`

### Hardware
- GPU: 1× **NVIDIA RTX PRO 6000 Blackwell Server Edition (96 GB, compute_cap 12.0)** on `hpc-node31`
- Attention backend: **NATIVE SDPA** forced via both `DIFFUSERS_ATTN_BACKEND=native` and in-script `set_attention_backend("native")`. Reason: flash-attn imports cleanly on `wan22-bw` but `flash_attn_func` is None at runtime on SM 12.0.
- Memory tripwire: 90 GB. Eval pipeline keeps `transformer + transformer_2 + VAE + controlnet` resident (~70 GB) — fits comfortably on Blackwell.
- Conda env: `wan22-bw` (no flash-attn)
- Estimated wall-time: ~7h (training ~5.5h + periodic eval ~1.5h + final eval ~30min — periodic-eval has 30 rounds of 10 samples now; the original 31×10 was 31 rounds because step 0 was included, here we start at step 10).
- Actual wall-time: `06:48:49`
- Cluster / partition: `dmb`

### Logging & checkpointing
- wandb project: `wan-controlnet-beta`, run name: `beta-007`
- wandb URL: `https://wandb.ai/robert-grgac2-university-of-twente/wan-controlnet-beta/runs/hj7of2b6`
- Per-effective-step metrics: `loss`, `loss_fm`, `loss_consistency`, `loss_ema` (window=20), `grad_norm`, `lr`, `controlnet_residual_norm`, `timestep`, `sigma`, `sigma_std`, `gpu_mem_gb`, `ema_decay_current`, `samples_seen`
- Periodic eval metrics (every 10 effective steps): `eval/mse_avg`, `eval/ssim_avg`, `eval/mse_sample_{00..09}`, `eval/ssim_sample_{00..09}`, `eval/wall_s`
- Final eval metrics: `eval_final/mse_avg`, `eval_final/ssim_avg`, `eval_final/wall_s`, `eval_final/per_sample_table` (columns: eval_idx, face_idx, slug, mse, ssim)
- Periodic checkpoints (EMA): `$HOME/checkpoints/wan-beta/beta-007_step{50,100,150,200,250,300}.safetensors`
- Final checkpoints:
  - `$HOME/checkpoints/wan-beta/beta-007_final.safetensors` — **EMA**, canonical for inference
  - `$HOME/checkpoints/wan-beta/beta-007_final_raw.safetensors` — raw weights, debug
- Eval log JSON: `training_cards/beta007/beta-007_eval.json` (per-sample MSEs+SSIMs, mp4 paths, per-checkpoint avg)

### Run metadata (auto)
- Status: `completed`
- Started: `2026-05-11T01:00:00+00:00`
- Finished: `2026-05-11T08:56:39+00:00`
- Git SHA: `8f4a4a1`

## 4. Results
*(machine fields auto-filled; qualitative fields human-filled after looking at wandb + inference videos)*
- Final loss (last effective step, raw): `0.172002`
- Final loss EMA (window 20): `0.179199`
- High-phase avg loss (over all eff steps): `0.198662`
- Final-eval mean MSE (100 samples, EMA controlnet): `0.107077`
- Final-eval mean SSIM (100 samples, EMA controlnet): `0.102811`
- Final-eval wall time: `3437.2`s
- GPU peak memory: `70.57` GB (90 GB tripwire)
- Effective batch size: 32 (= micro_batch × accum)
- Init mode: `cold`
- GPU: `NVIDIA RTX PRO 6000 Blackwell Server Edition` (compute_cap `12.0`)
- cn_end_fraction (dynamic): `0.14`
- Loss curve descended? (Y/N, `loss_ema` shape — monotonic / sawtooth / plateau / still-descending-at-end):
- Periodic-eval MSE trajectory shape (descending / U-shaped / flat / rising):
- Periodic-eval SSIM trajectory shape (rising / flat / falling):
- Periodic MSE at step 10 vs step 300 (overfit signal — should *descend* this time):
- Periodic vs final-eval MSE gap (overfit-on-train signal: periodic ≪ final → CN is fitting training samples but not generalizing):
- Comparison vs original beta-007 (videos noisy / silhouette visible / scene visible):
- Verdict (CN fix worked / partial / no change):
- Next action:
- **Pre-flagged risks:**
  1. EMA(0.99, update_after=10) + self-distillation: the consistency loss can collapse to ~0 by step ~30 once EMA ≈ live. If `loss_consistency` is flat at ~0 from step 30 onward and `loss_fm` looks similar to a no-self-distill run, self-distill is a 30%-wall-clock no-op — flag for removal next iteration.
  2. 309 effective steps is squarely in CLAUDE.md's "cold-start dead zone" (1k–3k steps for residuals to wake up). If periodic-eval MSE is flat after 309 steps even on trained samples, the natural extension is `max_steps=600` or warm-start from a prior beta-NNN.
  3. Dynamic `cn_end_fraction` depends on the FlowMatch Euler shift; for `boundary_ratio=0.875` and 50 inference steps it should land somewhere around `i*/50 ≈ 0.06–0.14`. If the computed value is implausibly small (< 0.02) or unity, something is wrong with the scheduler config; the run log prints `[cn-end] high-noise-only: ...` at startup — verify before training the same recipe in higher-cost variants.
  4. Pixel MSE remains a poorly-calibrated metric (noise vs natural image ≈ 0.10–0.15); the new SSIM track is the better signal. If MSE descends but SSIM stays near 0.0, the model is fitting luminance/mean-colour without structure — i.e., still noise just with the right palette.

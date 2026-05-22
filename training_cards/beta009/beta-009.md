---
run_id: beta-009
---

# Training Card — beta-009

Paired ablation documented as a single card. Two runs share one training
script and differ only in `--face_target_subdir`:

- **beta-009_raw_face** — `--face_target_subdir raw_face`
- **beta-009_silhouette** — `--face_target_subdir silhouette`

## 1. Goal
Test the additive face-target loss formulation: replace beta-008's spatially
weighted `(1 + α · mask) · MSE` with a clean

    loss = MSE(v_pred, v_target)                                  ← full-frame FM, unchanged
         + λ_face · face_region_MSE(v_pred, noise − z_face)        ← extra pull toward a face latent
         + λ_consistency · MSE(v_pred, v_pred_ema)                 ← self-distillation, on

and compare two choices for the additive target latent `z_face` — the
VAE-encoded **raw face image** (identity + interior features) vs the
VAE-encoded **silhouette drawing** (shape only, no identity).

The hypothesis is that the missing signal in beta-007/008 was *target shape*,
not target *region weight*. Pulling the face region in v-space toward an
explicit face-shaped clean latent (instead of just re-weighting the same
hidden/blended target) should produce more face-like structure in the
output. Comparing raw vs silhouette isolates whether the model needs the
full face image or just its outline.

## 2. Hypothesis & success criteria
- **Confirms (either run):**
  - ≥20/100 final-eval samples show clear face structure (vs ~2/100 in beta-008).
  - `eval_final/ssim_avg` strictly exceeds beta-008's value by ≥ 0.02.
  - Per-sample qualitative review: the face region in inference videos is
    recognizably face-shaped, not just textured noise.
- **Differentiates the two runs:**
  - If **raw ≫ silhouette** (≥ 0.03 SSIM gap, ≥ 10 more "face-visible" samples)
    → the CN genuinely needs pixel-level face content as the target; shape
    alone is insufficient.
  - If **raw ≈ silhouette** (within Δ0.02 SSIM) → shape supervision is enough
    and the CN can fill in identity from the prompt. Cheaper target wins for
    subsequent ablations.
  - If **silhouette > raw** → unlikely, would imply the raw face target
    introduces interference (e.g., model trying to fit unrealistic detail at
    high noise levels where structure dominates).
- **Rejects both:**
  - Both runs produce results indistinguishable from beta-008 (face structure
    still ~2–5/100 samples). Region-weighted loss isn't the bottleneck;
    architectural capacity or training budget is.
  - `loss_face` doesn't descend meaningfully across the run. The face term
    isn't actually fittable at λ_face=5 — try λ_face=15 next.

## 3. Setup

### Base model
- Repo: `Wan-AI/Wan2.2-T2V-A14B-Diffusers`
- Expert trained against: **high-noise** (`transformer`)
- Other components (frozen): `transformer_2`, VAE, T5

### ControlNet
- Architecture: `WanControlnet` (unchanged); single CN trained.
- Initialization: **Cold start.** Architecture config from
  `TheDenk/wan2.2-t2v-a14b-controlnet-hed-v1`; fresh `from_config` weights,
  output projections zeroed by `zero_module`. No `--num_cn_layers` override.
- Input modality (`--control_subdir`): **silhouette** (matches beta-007_silhouette / beta-008).
- Controlnet stride: 3.
- Trainable parameter count: `<AUTO:trainable_params>`
- Gradient checkpointing: ON.

### Data
- Source faces / targets / prompts / split: identical to beta-008. Manifest
  is the same; only the in-script split logic (face/prompt-stratified Latin
  pairing via `_build_eval_periodic_splits()`) is used.
- Train / eval / periodic counts: `9900 / 100 / 10` (asserted).
- Resolution: 512×512. Frames: 9 (replicated still).

### Loss (the only thing that changes from beta-008)
- **Full-frame FM loss:**
  `loss_fm = mean( (v_pred − (noise − z_real))² )`
  Unweighted. Recovers the beta-007 quantity — directly comparable across betas.
- **Additive face-region term:**
  `loss_face = sum( mask · (v_pred − (noise − z_face))² ) / (mask.sum() · C · T_lat)`
  - `mask`: per-face binary {0,1} silhouette mask (from `--face_mask_subdir`,
    default `silhouette`), thresholded then avg-pooled to latent
    resolution `(h_lat, w_lat) = (64, 64)`. Same mask source as beta-008's
    D1, *minus the +1 baseline* — the "+1 for background" role is now
    played by `loss_fm` itself.
  - `z_face`: VAE-encoded face image, loaded from `--face_target_subdir`
    (`raw_face` or `silhouette`). Encoded on-the-fly at training startup
    using the already-loaded eval VAE; ~30s for all 100 faces. Stored as
    bf16 on GPU keyed by `face_idx`. Normalized identically to z_real
    (`(z − latents_mean) / latents_std`).
  - Normalization by `mask.sum() · C · T_lat`: makes the term a true
    *face-region mean* MSE, independent of how much of the frame the face
    covers. λ_face is then interpretable as "extra MSEs of face-region
    accuracy added on top of the unit-weight full-frame loss".
- **Self-distillation consistency term (ON):**
  `λ_consistency · MSE(v_pred_live, v_pred_ema)`. Unchanged from beta-007/008.
  Enabled via `--use_self_distillation` in both sbatch files.
- **Total:**
  `loss = loss_fm + λ_face · loss_face + λ_consistency · loss_consistency`
- **λ_face = 5.0, λ_consistency = 0.5** (set explicitly in sbatch).

### Optimization (unchanged from beta-008)
- Optimizer: `bitsandbytes.optim.AdamW8bit`, LR 5e-5, weight decay 1e-4.
- Effective batch 32 = micro=1 × accum=32 via Accelerate manual-gate accumulate.
- Total optimizer steps: 309 (1 epoch × 9900 train / 32).
- Grad clipping 1.0; bf16 (fp32 kept for `_keep_in_fp32_modules`).
- EMA: `ema_pytorch.EMA(decay=0.99, update_after_step=10, update_every=1)`.
- Timestep sampling: high-noise regime (`sigma ≥ boundary_ratio`, uniform).

### Eval protocol (unchanged from beta-008)
- Periodic: every 10 effective steps on the fixed 10 train-set samples
  (overfit check), LIVE controlnet, dynamically-computed `cn_end_fraction`.
- Final: 100 held-out samples, EMA controlnet, same `cn_end_fraction`.
- Metrics: pixel MSE + SSIM per sample; wandb scalars + Table; sibling JSON
  `training_cards/beta009/{run_name}_eval.json`.
- Inference settings: 50 denoising steps, guidance_scale 5.0,
  controlnet_weight 1.0, controlnet_stride 3, controlnet_guidance_start 0.0,
  controlnet_guidance_end dynamic, seed 42, fps 8.

### Hardware
- GPU: 1× NVIDIA RTX PRO 6000 Blackwell Server (96 GB) on `hpc-node31`.
- Attention backend: native SDPA (forced).
- Conda env: `wan22-bw`.
- Estimated wall-time: ~7h per run (same as beta-007/008 — z_face encoding
  adds ~30s at startup, negligible).

### Logging & checkpointing
- wandb project: `wan-controlnet-beta`.
- Per-effective-step metrics: `loss`, `loss_fm`, `loss_face`,
  `loss_consistency`, `loss_ema`, `grad_norm`, `lr`,
  `controlnet_residual_norm`, `timestep`, `sigma`, `sigma_std`,
  `gpu_mem_gb`, `ema_decay_current`, `samples_seen`.
- Periodic eval metrics: `eval/mse_avg`, `eval/ssim_avg`,
  `eval/{mse,ssim}_sample_{00..09}`, `eval/wall_s`.
- Final eval metrics: `eval_final/mse_avg`, `eval_final/ssim_avg`,
  `eval_final/wall_s`, `eval_final/per_sample_table`.
- Final checkpoints:
  - `$HOME/checkpoints/wan-beta/{run_name}_final.safetensors` (EMA).
  - `$HOME/checkpoints/wan-beta/{run_name}_final_raw.safetensors` (raw).

### Run metadata (auto, per run)

#### beta-009_raw_face
- wandb URL: `<AUTO:wandb_url>`
- Status: `<AUTO:status>`
- Started: `<AUTO:date_started>`
- Finished: `<AUTO:date_finished>`
- Git SHA: `<AUTO:git_sha>`

#### beta-009_silhouette
- Filled by hand after the second run (autofill keys the same on both
  result JSONs; see `training_cards/beta009/beta-009_silhouette_results.json`).

## 4. Results

### beta-009_raw_face
*(auto-filled from `beta-009_raw_face_results.json`; qualitative fields human-filled after looking at wandb + inference videos)*
- Final loss: `<AUTO:final_loss>`
- Final loss EMA: `<AUTO:loss_ema_final>`
- High-phase avg loss: `<AUTO:high_phase_avg_loss>`
- Final-eval mean MSE (100 samples, EMA controlnet): `<AUTO:final_eval_mse_avg>`
- Final-eval mean SSIM (100 samples, EMA controlnet): `<AUTO:final_eval_ssim_avg>`
- Final-eval wall time: `<AUTO:final_eval_wall_s>`s
- GPU peak memory: `<AUTO:gpu_peak_mem_gb>` GB
- Wall time: `<AUTO:wall_time>`
- cn_end_fraction (dynamic): `<AUTO:cn_end_fraction>`
- z_face shape: `<AUTO:z_face_shape>` (encoded `<AUTO:z_face_count>` latents in `<AUTO:z_face_encode_seconds>`s)
- loss_face curve descended? (Y/N, shape):
- Comparison vs beta-008 (face structure visible / partial / no change):
- Number of final-eval samples with clearly face-shaped output (out of 100):
- Verdict (additive formulation works / no improvement / regression):

### beta-009_silhouette
*(fill from `beta-009_silhouette_results.json` after the second run)*
- Final loss:
- Final loss EMA:
- Final-eval mean MSE:
- Final-eval mean SSIM:
- z_face shape (should match raw_face — same VAE, same input resolution):
- loss_face curve descended? (Y/N):
- Number of final-eval samples with face-shaped output:
- Comparison vs beta-009_raw_face (raw better / silhouette better / tie):
- Verdict on shape-vs-identity question (raw needed / silhouette sufficient):

### Cross-run conclusion
- Does the additive formulation produce clearer face structure than beta-008's
  weighted formulation? (Y/N):
- Which target wins (raw / silhouette / tie)? Implication for follow-up runs:
- Next action (sweep λ_face on the winner / try a third target / move on):

### Pre-flagged risks
1. **`z_face` shape mismatch.** If the precomputed face/silhouette images
   were saved at a resolution that doesn't match `--height/--width`, the
   smoke's first batch will fail the `z_face.shape == z_real.shape` assert.
   `precompute_raw_face.py` and `precompute_silhouette.py` default to 512×512;
   the train script also enforces (3, H, W) shape on load. Run smoke first.
2. **High λ_face overrides scene blending.** At λ_face=5, face-region pull
   is comparable to FM gradient strength on a per-element basis. If the
   resulting videos show the raw face *uncamouflaged* instead of hidden in
   the scene, λ_face is too high — drop to 2 or 1 for a follow-up.
3. **Silhouette as both mask AND target.** When `--face_target_subdir=silhouette`
   and `--face_mask_subdir=silhouette`, the loss is "match the silhouette
   drawing in the silhouette region" — circular but well-defined. The
   silhouette image extends slightly beyond the {0,1}-thresholded mask
   (gray fill + brighter contour lines), so the additive term is not a
   trivial no-op even with these settings.
4. **Self-distillation collapse.** As with beta-007/008, `loss_consistency`
   may decay to ~0 once EMA ≈ live (around step 30). If true, the term is
   a wall-clock cost without benefit; flag for removal next iteration.

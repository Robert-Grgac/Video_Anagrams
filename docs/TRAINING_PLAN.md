# Training Plan: Hidden-Face Video Generation via ControlNet (v2 — Warped Pseudo-Targets)

## 1. Project Overview

The goal is to generate videos that, when viewed normally, depict a text-prompt scene (e.g. "forest leaves blowing in the wind"), but when viewed from afar (or low-pass filtered) reveal a moving human face matching a reference head-movement video. This is the video extension of the still-image technique demonstrated in PTDiffusion.

### Why a trained ControlNet rather than inference-time guidance

PTDiffusion-style guidance applied per-frame at inference time was tried first and produced unstable, non-converging generations with severe artifacts. Amortizing the blending behavior into a trained ControlNet is more stable: gradient descent learns the mapping once, the network reproduces it deterministically at inference, and Wan's frozen prior provides temporal coherence through its 3D attention.

### Why warped pseudo-targets rather than a two-loss equilibrium

An earlier version of this plan used a combined flow-matching + low-pass face loss to specify the blend as the equilibrium of two competing objectives. That plan was discarded in favor of the simpler approach below: PTDiffusion already produces high-quality blended stills, and head motion can be transferred onto those stills via optical flow warping. Treating the warped result as a direct supervision target reduces the training problem to standard ControlNet flow-matching — no per-frame perceptual loss, no VAE decode in the gradient graph, no two-loss balancing.

## 2. Conceptual Foundation

The decomposition:

1. **PTDiffusion solves the blending problem for stills**. For a given (face, prompt) pair it produces an image that looks like the prompt at high frequency and the face at low frequency.
2. **Optical flow captures head motion**. For a face video, dense optical flow describes per-pixel motion between frames.
3. **Warping a still by that flow synthesizes a pseudo-video** in which the blended content moves the way a head would. Texture moves rigidly with head motion, which is the *desired* low-frequency behavior for the illusion. High-frequency texture realism is provided by Wan's prior during diffusion training.
4. **Standard ControlNet training on these pseudo-targets** is then sufficient: the network learns to produce videos matching the pseudo-targets given (prompt, landmark heatmaps).

This trades a difficult joint optimization for a sequence of well-understood operations.

## 3. Training Data Structure

Each training sample is a tuple:

- **`face_video`**: head-movement clip. Used to derive landmarks and optical flow. Not used as a pixel target.
- **`landmark_heatmap_video`**: 6-channel anatomical-region heatmaps from `face_video` via MediaPipe Face Mesh. The ControlNet input. Shape `(6, T, H, W)`.
- **`prompt`**: text describing the surface content.
- **`pseudo_target_video`**: the L_FM target. Constructed offline by warping a single PTDiffusion still using optical flow extracted from `face_video`. Shape `(3, T, H, W)`.

`face_video`, `landmark_heatmap_video`, and `pseudo_target_video` are all time-aligned by construction.

### Pseudo-target construction (offline)

For each (face_video, prompt) pair:

1. **Pick a canonical frame** from `face_video`. Frame 0 by default; can be hand-picked if a more frontal pose exists elsewhere in the clip.
2. **Render the PTDiffusion still**: using the canonical frame's face as PTDiffusion's reference, generate a blended still for the prompt. This is a one-time per-pair cost.
3. **Extract dense optical flow** from the canonical frame to every other frame in `face_video`. Use RAFT or GMFlow (modern learned methods); classical methods are insufficient.
4. **Warp the PTDiffusion still** to each frame's pose using the composed flow. Always warp from the canonical frame, not chained frame-to-frame, to bound accumulated error.
5. **Handle disocclusions and out-of-frame regions** via padding strategy (see below).
6. **Save the resulting `pseudo_target_video`** to disk.

This is a one-time data preparation step. Training reads pseudo-targets from disk; no warping happens during training.

### Disocclusion / out-of-frame handling

When head motion reveals regions that were never in the canonical frame, the warped image has no source content there. Three options, in order of complexity:

1. **Reflection padding** (default): pad the source PTDiffusion still by ~20% on each side using mirror reflection before warping; the warped result's edges are then synthesized from valid mirrored content. Simplest, no extra dependencies, acceptable for moderate motion.
2. **Wider PTDiffusion render**: render PTDiffusion at a wider FOV than the training resolution, warp, then center-crop to the training resolution. Higher quality but requires re-rendering with awareness of crop margin.
3. **Per-frame inpainting**: use any inpainting model to fill disocclusion regions. Highest quality, most pipeline complexity. Defer until needed.

Start with option 1.

### Face-video / PTDiffusion-still alignment

The PTDiffusion still must be rendered using a reference face that **matches the canonical frame's face** in identity and rough pose. Options:

- **Re-render PTDiffusion per face video** (recommended): for each face video, extract the canonical frame, run PTDiffusion using that face as reference, save the still. Most flexible. Requires PTDiffusion compute budget proportional to the number of face videos.
- **Use existing PTDiffusion stills + matched face videos**: if you already have a set of (reference_face_image, prompt, blended_still) tuples from earlier work, source face videos containing those specific identities and use the matching frame as canonical. Avoids re-running PTDiffusion. Requires the more difficult sourcing.

### Optical flow notes

- Use RAFT or GMFlow at training resolution. Do not compute at higher resolution and downsample — flow fields do not resize cleanly.
- Compute flow from canonical frame to every other frame directly when possible (most modern flow methods support arbitrary frame pairs); otherwise compose pairwise flows with a forward-warping accumulator.
- Validate flow quality on a few samples before bulk processing: visualize warped pseudo-targets and inspect for severe distortion, especially at large head rotations.

### Recommended dataset size

- **Prompts**: 10–20 distinct prompts (textures with motion that could plausibly hide head motion: leaves, water, clouds, smoke, fire embers, grass, ripples, foliage).
- **Face videos**: 100–300 head-movement clips with variety in motion type, identity, lighting, and pose range.
- **PTDiffusion stills**: one per (face_video, prompt) pair — total ~1,000–6,000.
- **Pseudo-target videos**: one per still, same total.

Effective unique training samples ≈ number of pseudo-target videos. Diversity matters more than count; aim for variety in head motion (yaw, pitch, nod, shake combinations) since this defines the motion vocabulary the trained ControlNet inherits.

### Storage

Pseudo-target videos at 256×256 × 16 frames × 3 channels in fp16 ≈ ~6 MB per clip; 5,000 clips ≈ 30 GB. Comfortable on a single SSD.

## 4. Loss Function

```
L = L_FM
```

Standard flow-matching loss:

1. Encode `pseudo_target_video` to latent `z_real` via the frozen VAE encoder.
2. Sample timestep `t` and Gaussian noise; build noisy latent `z_t`.
3. ControlNet receives `z_t`, timestep, prompt embedding, and `landmark_heatmap_video`; emits per-block residuals.
4. Main transformer consumes residuals and predicts velocity `v_pred`.
5. `L_FM = MSE(v_pred, v_target)` where `v_target` is defined by the FlowMatchEulerDiscreteScheduler's flow-matching parameterization.

No per-frame loss. No VAE decode in the gradient graph. No second loss term.

### Why this is sufficient

The pseudo-target video already encodes both objectives:

- It has prompt-aligned high-frequency texture (from the PTDiffusion still).
- It has face-shaped low-frequency content following the head motion (from the optical flow warp).

Training the ControlNet to produce videos matching these targets, given the landmark heatmaps as input, transfers the entire blending behavior into the network in one shot.

## 5. Training Architecture

### Frozen

- Text encoder (UMT5).
- VAE encoder (used to encode `pseudo_target_video` per step) and decoder (used only at eval/inference).
- Main Wan transformer (and `transformer_2` if present).
- Pipeline scheduler.

### Trainable

- Only `WanControlnet` parameters.

### Per training step

1. Sample tuple `(landmark_heatmap_video, prompt, pseudo_target_video)`.
2. Encode prompt text → `prompt_embeds` (no_grad, cached if possible).
3. Encode `pseudo_target_video` → `z_real` via VAE encoder (no_grad).
4. Sample timestep `t` and Gaussian noise; build `z_t`.
5. Forward: ControlNet residuals → main transformer → `v_pred`.
6. Compute `L_FM = MSE(v_pred, v_target)`.
7. Backward; clip grads; optimizer step (only ControlNet parameters update).

### Implementation notes

- Gradient checkpointing on the ControlNet (`gradient_checkpointing = True`) to fit larger batches.
- Mixed precision (bf16) recommended.
- TeaCache off during training (inference-only feature).
- Classifier-free guidance off during training (train conditional path only).
- VAE encoding can be precomputed and cached per pseudo-target if disk space permits — this saves the encode pass per training step at the cost of storing latent tensors. Recommended for large training runs.

## 6. Hyperparameters

### Architectural / fixed

| Parameter | Value | Notes |
|---|---|---|
| `in_channels` | 6 | Six anatomical heatmap channels. |
| `controlnet_stride` | 3 | Pipeline default; must match at inference. |
| `controlnet_weight` | 1.0 | Not learnable; runtime-only inference dial. |
| `controlnet_guidance_start` | 0.0 | Train across full schedule. |
| `controlnet_guidance_end` | 1.0 | Train across full schedule. |
| Number of ControlNet blocks | 20 | Default. |

### Optimization

| Parameter | Value |
|---|---|
| Optimizer | AdamW |
| Peak learning rate | 5e-5 |
| LR warmup | 0 to peak over first 500–1,000 steps |
| LR schedule after warmup | Constant (or cosine to 1e-5 over total training) |
| Weight decay | 1e-2 |
| Gradient clipping | 1.0 (norm) |
| Batch size (per device) | 1–2 video clips |
| Gradient accumulation | 2–4 (effective batch 4–8) |
| Precision | bf16 mixed |

### Loss weights

Single loss, no lambdas to schedule. Just `L_FM`.

### Timestep sampling bias (optional but recommended)

Flow-matching defaults to logit-normal timestep sampling. Consider biasing slightly toward higher noise (mean = 0.0 → mean = 0.3 in logit-space) so the model emphasizes coarse-structure learning over fine-detail reproduction. This reduces the risk of overfitting to optical-flow warping artifacts that live primarily in high frequencies.

## 7. Training Schedule

No lambda annealing — simpler than the previous plan. Just two stages:

### Stage 1: Sanity (1–2k steps, ~few hours)

- Standard L_FM training on a small subset (5 face videos, 5 prompts).
- Confirm: L_FM converges, ControlNet residuals are non-zero (norm rises from 0), generation visually tracks pseudo-targets.
- Acceptance: training is stable, residuals not degenerate, basic forward/backward wiring works.

### Stage 2: Main training (20k–60k steps, ~1–3 days)

- Full dataset, full batch size.
- Periodic eval generations and qualitative checks.
- Acceptance: blended-face illusion visible in at least 50% of eval samples; texture quality preserved.

### Optional Stage 3: Resolution / length fine-tune

If bulk training was at 256×256 × 16 frames, optional final 5k–10k steps at 480×832 × 24 frames (or whatever inference resolution targets). Most learnings transfer; this polishes for the deployment configuration.

### Resolution and clip length during training

- Bulk training: 256×256, 16 frames.
- Final fine-tune (optional): 480×832, 24 frames.

Resolution scales memory hard; keep bulk training at low resolution for iteration speed.

Don't think in epochs — think in steps. Cross-product of face videos × prompts gives effectively unlimited training pairs.

## 8. Evaluation Strategy

### Fixed eval set

Construct ~10 (face_video, prompt) pairs that are held out from training. Pre-compute landmark heatmaps for them. **No pseudo-targets needed for eval** — eval is open-ended generation, not target-matching.

### Eval cadence

- Every 1,000 steps during Stage 1.
- Every 2,000–5,000 steps during Stage 2.

### Per-eval outputs

For each eval pair, run full inference with the current ControlNet checkpoint and save:

- The raw generated video (for qualitative inspection).
- A side-by-side composite: generated video vs. blurred-generated vs. blurred-face-video. This is the most informative eval visualization.

Log to W&B / TensorBoard as video artifacts.

### Quality checks (manual, per eval)

1. **Texture quality**: does it still look like the prompt content?
2. **Hidden-face visibility**: blur the video; does the face emerge with correct pose tracking?
3. **Temporal coherence**: any flicker, stutter, or motion artifacts?
4. **Warp-artifact reproduction**: does the output look "warped from a still" rather than natively generated? This is the most likely failure mode.

### Quantitative metric: blurred-face cosine similarity

Automate "from afar" evaluation:

1. Generate the eval video with the current ControlNet.
2. Apply Gaussian blur (sigma ≈ 16 at output resolution) to each frame.
3. Run a face detector / ArcFace embedding on each blurred frame.
4. Compute cosine similarity to the corresponding `face_video` blurred-frame embedding.
5. Average over frames where both detections succeed.

Track over training; rising = illusion strengthening.

## 9. Metrics to Track

### Every step (or every 100 steps)

- `L_FM` (raw value).
- ControlNet parameter gradient norm.
- ControlNet residual magnitude (mean L2 norm of the residual tuple). Confirms the network has woken up from zero-init.
- Learning rate.

### Every 1k–5k steps

- Sample inference videos on the fixed eval set, logged as W&B videos / TB GIFs.
- Blurred-face cosine similarity (validation metric).

### What healthy training looks like

- L_FM: drops in Stage 1, continues slow decline through Stage 2, eventually plateaus.
- Gradient norm: stable in the 0.1–10 range. Spikes above 100 are warning signs.
- Residual magnitude: rises from 0 at the start, then stabilizes at a non-trivial value.

## 10. Failure Modes and Mitigation

| Symptom | Likely cause | Mitigation |
|---|---|---|
| Output has "warped from a still" look (smearing, fixed-pattern distortion) | Model overfit to warping artifacts in pseudo-targets | Improve flow quality (RAFT > classical); use reflection padding / wider source render to reduce edge artifacts; bias timestep sampling toward higher noise; reduce training length |
| Generated face shape doesn't track the input landmarks | ControlNet input not actually reaching the network, or landmarks too sparse | Verify landmark heatmaps non-zero per frame; increase Gaussian sigma to make heatmaps more diffuse; verify ControlNet first-conv weights are non-zero (not stuck) |
| Output looks like vanilla Wan, no face | ControlNet residuals near zero | Inspect residual magnitude metric; check learning rate not too low; verify gradients flow into ControlNet |
| Severe per-frame flicker | Pseudo-targets themselves flicker (poor flow quality) | Inspect raw pseudo-target videos; switch flow method; smooth flow temporally before warping |
| Pseudo-target video has visible warp artifacts at edges | Disocclusion handling insufficient | Switch from reflection padding to wider PTDiffusion render with center-crop |
| Model collapses to generating literal blurred faces | Pseudo-targets dominated by face content over texture | Inspect pseudo-targets; ensure PTDiffusion stills have strong prompt content, not face-dominated; potentially re-run PTDiffusion with stronger prompt weighting |
| Gradient norm spikes / NaNs | LR too high, fp16 instability | Lower LR; switch to bf16; clip more aggressively |
| ControlNet residuals stay at zero | Wiring bug, dead path | Inspect tensor shapes; verify landmarks non-zero; check ControlNet is in train mode and gradients flow |

## 11. Inference at the End

After training, runtime knobs become useful:

- `controlnet_weight`: dial up/down the strength of the trained ControlNet (e.g., 0.7 for subtler illusions, 1.2 for stronger). Note: with this v2 plan trained on pseudo-targets only, dialing `controlnet_weight` to 0 does NOT recover vanilla-Wan output cleanly — the controlnet's contribution is "always on" because the model never saw vanilla-Wan targets. This is a known limitation of the v1 simplification (see section 12).
- `controlnet_guidance_start` / `end`: restrict ControlNet activity to a window of denoising steps if late-step ControlNet activity hurts texture quality. May help mitigate warp-artifact reproduction.
- TeaCache: re-enable for inference speedup.
- Standard CFG: re-enable.

## 12. Future Variants

Documented for reference; not in scope for the initial version.

### Mixing vanilla-Wan targets

The current plan uses pseudo-targets exclusively, simplifying training but losing the ability to dial the ControlNet down to vanilla-Wan output at inference. If this becomes a problem (e.g., the trained model produces face-like content even when undesired), the fix is to mix in vanilla-Wan generations as L_FM targets at some ratio (e.g., 50/50). The ControlNet then learns "act as a no-op when controlnet_weight is low" and "produce blends when controlnet_weight is high." Implementation cost: moderate (additional dataset path, no training-loop change).

### Adding a low-pass face loss back

If the warped-target approach plateaus at insufficient face visibility, consider re-introducing the low-pass L1 face loss as an *auxiliary* term with small weight (`lambda_face = 0.05`). This provides additional gradient signal toward the perceptual goal beyond what L_FM alone supervises. Implementation cost: significant (per-frame VAE decode in gradient graph).

### Per-pose PTDiffusion rendering with interpolation

Instead of warping a single still, render PTDiffusion at 3–5 key head poses and interpolate-and-warp between them. Reduces warp-artifact severity for large motions at the cost of more PTDiffusion compute.

### ArcFace-based eval / loss

Identity-aware face similarity (rather than pixel-level low-pass) for both eval and as an auxiliary training signal. Defer until basic version works.

## 13. Out of Scope (For Now)

- Real-video L_FM targets (sticking with PTDiffusion-warped pseudo-targets only).
- Full Wan 2.2 high-noise + low-noise stage support beyond what the existing pipeline provides.
- Multi-resolution training beyond a single bulk + optional final-stage resolution.
- Hyperparameter automation based on loss curves.
- Inpainting-based disocclusion handling.

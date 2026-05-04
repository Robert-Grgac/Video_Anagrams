# Implementation Plan: Hidden-Face ControlNet Training Pipeline (v2 — Warped Pseudo-Targets)

This plan converts the design in `TRAINING_PLAN.md` into a concrete sequence of code changes. It is structured for incremental implementation: each phase has a clear deliverable and validation step. Do not skip phases — each builds on the previous.

## 0. Repository Context (existing files)

- `wan_controlnet.py` — `WanControlnet` model. Already accepts `in_channels` as a config parameter (default 3); architecture downstream of the first Conv3d is invariant to channel count. Inherits from `PeftAdapterMixin` and `ModelMixin`. Supports gradient checkpointing.
- `wan_transformer.py` — `CustomWanTransformer3DModel`. Subclasses Wan's diffusers transformer to inject ControlNet residuals at every `controlnet_stride` blocks during the block loop. Also wires TeaCache (inference-only).
- `wan_t2v_controlnet_pipeline.py` — `WanTextToVideoControlnetPipeline`. Full inference pipeline: text encode, prepare latents, ControlNet residual extraction, denoising loop, VAE decode.
- `wan_teacache.py` — Inference-only caching. Off during training.
- `inference/cli_demo.py` — Existing CLI inference entry point. Reference for how the pipeline is loaded and invoked.

No training code exists yet. All new training code must be additive — do not break the existing inference path.

## 1. Goal

Build the pipeline that:

1. Extracts 6-channel anatomical-region landmark heatmaps from face videos.
2. Generates warped pseudo-target videos by applying optical flow (extracted from face videos) to PTDiffusion-rendered stills.
3. Trains a 6-input-channel `WanControlnet` against those pseudo-targets with standard flow-matching loss; Wan/VAE/text-encoder remain frozen.
4. Saves checkpoints and logs metrics for periodic evaluation.

Modify the inference pipeline minimally to consume 6-channel heatmap inputs at inference time.

## 2. Phased Implementation

### Phase 1: Landmark heatmap extraction

**Deliverable**: a module that converts a video file (or list of frames) into a `(6, T, H, W)` heatmap tensor using MediaPipe Face Mesh.

**New file**: `landmark_extraction.py`

**Specification**:

- `extract_landmarks_from_video(video_path: str | Path) -> np.ndarray`. Returns shape `(T, N_landmarks, 2)` of pixel-space landmark coordinates from MediaPipe Face Mesh.
- Constant `LANDMARK_GROUPS: dict[str, list[int]]` mapping each of the 6 anatomical regions to its MediaPipe Face Mesh landmark indices (face_outline, left_eye, right_eye, nose, mouth, jaw). Use canonical MediaPipe groupings.
- `render_heatmap(landmarks_2d: np.ndarray, group_indices: list[int], height: int, width: int, sigma: float) -> np.ndarray`. Returns `(H, W)`. Sums isotropic 2D Gaussians at each landmark in the group. Vectorized: `np.exp(-((xx - x0)**2 + (yy - y0)**2) / (2 * sigma**2))`, summed.
- `landmarks_to_heatmap_video(landmarks: np.ndarray, height: int, width: int, sigma: float) -> np.ndarray`. Returns `(T, 6, H, W)`.
- `prepare_landmark_heatmaps(face_video_path: str | Path, height: int, width: int, sigma: float, dtype: torch.dtype, device: torch.device) -> torch.Tensor`. Returns `(1, 6, T, H, W)` ready for the ControlNet. Mirrors `prepare_controlnet_frames` in the existing pipeline.

**Heatmap value range**: per-channel max-normalize so each heatmap's max equals 1.0 within a frame (clip if > 1.0 after summation). Document in docstring; must match between training and inference.

**Failure handling**: if MediaPipe fails to detect a face on a frame, emit zeros for that frame's heatmap. Log frame-level detection success rate. Reject training samples with detection rate below 90%.

**Validation step**:
- Write a `__main__` block that loads a test face video, runs full extraction, and saves the 6 heatmap channels as PNGs (one row per frame, six columns) for visual inspection.
- Verify Gaussians appear at correct anatomical locations and are temporally continuous.

### Phase 2: Pseudo-target generation pipeline

**Deliverable**: an offline data-prep pipeline that, given face videos and PTDiffusion stills, produces warped pseudo-target videos saved to disk.

**New file**: `pseudo_target_generation.py`

**Specification**:

#### 2.1 Optical flow extraction

- `compute_flow_canonical_to_all(face_video: np.ndarray, canonical_idx: int, model: str = "raft") -> np.ndarray`. Returns shape `(T, 2, H, W)` of dense flow from `face_video[canonical_idx]` to each frame `t`.
- Use **RAFT** via `torchvision.models.optical_flow.raft_large` (preferred — already in torchvision). GMFlow is an acceptable alternative if available.
- Compute pairwise flows (canonical → t) directly when the flow model supports arbitrary frame pairs. Otherwise compose pairwise consecutive-frame flows: `flow[0→t] = compose(flow[0→1], flow[1→2], ..., flow[t-1→t])` using forward-warping accumulation.
- Run at training resolution. Do not compute at higher resolution and resize the flow.

#### 2.2 Warping

- `warp_image(image: np.ndarray, flow: np.ndarray) -> np.ndarray`. Given source image `(3, H, W)` and flow `(2, H, W)`, return `(3, H, W)` warped image. Use `torch.nn.functional.grid_sample` with `mode="bilinear"`, `padding_mode="reflection"`, and `align_corners=False`.
- Build the sampling grid by adding the flow to the identity grid, then normalizing to `[-1, 1]`.

#### 2.3 Disocclusion handling (Phase 1 strategy: reflection padding)

- Pad the source PTDiffusion still by 20% on each side using `np.pad(..., mode="reflect")` before warping. The warp's `padding_mode="reflection"` handles any remaining out-of-bounds samples.
- Center-crop the warped result back to training resolution.
- Document this margin choice; if motion is too aggressive for 20% margin, increase or switch to wider PTDiffusion render strategy (option 2 in TRAINING_PLAN.md).

#### 2.4 Top-level orchestration

- `generate_pseudo_target_video(face_video_path: str | Path, ptdiffusion_still_path: str | Path, canonical_frame_idx: int, height: int, width: int, num_frames: int, output_path: str | Path) -> None`.
  1. Load face video; resize/crop to (height, width); take `num_frames` from a stable temporal offset.
  2. Load PTDiffusion still; resize to (height, width); reflection-pad.
  3. Compute flow from canonical frame to all other frames.
  4. Warp the padded still using each frame's flow.
  5. Center-crop each warped frame to (height, width).
  6. Save as `.mp4` (h264, lossless preset) or `.npy` / `.pt` for quality preservation. Recommended: save as `.pt` containing `(3, T, H, W)` fp16 tensor — direct training input, no encoding loss.

#### 2.5 Batch script

- `data_prep/generate_all_pseudo_targets.py`: takes a CSV or JSON manifest mapping face_video → ptdiffusion_still → prompt; iterates and calls `generate_pseudo_target_video` for each.
- Parallelize via `concurrent.futures` if multiple GPUs available (one flow extraction per GPU).

**Validation step**:
- Generate pseudo-targets for 5 (face_video, ptdiffusion_still, prompt) tuples.
- Save side-by-side videos showing: face_video | warped pseudo-target | blurred pseudo-target. Inspect visually.
- Confirm the blurred pseudo-target shows a moving face shape; confirm the unblurred one shows prompt-aligned texture. Reject any pseudo-targets with severe warp artifacts at the edges (means motion exceeds reflection-padding margin).

### Phase 3: ControlNet `in_channels=6` support

**Deliverable**: `WanControlnet` instantiable with `in_channels=6`, with a clean way to load a 3-channel pretrained checkpoint and reinitialize only the first conv.

**File to modify**: `wan_controlnet.py` — minimal change; the architecture already supports it.

**Add helper function**:

```python
def init_from_pretrained(controlnet: WanControlnet, pretrained_path: str | Path, new_in_channels: int) -> WanControlnet:
    """Load all weights from a 3-channel checkpoint, then reinitialize the first
    Conv3d of control_encoder[0] to accept new_in_channels inputs.
    """
```

If no pretrained weights are available, skip this and just instantiate fresh. Document both paths in the training script.

**Validation step**:
- Instantiate `WanControlnet(in_channels=6, ...)` and run a forward pass with `(1, 6, T, H, W)` input plus a dummy noisy latent. Confirm output is a tuple of 20 residuals with expected shapes.
- Confirm zero-init: residuals from a fresh model should be all zeros.

### Phase 4: Dataset and dataloader

**Deliverable**: a PyTorch `Dataset` yielding training tuples and a `DataLoader` producing batches.

**New file**: `train_dataset.py`

**Specification**:

- Class `HiddenFacePseudoTargetDataset(Dataset)`:
  - Constructor args: `manifest_path` (JSON or CSV listing tuples), `height`, `width`, `num_frames`, `landmark_sigma`, `cache_dir` (optional).
  - Manifest format: list of records, each `{face_video: str, pseudo_target: str, prompt: str, canonical_frame_idx: int}`.
  - `__init__`: load manifest; validate all referenced files exist; log counts.
  - `__len__`: `len(records)`.
  - `__getitem__(idx)`:
    1. Load `face_video` (decord or imageio); take `num_frames` from `canonical_frame_idx`; resize/center-crop to (height, width).
    2. Run `prepare_landmark_heatmaps` on the face frames → `(6, T, H, W)`.
    3. Load `pseudo_target` from `.pt` file → `(3, T, H, W)`. No re-encoding needed since pseudo-targets are pre-computed.
    4. Look up the prompt text.
    5. Return dict: `{"landmark_heatmaps": ..., "pseudo_target": ..., "prompt": str}`. Note: `face_video` itself is no longer returned — only the heatmaps derived from it are needed for training.
  - Caching: optionally pre-extract heatmaps to `.pt` files keyed by face video path; on subsequent epochs load from cache. Make it opt-in via `cache_dir` arg.

**Manifest example** (`manifest_train.json`):
```json
[
  {
    "face_video": "data/face_videos/0001.mp4",
    "pseudo_target": "data/pseudo_targets/0001_leaves.pt",
    "prompt": "forest leaves blowing in the wind",
    "canonical_frame_idx": 0
  },
  ...
]
```

**Validation step**:
- Instantiate dataset on a small subset, iterate 10 samples, confirm shapes and that heatmaps and pseudo-targets are time-aligned (save sample's heatmap channels and pseudo-target frames side-by-side as PNG).

### Phase 5: Training script

**Deliverable**: a runnable training script. Initial version performs one full forward + backward pass without errors; full version includes logging, checkpointing, and evaluation.

**New file**: `train_controlnet.py`

**Specification**:

```python
# Pseudocode-level structure:

def main(config):
    # 1. Init accelerator (accelerate library; bf16 mixed precision)
    # 2. Load tokenizer, text_encoder (frozen, eval mode, no_grad)
    # 3. Load vae (frozen, eval mode, no_grad — encoder is used per step)
    # 4. Load main transformer (frozen, eval mode); optionally transformer_2
    # 5. Build or load WanControlnet with in_channels=6 (trainable)
    # 6. Set requires_grad=False on everything except controlnet
    # 7. Enable gradient checkpointing on controlnet
    # 8. Build optimizer (AdamW) on controlnet.parameters() only
    # 9. Build LR scheduler (warmup 500-1000 steps then constant or cosine)
    # 10. Build dataset + dataloader
    # 11. Load FlowMatchEulerDiscreteScheduler (for noise scheduling)
    # 12. Init logging (wandb or tensorboard)
    
    for step, batch in enumerate(dataloader):
        # 1. Encode prompt text -> prompt_embeds (no_grad)
        # 2. Encode pseudo_target -> z_real via vae encoder (no_grad)
        #    Apply latent normalization: (z_real - mean) * std (matching pipeline)
        # 3. Sample timestep t (logit-normal default; consider bias toward higher noise)
        # 4. Sample noise; build z_t per FlowMatchEulerDiscreteScheduler
        # 5. Compute v_target per the scheduler's flow-matching parameterization
        # 6. Forward:
        #    - controlnet residuals from (z_t, t, prompt_embeds, landmark_heatmaps)
        #    - main transformer prediction v_pred (with controlnet_states=residuals,
        #      controlnet_weight=1.0, controlnet_stride=3)
        # 7. L_FM = MSE(v_pred, v_target)
        # 8. accelerator.backward(L_FM); clip grads to 1.0; optimizer.step(); zero_grad()
        # 9. Log scalars; periodically run eval; periodically checkpoint
```

**Critical implementation notes**:

- **Latent normalization for VAE encode**: the existing pipeline does `latents / latents_std + latents_mean` before VAE *decode*. For VAE *encode* during training, apply the inverse: `z_real = (vae_encode(x) - mean) * std`. Reuse constants from `vae.config.latents_mean` / `latents_std`. **Verify this matches what Wan was originally trained with — get it wrong and FM loss is meaningless.**
- **FM target velocity**: for `FlowMatchEulerDiscreteScheduler`, `v_target = noise - z_real` (linear interpolation flow) and `z_t = (1 - t_normalized) * z_real + t_normalized * noise`. Verify against the scheduler's `step` and `add_noise` implementations; the exact formulation can vary between flow-matching variants.
- **Timestep sampling**: default to logit-normal (mean 0, std 1) — same as Wan's pretraining. Optional: bias mean to 0.3 to emphasize higher noise levels (see TRAINING_PLAN.md section 6).
- **No VAE decode in the gradient graph** (this is the key simplification vs. v1 — confirm it's actually absent).
- **CFG off**: train conditional path only. Do not generate negative_prompt embeddings for L_FM.

**Validation step**:
- Run for 10 steps with a 2-sample dataset, batch size 1, no logging beyond stdout. Confirm:
  - No shape errors.
  - Loss values finite.
  - Only ControlNet parameters have non-zero gradients (`assert all(not p.requires_grad or p.grad is None for p in main_transformer.parameters())`).
  - Memory peaks within budget.

### Phase 6: Logging, checkpointing, evaluation

**Deliverable**: training script logs all required metrics, saves checkpoints, runs periodic evals.

**Modify** `train_controlnet.py`:

**Logging (every step or every 100 steps depending on metric cost)**:
- Scalars: `L_FM`, `learning_rate`, `grad_norm`, `controlnet_residual_norm` (mean L2 norm of the residual tuple).

**Checkpointing**:
- Save ControlNet state dict (only — not Wan, not VAE) every 2,000 steps to `checkpoints/controlnet_step_{step}.pt`.
- Keep last 3 checkpoints; delete older. Keep a separate `best_eval` checkpoint pointer.

**Evaluation**:
- Hold-out eval set: 10 (face_video, prompt) pairs in `eval_data/`. **No pseudo-targets needed for eval** — eval is open-ended generation.
- Function `run_eval(controlnet, step) -> dict`:
  1. Build a `WanTextToVideoControlnetPipeline` with the current ControlNet and the frozen base models.
  2. For each eval pair:
     - Extract landmark heatmaps via `prepare_landmark_heatmaps`.
     - Run inference (full denoising loop, 30–50 steps).
     - Save the resulting video to `eval_outputs/step_{step}/{pair_idx}.mp4`.
     - Save a side-by-side composite (generated | blurred-generated | blurred-face-video) for quick visual scan.
     - Log to W&B as a video artifact.
  3. Compute blurred-face cosine similarity (Phase 7).
  4. Return dict of metrics.
- Cadence: every 1k steps in Stage 1 (sanity); every 2k–5k in Stage 2 (main).

**Validation step**:
- Run training for 200 steps with logging and one eval at step 100. Verify W&B receives scalar metrics, gradient norms, and at least one video artifact.

### Phase 7: Quantitative eval — blurred-face similarity

**Deliverable**: a function returning a single scalar measuring how face-like the generated video looks when blurred.

**New file**: `eval_metrics.py`

**Specification**:

- `blurred_face_similarity(generated_video: torch.Tensor, reference_face_video: torch.Tensor, face_embedding_model) -> float`.
  1. Apply Gaussian blur (sigma 16 at output resolution) to both videos.
  2. For each frame: detect face, extract ArcFace embedding (or InsightFace).
  3. Compute cosine similarity between corresponding-frame embeddings.
  4. Return mean cosine similarity across frames where both detections succeed.
  5. If detection fails on >50% of frames, return -1 (signal failure).

**Dependencies**: `insightface` or `facenet-pytorch`. Add to `requirements_train.txt`.

**Validation step**:
- Run on a known face image vs. itself (similarity should approach 1.0) and on random noise (should approach 0 or return -1). Sanity-check before relying on it.

### Phase 8: Inference pipeline update for 6-channel heatmaps

**Deliverable**: the existing pipeline accepts 6-channel heatmap input at inference time.

**File to modify**: `wan_t2v_controlnet_pipeline.py`

**Changes**:
- Add new parameter to `__call__`: `landmark_heatmap_video: Optional[torch.Tensor] = None` of shape `(1, 6, T, H, W)`.
- If `controlnet_latents is None and landmark_heatmap_video is not None`, use `landmark_heatmap_video` directly (already in the right shape and dtype) instead of going through `prepare_controlnet_frames`.
- Keep the existing `controlnet_frames` PIL-image path intact for backward compatibility.

**Do not** remove `prepare_controlnet_frames` or break `inference/cli_demo.py`. The new path is additive.

**Validation step**:
- Call the pipeline with a hand-crafted heatmap tensor (e.g., a single Gaussian blob in the center) and confirm a video is produced without errors. Behavior need not be meaningful — just confirm wiring.

### Phase 9: End-to-end smoke test

**Deliverable**: train for 500 steps on a 5-tuple dataset; run an eval; confirm sensible loss curve and at least one video output.

This is not a real training run — it's the final integration test before launching real training.

**Validation criteria**:
- L_FM decreases meaningfully over 500 steps.
- No NaNs, no shape errors, no OOMs.
- Eval runs successfully and produces a video.
- All metrics logged to W&B / TB.

After Phase 9 passes, real training runs are an operational concern, not an implementation concern.

## 3. File Inventory After Implementation

### New files

- `landmark_extraction.py` — MediaPipe-based heatmap extraction.
- `pseudo_target_generation.py` — Optical flow extraction and warping for offline pseudo-target generation.
- `train_dataset.py` — `HiddenFacePseudoTargetDataset` and dataloader utilities.
- `train_controlnet.py` — main training script.
- `eval_metrics.py` — blurred-face cosine similarity.
- `data_prep/generate_all_pseudo_targets.py` — batch script for offline pseudo-target generation.
- `data_prep/generate_ptdiffusion_stills.py` — batch script orchestrating PTDiffusion rendering (uses existing PTDiffusion code; this repo just calls it).
- `manifest_train.json` — training tuples manifest (data, generated as part of dataset prep).
- `manifest_eval.json` — eval pairs manifest.
- `requirements_train.txt` — additional training dependencies (mediapipe, accelerate, wandb, insightface, decord, torchvision RAFT).

### Modified files

- `wan_controlnet.py` — add `init_from_pretrained` helper. Architecture unchanged.
- `wan_t2v_controlnet_pipeline.py` — add `landmark_heatmap_video` arg to `__call__`; route around `prepare_controlnet_frames` when used.

### Untouched files

- `wan_transformer.py` — no changes needed; already supports ControlNet residual injection.
- `wan_teacache.py` — no changes; off during training.
- `inference/cli_demo.py` — no changes; existing inference path preserved.

## 4. Data Preparation (Out-of-Band)

Before any training, generate the data. This is one-time work, not part of the training script.

### 4.1 Face videos

**Source**: 100–300 head-movement clips from any source (AI-generated, webcam-captured, stock).

- Crop/center on the face.
- Resize to a working resolution (256×256 for bulk training, 480×832 for final fine-tune).
- Save as `data/face_videos/{idx:04d}.mp4`.

### 4.2 PTDiffusion stills

**Script**: `data_prep/generate_ptdiffusion_stills.py`

- For each face video, extract the canonical frame (default: frame 0).
- For each (canonical_face, prompt) pair, call PTDiffusion to render a blended still.
- Save as `data/ptdiffusion_stills/{face_idx:04d}_{prompt_slug}.png`.

This step requires the PTDiffusion codebase available externally; this script is a thin wrapper.

### 4.3 Pseudo-target videos

**Script**: `data_prep/generate_all_pseudo_targets.py` (uses `pseudo_target_generation.py` from Phase 2).

- For each (face_video, ptdiffusion_still, prompt) tuple, run `generate_pseudo_target_video`.
- Save as `data/pseudo_targets/{face_idx:04d}_{prompt_slug}.pt` (fp16 tensor `(3, T, H, W)`).
- Append entry to `manifest_train.json`.

### 4.4 Eval set

**Script**: `data_prep/build_eval_set.py`

- Hold out 10 (face_video, prompt) pairs to `data/eval/`. **These must not appear in the training manifest.**
- Pre-compute landmark heatmaps for them (cache to `.pt` files) for fast eval.
- Append to `manifest_eval.json`.

## 5. Implementation Order Summary

Strictly sequential — each phase blocks the next:

1. **Phase 1**: Landmark extraction (independent; testable in isolation).
2. **Phase 2**: Pseudo-target generation (depends on having PTDiffusion stills and face videos available; independent of training code).
3. **Phase 3**: ControlNet 6-channel support (independent of Phases 1–2 except for the validation step).
4. **Phase 4**: Dataset (depends on Phases 1, 2 — needs both heatmaps and pseudo-targets).
5. **Phase 5**: Training script (depends on Phases 3, 4).
6. **Phase 6**: Logging + checkpointing + eval (depends on Phase 5).
7. **Phase 7**: Quantitative eval metric (independent; can be developed in parallel with Phase 6).
8. **Phase 8**: Inference pipeline update (independent; needed for Phase 6's eval).
9. **Phase 9**: End-to-end smoke test (depends on all prior).

Out-of-band data prep (PTDiffusion rendering, face video collection) can happen in parallel with Phases 1, 3, 7, 8.

## 6. What Claude Should Do When Executing This Plan

For each phase:

1. Read the relevant existing files in full before modifying.
2. Implement only what the phase specifies — do not add features beyond the phase's deliverable.
3. Run the validation step explicitly. If validation fails, debug before proceeding.
4. Commit per-phase (or per logical unit within a phase) with a clear message.
5. Do not modify the inference path beyond Phase 8's narrow additive change.
6. Do not add backwards-compatibility shims for code that does not exist yet.
7. Defer to `TRAINING_PLAN.md` for any conceptual or hyperparameter question; defer to this file for any code-structure question.

## 7. Open Questions to Resolve Before Phase 5

These need user input before launching real training:

- **Pretrained ControlNet checkpoint**: is one available to warm-start from, or train from scratch? Affects Phase 3's `init_from_pretrained` priority.
- **Training resolution**: 256×256 confirmed for bulk, 480×832 for final fine-tune?
- **Hardware**: single A100? Multiple GPUs (affects accelerate config)?
- **Face video source and identity matching**: re-render PTDiffusion per face video, or use existing stills with matched face videos?
- **Logging backend**: W&B project + entity, or TensorBoard local logs?
- **PTDiffusion compute budget**: how many (face, prompt) pairs can be rendered? This bounds dataset size.

These do not block Phases 1, 2, 3, 7, or 8.

# Implementation Plan: BETA — Static-Video ControlNet Sanity Test

This is a minimal, time-boxed sanity test before committing to the full warped-pseudo-target pipeline in `IMPLEMENTATION_PLAN.md`. Goal: train the **existing 3-channel `WanControlnet` from a cold start** on (canny-face → PTDiffusion-still) static-video pairs and verify whether any learning signal emerges.

**No structural changes to `WanControlnet`, `CustomWanTransformer3DModel`, or the inference pipeline.** Only training-side scripts are added.

If this run shows visible loss descent and any structure transfer in inference samples, we proceed to the full pipeline. If the loss is flat throughout, we will rerun with a warm start (HED-A14B checkpoint) before concluding the approach is broken — cold start is the simplest and most defensible *first* run, but a flat curve is ambiguous (broken approach vs. residuals haven't woken up yet).

### Why cold start (Scenario C)

We picked cold init for the first run because:
- Simplest to justify methodologically (no pretrained-prior bias on results).
- No download or path management for a warm-start checkpoint.
- A clean descent under cold start is the strongest possible positive signal.

The known trade-off: cold-init ControlNet output projections are zeroed (`zero_module`), so residuals are exactly 0 at step 0 and ramp up slowly. Loss is expected to be **roughly flat for the first ~1k–3k steps** before any meaningful descent. The training card explicitly tracks this expectation so we don't misread a slow start as failure.

---

## 0. Locked decisions

| Decision | Value | Rationale |
|---|---|---|
| Base model | `Wan-AI/Wan2.2-T2V-A14B-Diffusers` | user |
| Expert trained | **High-noise only** (`transformer` subfolder) | fits 1× A40, structural learning lives here |
| Other expert | Not loaded during training | `transformer_2` (low-noise) untouched |
| ControlNet init | **Cold start.** Load architecture config from `TheDenk/wan2.2-t2v-a14b-controlnet-hed-v1` (`config.json` only), instantiate fresh weights via `WanControlnet.from_config(...)`. Output projections zeroed by `zero_module()` inside `__init__` → residuals = 0 at step 0. | Simplest first run, no prior bias |
| Input channels | 3 (Canny edges from `cv2.Canny`, RGB-replicated) | no architecture change |
| Resolution | 512×512 | matches target images; input faces resized 528→512 |
| Frame count `T` | **9 (4n+1)** | default for BETA — half the per-step cost vs T=17; T=17 is a stretch goal after the smoke gives a real per-step number |
| Hardware | 1× A40 (45GB) | user preference (queue) |
| Optimizer | `bitsandbytes.optim.AdamW8bit`, lr=1e-4, wd=0.01 | optimizer-state-only quantization → no quality loss |
| Mixed precision | bf16, **respecting `_keep_in_fp32_modules`** (norms, `time_embedder`, `scale_shift_table`) | blanket `.to(bf16)` would silently demote numerically fragile layers — see §4.2 |
| Gradient checkpointing | **Both ControlNet and the (frozen) transformer** | memory fit on 1× A40 — frozen transformer still retains activations for backward through `cn_for_tx` (which requires grad), so checkpoint to recompute on demand |
| Loss | Flow-matching MSE (`v_pred` vs `v_target`) | matches inference scheduler |
| Timestep sampling | High-noise regime only (see §5.1) | matches expert |
| Dataset | 100 faces × 100 prompts = 10000 pairs, **1 epoch** | user |
| Batch size | 1, no grad accumulation | memory fit |
| Pre-cache | canny + VAE latents + T5 embeds → `cache/` | precompute fine per user |
| Logging | wandb, project `wan-controlnet-beta`, every step | user |
| Checkpointing | **Every 2k steps + final** → `checkpoints/beta-001_step{N}.safetensors` and `beta-001_final.safetensors` | preempt-/OOM-safe on a 24h job; cost is ~5GB × 5 = 25GB |
| Inference smoke at end | 1 (face, prompt) pair → `outputs/beta_final.mp4` | quick visual check |

---

## 1. Repo layout & file inventory

### Final repo layout (after Phase 0 cleanup)

```
wan2.2-controlnet/
├── README.md
├── .gitignore                               # ← updated to ignore data/cache/checkpoints/outputs/logs
├── requirements.txt
├── requirements_train_beta.txt              # NEW
│
├── wan_controlnet.py                        # existing — stays at root
├── wan_transformer.py                       # existing — stays at root
├── wan_t2v_controlnet_pipeline.py           # existing — stays at root
├── wan_teacache.py                          # existing — stays at root (imported by pipeline)
│
├── docs/                                    # NEW — plans live here
│   ├── IMPLEMENTATION_PLAN.md
│   ├── IMPLEMENTATION_PLAN_BETA.md          # this file
│   └── TRAINING_PLAN.md
│
├── training/                                # NEW — all BETA training code
│   ├── input_prompts.py                     # moved from root
│   ├── precompute_beta.py
│   ├── dataset_beta.py
│   ├── train_beta.py
│   ├── smoke_test_beta.py
│   └── autofill_card.py
│
├── slurm/                                   # NEW — sbatch job scripts
│   ├── smoke_test.sbatch
│   ├── precompute.sbatch
│   └── train_beta.sbatch
│
└── training_cards/                          # NEW
    ├── TEMPLATE.md
    └── beta-001.md
```

Why model files stay at root: `wan_t2v_controlnet_pipeline.py` uses bare imports (`from wan_controlnet import WanControlnet`). Moving them into `src/` would force editing the pipeline, which violates our "no structural changes" rule. Scripts in `training/` reach the root modules via `sys.path.insert(0, str(Path(__file__).parent.parent))` (same trick `inference/cli_demo.py` uses today).

### Phase 0 cleanup (one-time, before any new code)

Delete from the repo:
- `inference/` — `cli_demo.py` is demo-only; our end-of-run inference smoke is self-contained.
- `resources/` — demo videos used only by `cli_demo.py`.

Keep:
- `wan_teacache.py` — imported by the pipeline (`from wan_teacache import TeaCache`); off at training time but cannot be deleted without touching the pipeline.

Move:
- `IMPLEMENTATION_PLAN*.md`, `TRAINING_PLAN.md` → `docs/`
- `input_prompts.py` → `training/`

### New files (created during phases below)

- `training/precompute_beta.py` — extracts Canny, encodes VAE latents and T5 embeds; writes to `cache/`.
- `training/dataset_beta.py` — `BetaPairDataset`: loads from `cache/` only.
- `training/train_beta.py` — single-GPU training loop (cold init).
- `training/smoke_test_beta.py` — runs precompute on 5 pairs + 5 training steps; designed for srun verification before launching the real job.
- `training/autofill_card.py` — fills `<AUTO:key>` markers in a training card from a sibling `_results.json` (see §6.5).
- `slurm/{smoke_test,precompute,train_beta}.sbatch` — cluster job scripts (see §11).
- `training_cards/TEMPLATE.md` — blank template for future runs/ablations.
- `training_cards/beta-001.md` — this run's card; pre-filled before launch, machine fields auto-filled at end.
- `requirements_train_beta.txt` — adds `bitsandbytes`, `wandb`, `opencv-python` (anything not already in `requirements.txt`).

### Modified files
- `.gitignore` — adds `cache/`, `checkpoints/`, `outputs/`, `logs/`, `data/`, `wandb/`, `*.pt`, `*.safetensors`, etc.

### Untouched files
`wan_controlnet.py`, `wan_transformer.py`, `wan_t2v_controlnet_pipeline.py`, `wan_teacache.py`.

---

## 2. Phase 1 — Precompute script (`training/precompute_beta.py`)

### Purpose
Move all VAE encoding, T5 encoding, and Canny extraction out of the training loop. Training loop loads small `.pt` files only.

### CLI
```
python -m training.precompute_beta \
    --input_faces_dir $WORK/wan-beta/data/input_faces \
    --targets_dir   $WORK/wan-beta/data/targets \
    --output_dir    $WORK/wan-beta/cache \
    --base_model_path $WAN_MODEL \
    --height 512 --width 512 --num_frames 9 \
    [--limit N]   # optional, for smoke test
```

### Steps

1. **Pair discovery.** Enumerate `targets/face_{idx}_{slug}.jpg`. For each:
   - Verify `input_faces/face_{idx}.png` exists.
   - Verify `slug` is a key in `PROMPTS_BATCH_1 | PROMPTS_BATCH_2` from `input_prompts.py`.
   - Skip with a warning if either check fails. Final manifest contains only validated pairs.

2. **Canny cache** (per *unique face*, ~100 entries):
   - Load PNG, resize to 512×512 (PIL `LANCZOS`), convert to grayscale.
   - `edges = cv2.Canny(gray, 100, 200)` — defaults; document choice in training card.
   - Stack to 3 channels: `np.stack([edges, edges, edges], axis=0)` → `(3, H, W)` uint8.
   - Save `cache/canny/face_{idx}.pt`.

3. **VAE latent cache** (per *unique target*, ~10000 entries):
   - Load JPG (already 512×512), to tensor in `[-1, 1]` shape `(3, H, W)`.
   - Replicate temporally: `(3, T, H, W)` then add batch → `(1, 3, T, H, W)`.
   - Encode through Wan VAE: `z = vae.encode(x).latent_dist.sample()` (or `.mean` for determinism — pick `.mean` for reproducibility).
   - **Apply Wan latent normalization (encode-side):**
     - Inference does decode-side `z * latents_std + latents_mean`. Encode-side inverse is `z = (z - latents_mean) / latents_std`.
     - **VERIFY by reading `wan_t2v_controlnet_pipeline.py`** — the exact constants and direction. Wrong sign = silent data corruption, FM loss is meaningless. The self-test in step 4 (below) is a hard gate that catches this.
   - Save `cache/latents/face_{idx}_{slug}.pt` shape `(C_lat, T_lat, H_lat, W_lat)` fp16.

4. **Latent normalization self-test (HARD GATE).** Before encoding any production tensors, run this round-trip on a single sample image and abort the script if it fails:
   ```python
   x = load_one_target_jpg_to_minus1_plus1()  # shape (1, 3, T, H, W)
   z = vae.encode(x).latent_dist.mean
   z_norm = (z - latents_mean) / latents_std            # encode-side normalization
   z_unnorm = z_norm * latents_std + latents_mean       # decode-side denormalization (must match pipeline)
   x_rec = vae.decode(z_unnorm).sample
   mse = F.mse_loss(x_rec.float(), x.float()).item()
   assert mse < 1e-2, f"VAE round-trip MSE {mse:.4f} too high — latent norm constants likely wrong"
   ```
   Threshold `1e-2` accounts for VAE reconstruction error (lossy autoencoder) but is far below what wrong-sign or wrong-constant errors produce (those typically yield MSE > 0.5). Log the actual MSE for the training card; this is also exercised by the smoke test since smoke calls `precompute_beta.main()`.

5. **Prompt embeds cache** (per *unique slug*, ~100 entries):
   - For each slug, look up the long prompt from `input_prompts.py`.
   - T5-tokenize and encode using the same tokenizer/encoder as inference (`UMT5EncoderModel` from `base_model_path` subfolder `text_encoder`).
   - Save `cache/prompts/{slug}.pt` shape `(L, D)` bf16.
   - Also save `cache/prompts_negative.pt` for `"bad quality, worst quality"` — used only by the optional end-of-run inference smoke (training itself is conditional-only, no CFG).

6. **Manifest.** Write `cache/manifest.json`:
   ```json
   [
     {
       "face_idx": 52,
       "slug": "sky",
       "canny_path": "cache/canny/face_52.pt",
       "latent_path": "cache/latents/face_52_sky.pt",
       "prompt_path": "cache/prompts/sky.pt"
     },
     ...
   ]
   ```

### Notes
- VAE and T5 are loaded just for this script and freed at the end. They never appear in the training loop.
- **Disk estimate:** ~10000 latents × ~5MB ≈ **50GB**. If too large, fall back to caching only canny + embeds and encoding latents per-batch (adds ~0.3-0.5s/step). Decide before running bulk precompute.
- `--limit N` flag truncates to first N pairs — used by smoke test.
- Logs at end: counts cached per category, total disk used, 5 random sanity reads.

### Validation
- Manual: check 5 random `cache/canny/*.pt` decode back to a recognizable face outline.
- Automated (in script): assert all manifest paths exist and load with expected shapes.

---

## 3. Phase 2 — Dataset (`dataset_beta.py`)

### `BetaPairDataset(Dataset)`

```python
class BetaPairDataset(Dataset):
    def __init__(self, cache_dir: str, num_frames: int):
        self.records = json.load(open(f"{cache_dir}/manifest.json"))
        self.num_frames = num_frames

    def __len__(self): return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        canny = torch.load(r["canny_path"])               # (3, H, W) uint8
        canny = canny.float() / 127.5 - 1.0               # → [-1, 1]
        canny = canny.unsqueeze(1).expand(-1, self.num_frames, -1, -1)  # (3, T, H, W)
        latent = torch.load(r["latent_path"]).to(torch.bfloat16)        # (C, T_lat, H_lat, W_lat)
        prompt_embeds = torch.load(r["prompt_path"]).to(torch.bfloat16) # (L, D)
        return {"canny": canny.to(torch.bfloat16), "latent": latent, "prompt_embeds": prompt_embeds}
```

DataLoader: `batch_size=1, num_workers=2, persistent_workers=True, pin_memory=True, shuffle=True`.

---

## 4. Phase 3 — Training script (`training/train_beta.py`)

### CLI
```
python -m training.train_beta \
    --cache_dir       $WORK/wan-beta/cache \
    --base_model_path $WAN_MODEL \
    --controlnet_config_repo TheDenk/wan2.2-t2v-a14b-controlnet-hed-v1 \
    --output_dir      $WORK/wan-beta/checkpoints \
    --card_path       training_cards/beta-001.md \
    --wandb_project   wan-controlnet-beta \
    --run_name        beta-001 \
    --num_frames 9 --height 512 --width 512 \
    --lr 1e-4 --weight_decay 0.01 --grad_clip 1.0 \
    --num_epochs 1 \
    --checkpoint_every 2000
```

`--controlnet_config_repo` is used for **architecture config only** (`config.json`). No weights are loaded from it. Can be a HF repo ID or a local path containing `config.json`. We reuse the HED A14B config to guarantee architectural compatibility with the A14B transformer.

### Pseudocode

```python
def main(cfg):
    wandb.init(project=cfg.wandb_project, name=cfg.run_name, config=vars(cfg))

    # --- Models ---
    # HIGH-NOISE expert only:
    transformer = CustomWanTransformer3DModel.from_pretrained(
        cfg.base_model_path, subfolder="transformer", torch_dtype=torch.bfloat16,
    )
    transformer.requires_grad_(False).eval()
    # Frozen, but cn_for_tx (input) requires grad → without checkpointing the
    # transformer's per-block activations stay live for backward and OOM the
    # A40. Recompute on demand instead.
    transformer.enable_gradient_checkpointing()

    # Cold init: load architecture config from HED checkpoint (no weights),
    # instantiate fresh. Output projections are zeroed by zero_module() inside
    # __init__ → residuals are exactly 0 at step 0.
    config = WanControlnet.load_config(cfg.controlnet_config_repo)
    controlnet = WanControlnet.from_config(config)
    cast_respecting_fp32_modules(controlnet, torch.bfloat16)  # see §4.2
    controlnet.enable_gradient_checkpointing()
    controlnet.train()

    transformer.to("cuda"); controlnet.to("cuda")

    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        cfg.base_model_path, subfolder="scheduler",
    )

    # --- Optimizer ---
    optimizer = bnb.optim.AdamW8bit(
        controlnet.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay,
    )

    # --- Data ---
    dataset = BetaPairDataset(cfg.cache_dir, num_frames=cfg.num_frames)
    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2,
                        persistent_workers=True, pin_memory=True)

    # --- Training loop ---
    for epoch in range(cfg.num_epochs):
        for step, batch in enumerate(loader):
            canny = batch["canny"].to("cuda")            # (1, 3, T, H, W)
            z_real = batch["latent"].to("cuda")          # (1, C, T_lat, H_lat, W_lat)
            prompt_embeds = batch["prompt_embeds"].to("cuda")  # (1, L, D)

            # 1. Sample noise + timestep in HIGH-NOISE regime only
            noise = torch.randn_like(z_real)
            t_idx, sigma = sample_high_noise_timestep(scheduler)  # see §5.1
            t = scheduler.timesteps[t_idx].to("cuda")

            # 2. Build noisy latent and FM target
            z_t = (1 - sigma) * z_real + sigma * noise
            v_target = noise - z_real

            # 3. ControlNet forward
            controlnet_states = controlnet(
                hidden_states=z_t, timestep=t,
                encoder_hidden_states=prompt_embeds,
                controlnet_states=canny,
            )

            # 4. Transformer forward (frozen, but gradient flows through residuals)
            v_pred = transformer(
                hidden_states=z_t, timestep=t,
                encoder_hidden_states=prompt_embeds,
                controlnet_states=controlnet_states,
                controlnet_weight=1.0, controlnet_stride=3,
            ).sample

            # 5. Loss
            loss = F.mse_loss(v_pred.float(), v_target.float())

            # 6. Backward — only ControlNet params accumulate grad
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(controlnet.parameters(), cfg.grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            # 7. Log every step
            global_step = step + epoch * len(loader)
            wandb.log({
                "loss": loss.item(),
                "grad_norm": grad_norm.item(),
                "lr": optimizer.param_groups[0]["lr"],
                "controlnet_residual_norm": mean_residual_l2(controlnet_states),
                "timestep": t.item(),
                "step": global_step,
                "gpu_mem_gb": torch.cuda.max_memory_allocated() / 1e9,
            })

            # 8. Periodic checkpoint (preempt-/OOM-safe)
            if (global_step + 1) % cfg.checkpoint_every == 0:
                ckpt_path = f"{cfg.output_dir}/{cfg.run_name}_step{global_step + 1}.safetensors"
                save_file(controlnet.state_dict(), ckpt_path)

    # --- Save final checkpoint ---
    save_file(controlnet.state_dict(), f"{cfg.output_dir}/{cfg.run_name}_final.safetensors")

    # --- End-of-run inference smoke (optional but recommended) ---
    # Build a WanTextToVideoControlnetPipeline with this controlnet + frozen transformer (high-noise)
    # + load transformer_2 (low-noise) for inference only + VAE + T5
    # Run on 1 (face, prompt) held-out pair, save mp4 to outputs/beta_final.mp4
    # Log video to wandb
```

### 4.1 Critical implementation notes

- **Freeze direction.** `transformer.requires_grad_(False)` zeros `requires_grad` on every transformer parameter. Activations still build a graph through it; gradients flow back into `controlnet_states` (which were computed with `requires_grad=True` params), so ControlNet learns. Do **not** wrap the transformer call in `torch.no_grad()` — that would cut the graph.
- **Cold-start gradient flow.** With `zero_module()`-initialized output projections, the residuals are 0 at step 0, but the gradient w.r.t. those zero weights is `input_activation × output_grad`, which is **non-zero**. So the zero-init layers do begin training from step 1 — just slowly, because their effect on the loss starts tiny and grows as their weights move away from zero. Expect the loss curve to look roughly flat for the first ~1k–3k steps before meaningful descent.
- **Verification assert** after step 1 (not step 0 — we need at least one optimizer step):
  ```python
  assert all(p.grad is None for p in transformer.parameters())
  # At least SOME controlnet params must have nonzero grad. Output-projection
  # layers (zero_module-init) get small but nonzero grads from step 1.
  assert any(
      p.grad is not None and p.grad.abs().sum() > 0
      for p in controlnet.parameters()
  )
  ```
- **High-noise timestep sampling**: see §4.3 below — needs implementation choice.
- **Latent normalization**: must match `training/precompute_beta.py`'s convention. Verify against the inference pipeline's decode-side denormalization. Wrong = silent failure.
- **Memory tripwire**: log `torch.cuda.max_memory_allocated()` every step. If > 43GB, abort with a clear error suggesting `T=5`, smaller resolution, or 2-GPU DDP.
- **`controlnet_stride=3`** matches the inference default and the HED-A14B ControlNet's residual layout (we inherit its config). Don't change for BETA.
- **No CFG.** Conditional path only. Don't load or use a negative prompt during training. The cached `prompts_negative.pt` is only for end-of-run inference.
- **Results JSON write.** Training script writes `training_cards/{run_id}_results.json` at start (status=`running`, start time) and updates at end (status=`completed`/`failed`, end time, wall-time, final loss, peak GPU mem, wandb URL, trainable param count, boundary sigma actually used, git SHA). Final step calls `training/autofill_card.py` to substitute `<AUTO:key>` markers in the card. See §6.5.

### 4.2 fp32-respecting cast helper

`WanControlnet` declares two class lists (`wan_controlnet.py:74-76`):
- `_keep_in_fp32_modules = ["time_embedder", "scale_shift_table", "norm1", "norm2", "norm3"]`
- `_skip_layerwise_casting_patterns = ["patch_embedding", "condition_embedder", "norm"]`

These name layers diffusers wants kept in fp32 when the rest of the model goes to bf16 (norms suffer catastrophic cancellation in bf16; sinusoidal time embeddings lose resolution; scale_shift tables multiply by tiny numbers). **A blanket `.to(torch.bfloat16)` ignores both lists** — only diffusers' own loaders (`from_pretrained(torch_dtype=...)`, `enable_layerwise_casting`) honor them. Cold init via `from_config` does not go through those loaders.

Implement and call this once after construction:

```python
def cast_respecting_fp32_modules(model: nn.Module, dtype: torch.dtype) -> None:
    """Cast every parameter to `dtype` except those whose qualified name
    matches a substring in `model._keep_in_fp32_modules`."""
    keep = getattr(model, "_keep_in_fp32_modules", []) or []
    skipped, casted = [], []
    for name, param in model.named_parameters():
        if any(k in name for k in keep):
            skipped.append(name)
        else:
            param.data = param.data.to(dtype)
            casted.append(name)
    # Buffers too (norm running stats, etc.)
    for name, buf in model.named_buffers():
        if any(k in name for k in keep):
            continue
        buf.data = buf.data.to(dtype)
    print(f"[cast] {len(casted)} params → {dtype}; {len(skipped)} kept fp32 ({skipped[:3]}...)")
```

Post-cast assertions (sanity, against silent regressions if `_keep_in_fp32_modules` is edited upstream):

```python
# At least one fp32 module survived:
fp32_params = [n for n, p in controlnet.named_parameters() if p.dtype == torch.float32]
assert any("norm" in n or "time_embedder" in n or "scale_shift" in n for n in fp32_params), \
    "Expected norm/time_embedder/scale_shift_table params to remain fp32"
# At least one bf16 module exists:
assert any(p.dtype == torch.bfloat16 for p in controlnet.parameters()), \
    "Expected most controlnet params to be bf16"
```

These asserts run once at startup, cost nothing, and turn a silent precision regression into a loud failure.

### 4.3 High-noise timestep sampling

Wan 2.2 A14B uses a mixture-of-experts split between two transformers along the noise schedule. The boundary sigma may be exposed in `transformer.config` or the model card.

**Step at implementation time**:
1. Read `transformer.config` after loading; look for fields like `boundary_ratio`, `boundary_sigma`, or similar.
2. If present: sample timesteps `t` such that `sigma(t) >= boundary`.
3. If absent: default to **upper 50% of `scheduler.timesteps`** (high-noise = early denoising = high sigma).
4. Document the chosen rule and boundary value in `training_cards/beta-001.md`.

A simple implementation is to filter `scheduler.timesteps` once at startup into a `high_noise_timesteps` tensor, then `t_idx = randint(0, len(high_noise_timesteps))`.

---

## 5. Phase 4 — Smoke test (`training/smoke_test_beta.py`)

### Purpose
A single Python script the user runs in srun on the cluster to verify the entire pipeline compiles and runs end-to-end on a tiny subset, before scheduling the real 24h job. Catches missing deps, HF auth issues, OOMs, shape mismatches.

### Spec
```python
# 1. Pick 5 hardcoded (face_idx, slug) tuples whose files are known to exist.
# 2. Run training.precompute_beta.main() with --limit 5 → $WORK/wan-beta/cache/_smoke/
#    This implicitly exercises the latent-norm round-trip self-test (§2 step 4),
#    which aborts the run if the VAE normalization constants are wrong.
# 3. Assert all 5 expected cache files exist with expected shapes.
# 4. Run training.train_beta.main() with cache_dir=$WORK/wan-beta/cache/_smoke/,
#    num_epochs=4 (5 pairs × 4 epochs = 20 steps — enough to get a stable
#    per-step wall-time estimate after warmup).
# 5. Per-step wall-time measurement:
#    - Skip the first 3 steps (CUDA warmup, autograd graph build, allocator stabilization).
#    - Wrap each subsequent step in:
#          torch.cuda.synchronize(); t0 = time.perf_counter()
#          ... full training step (forward + backward + optimizer.step) ...
#          torch.cuda.synchronize(); dt = time.perf_counter() - t0
#    - Collect dts for steps 4..20 (17 samples).
#    - Report: median, p90, mean ± std.
# 6. After step 1: assert no NaN loss, GPU mem < 44GB,
#    transformer params have grad=None, controlnet params have at least one nonzero grad
#    (note: most grads will be tiny under cold init — that's fine, "any nonzero" suffices).
# 7. After step 5 of training (i.e. after warmup): assert post-cast dtypes are correct
#    (at least one fp32 param matching norm/time_embedder/scale_shift; majority bf16).
# 8. End-of-script summary print:
#       SMOKE TEST PASSED
#       per-step wall-time (median): X.XXs
#       per-step wall-time (p90):    X.XXs
#       projected wall-time for 10000 steps @ median: HH:MM:SS
#       projected wall-time for 10000 steps @ p90:    HH:MM:SS
#       peak GPU memory: XX.XX GB
#    Also append these numbers to training_cards/beta-001_smoke_results.json so they
#    can be referenced when filling beta-001.md before launch.
# 9. Hard fail conditions: NaN loss, GPU mem ≥ 44GB, transformer param has grad,
#    no controlnet param has grad, fp32 dtype assert fails, or median per-step > 8.5s
#    (would push the real run over 24h — see §7). Exit 1 with a clear error.
# 10. Soft warning conditions (print but exit 0): p90 per-step > 8.5s but median ≤ 8.5s
#     (variance suggests possible borderline; user decides whether to drop T=9 → T=5
#     before launching the 24h job).
```

### Suggested srun invocation (user runs this)
```
srun --gres=gpu:1 --mem=64G --time=00:20:00 --partition=<your_gpu_partition> \
    python -m training.smoke_test_beta
```

Or via the prepared sbatch (see §11): `sbatch slurm/smoke_test.sbatch`.

### Hard timing budget
Smoke test should complete in <15 min (load models ~5min, precompute 5 pairs ~2min, 5 training steps ~1min, end-of-script teardown).

---

## 6. Phase 5 — Training cards

### Folder
`training_cards/`

### Files
- `TEMPLATE.md` — empty card with all sections, ready to copy.
- `beta-001.md` — this run's card. Pre-filled before launch; results appended after.

### Card format

```markdown
---
run_id: beta-001
---

# Training Card — {run_id}

## 1. Goal
One sentence.

## 2. Hypothesis & success criteria
- What outcome confirms the hypothesis?
- What outcome rejects it?
- Quantitative bar (e.g., "FM loss drops by ≥ X% over the run", "qualitative: face structure visible in ≥ 3 / 5 inference samples").

## 3. Setup

### Base model
- Repo: `Wan-AI/Wan2.2-T2V-A14B-Diffusers`
- Expert(s) trained against: high-noise (`transformer`)
- Other components (frozen): `transformer_2`, VAE, T5

### ControlNet
- Architecture: `WanControlnet` (unchanged)
- Initialization: **Cold start.** Architecture config loaded from `TheDenk/wan2.2-t2v-a14b-controlnet-hed-v1` (`config.json` only); weights freshly initialized. Output projections zeroed by `zero_module()` → residuals = 0 at step 0.
- Input channels: 3 (Canny edges, RGB-replicated)
- Trainable parameter count: `<AUTO:trainable_params>`
- Gradient checkpointing: ON for both ControlNet and the (frozen) transformer (the latter is frozen but its activations must persist for backward through `cn_for_tx`, so checkpoint to recompute)

### Data
- Source faces: `data/input_faces/` (100 PNGs, 528×528 → resized 512×512)
- Source targets: `data/targets/` (10000 PTDiffusion JPGs at 512×512, named `face_{idx}_{slug}.jpg`)
- Prompt dictionary: `training/input_prompts.py` (`PROMPTS_BATCH_1 | PROMPTS_BATCH_2`)
- Pair count after validation: `<AUTO:pair_count>`
- Canny preprocessing: `cv2.Canny(gray, 100, 200)`, stacked to 3 channels
- Resolution: 512×512
- Frame count `T`: 9 (replicated still → static video)
- Cache directory: `$WORK/wan-beta/cache`, `<AUTO:cache_disk_gb>` GB on disk

### Optimization
- Loss: flow-matching MSE (`v_pred = noise - z_real`)
- Optimizer: `bitsandbytes.optim.AdamW8bit`
- LR: 1e-4, weight decay 0.01
- LR schedule: constant (no warmup; single-pass run)
- Batch size: 1, no grad accumulation
- Grad clipping: 1.0
- Mixed precision: bf16
- Timestep sampling: high-noise regime only (rule used: `<AUTO:high_noise_rule>`; boundary sigma: `<AUTO:boundary_sigma>`)

### Hardware
- GPUs: 1× NVIDIA A40 (45GB)
- Smoke-test per-step wall-time (median / p90): `<AUTO:smoke_step_median>`s / `<AUTO:smoke_step_p90>`s
- Estimated wall-time (10000 steps @ smoke median): `<AUTO:smoke_projected_wall_time>`
- Actual wall-time: `<AUTO:wall_time>`
- Cluster / partition: `<AUTO:cluster_partition>`

### Logging & checkpointing
- wandb project: `wan-controlnet-beta`, run name: `beta-001`
- wandb URL: `<AUTO:wandb_url>`
- Per-step metrics: loss, grad_norm, lr, controlnet_residual_norm, timestep, gpu_mem_gb
- Periodic checkpoints: `$WORK/wan-beta/checkpoints/beta-001_step{2000,4000,...}.safetensors`
- Final checkpoint: `$WORK/wan-beta/checkpoints/beta-001_final.safetensors`
- End-of-run inference: 1 (face, prompt) held-out pair → `$WORK/wan-beta/outputs/beta-001_final.mp4`

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

```

The `TEMPLATE.md` is the same skeleton with `<AUTO:key>` markers in place and all human fields blank.

---

## 6.5 Autofill mechanism (`training/autofill_card.py`)

### Marker convention
Pre-launch, the human fills all judgment fields (Goal, Hypothesis, qualitative Results entries). All machine-derivable fields are written as **explicit autofill markers** of the form `<AUTO:key>` (e.g., `<AUTO:wall_time>`, `<AUTO:final_loss>`).

After the run, the training script:
1. Writes a sibling JSON file `training_cards/{run_id}_results.json` with all known machine values.
2. Calls `training/autofill_card.py` which:
   - Reads the card.
   - Reads the JSON.
   - Regex-replaces every `<AUTO:key>` with `str(json[key])`.
   - Markers without a JSON entry become `<AUTO:key — MISSING>` (visible, not silent).
   - Writes the card back in place.

### Auto-filled fields (script knows these)
From the training run: `status`, `date_started`, `date_finished`, `wall_time`, `final_loss`, `gpu_peak_mem_gb`, `wandb_url`, `trainable_params`, `pair_count`, `boundary_sigma`, `high_noise_rule`, `cache_disk_gb`, `cluster_partition`, `git_sha`.

From the smoke test (sourced from `training_cards/beta-001_smoke_results.json` if present): `smoke_step_median`, `smoke_step_p90`, `smoke_projected_wall_time`, `smoke_latent_roundtrip_mse`. The autofill script reads both `_results.json` and `_smoke_results.json`; the smoke file is optional but lets the user fill the card before launch.

### Human-filled fields (post-run, after looking at wandb + video)
"Loss curve descended?", "Inference sample observations", "Verdict", "Next action".

### Robustness
- Training script wraps the autofill call in `try/except` — a failed autofill never kills an otherwise-good run, just logs a warning.
- If the script crashes mid-training, an `atexit` hook still writes a partial JSON with `status: failed` and partial wall-time. The next manual run of `python -m training.autofill_card training_cards/beta-001.md` will fill what's available.
- The card is valid markdown throughout; markers are visible placeholders, not blockers.

### Standalone usage
```
python -m training.autofill_card training_cards/beta-001.md
```
Reads `training_cards/beta-001_results.json` and updates the card.

---

## 7. Wall-time estimate

Per-step cost on 1× A40, bf16, **T=9** (BETA default), 512×512, A14B high-noise expert + ControlNet:
- ControlNet forward: ~0.2s
- Transformer forward (frozen, 14B params, ~40 blocks): ~2-3s
- Backward (controlnet only, plus through frozen transformer activations): ~0.5-1s
- Optimizer step (8-bit AdamW): negligible
- DataLoader I/O (cached `.pt` files): negligible with workers
- **Per-step total: ~3-4s** (rough; T=9 has ~half the per-step cost of T=17)
- 10000 steps × 3.5s ≈ **10h**, realistic upper bound **~12-14h** with overhead.

Fits the 24h budget with comfortable margin.

**The smoke test produces an actual measured per-step number on this exact hardware** (see §5 step 5). Use that instead of the rough estimate above — read it from `training_cards/beta-001_smoke_results.json`.

Decision rule after smoke:
- median per-step ≤ 4s → launch as-is.
- 4s < median ≤ 6s → launch as-is, but expect to use most of the 24h window.
- 6s < median ≤ 8.5s → consider DDP on 2× A40 if available, else launch and accept tightness.
- median > 8.5s → **do not launch**; either drop `T` to 5, drop resolution, or move to 2× A40 DDP. The smoke test exits 1 in this case.

If the smoke result is comfortably under budget and you want a richer signal, you can also bump `T` to 17 as a stretch goal — but only after the T=9 run completes and shows healthy descent. Don't gamble the first run on T=17.

---

## 8. Implementation order

**Phase 0** (one-time): repo cleanup per §1 — delete `inference/`, `resources/`; move plans to `docs/`; move `input_prompts.py` to `training/`; create `training/`, `slurm/`, `training_cards/` folders; update `.gitignore`. Commit as `chore: BETA repo restructure`.

Phases 1–5 follow:

1. **Phase 1**: `training/precompute_beta.py` — implement and dry-run on 5 pairs locally (no GPU needed if VAE/T5 fit in CPU-RAM-only, otherwise on cluster smoke).
2. **Phase 2**: `training/dataset_beta.py` — depends on Phase 1's manifest format.
3. **Phase 3**: `training/train_beta.py` + `training/autofill_card.py` — depends on Phase 2.
4. **Phase 4**: `training/smoke_test_beta.py` — depends on Phases 1, 2, 3.
5. **Phase 5**: `training_cards/TEMPLATE.md` + `training_cards/beta-001.md` — write template, fill beta-001 *before* launch (everything except `<AUTO:>` markers and qualitative Results).
6. **Phase 6**: `slurm/*.sbatch` — wrap the three commands above in cluster-ready job scripts (see §11).

### User's launch sequence on cluster
1. `sbatch slurm/smoke_test.sbatch` → fix anything that fails.
2. `sbatch slurm/precompute.sbatch` → check `$WORK/wan-beta/cache/` disk usage afterward.
3. `sbatch slurm/train_beta.sbatch` → monitor wandb. Auto-saves checkpoint, runs end-of-run inference, autofills `training_cards/beta-001.md`.
4. After completion, look at the wandb loss curve and the inference video, then fill the qualitative Results entries in `training_cards/beta-001.md`.

---

## 9. Open items to confirm before Phase 1 implementation starts

These are quick yes/no questions but each blocks implementation if wrong:

1. **Local model layout**: confirm path under `$WORK/wan-beta/models/Wan2.2-T2V-A14B-Diffusers` (or wherever you have it) contains the standard diffusers subfolders: `transformer/`, `transformer_2/`, `vae/`, `text_encoder/`, `tokenizer/`, `scheduler/`. If subfolders are named differently, the training script's `subfolder=` args need updating.
2. **Cluster filesystem**: which env var should the sbatch files use — `$WORK`, `$SCRATCH`, `$PROJAPPL`, or something else? And what's the absolute path under it? (I'll templatize the sbatch with this.)
3. **Cluster partition / account**: what `--partition` (and `--account`/`--qos` if required) should the sbatch files specify?
4. **Latent cache disk budget**: ~50GB OK on the cluster filesystem? If not → drop to caching only Canny + embeds, encode latents per-batch (adds ~0.3-0.5s/step = ~1.5h total, still within 24h).
5. **wandb access on cluster**: do compute nodes have outbound HTTPS to wandb.ai? If not, we set `WANDB_MODE=offline` in the sbatch and you `wandb sync` from a login node afterward.
6. **`bitsandbytes` install on cluster**: confirmed installable for the cluster's CUDA version? (CUDA 11.8 / 12.1 / 12.4 each need a different wheel.)
7. **HED config download**: cold init still needs the HED A14B's `config.json` (~10KB) for architecture spec. Either compute nodes have internet for the one-time HF metadata fetch, or pre-download once on a login node to `$WORK/wan-beta/models/wan2.2-t2v-a14b-controlnet-hed-v1/` and pass that local path to `--controlnet_config_repo`.

These do not block plan-writing, but answer before implementation.

---

## 10. What this BETA does NOT do (to prevent scope creep)

- No warped pseudo-targets — targets are static replicated stills.
- No 6-channel landmark heatmaps — uses existing 3-channel input.
- No optical flow / RAFT / disocclusion handling.
- No multi-checkpoint saving, no early-stopping, no eval set, no quantitative blurred-face metric.
- No modification to `wan_t2v_controlnet_pipeline.py` — end-of-run inference uses the existing pipeline as-is, reading the trained ControlNet from disk.
- No multi-GPU / DDP / FSDP — single-GPU only.
- No support for low-noise expert training — only high-noise.

All of the above belong to `IMPLEMENTATION_PLAN.md` (the full pipeline). BETA is the question "does this approach work *at all*?", nothing more.

---

## 11. Cluster setup

### Filesystem layout (outside the git repo)

```
$WORK/wan-beta/                                # base work directory
├── models/                                    # large pretrained weights — never in repo
│   ├── Wan2.2-T2V-A14B-Diffusers/             # ~30+ GB; user already has this
│   └── wan2.2-t2v-a14b-controlnet-hed-v1/     # config.json only is strictly required
├── data/
│   ├── input_faces/                           # 100 PNGs (528×528)
│   └── targets/                               # 10000 JPGs (512×512)
├── cache/                                     # ~50GB, populated by precompute_beta
│   ├── canny/
│   ├── latents/
│   ├── prompts/
│   ├── prompts_negative.pt
│   └── manifest.json
├── checkpoints/                               # ~3-5GB, output of train_beta
│   └── beta-001_final.safetensors
├── outputs/                                   # mp4 inference smokes
│   └── beta-001_final.mp4
├── logs/                                      # slurm stdout/stderr
│   └── slurm-*.out
└── hf_cache/                                  # set HF_HOME here so HF doesn't blow up $HOME quota
```

### One-time setup on the cluster

```bash
# On a login node:
git clone <repo_url> $HOME/wan2.2-controlnet
cd $HOME/wan2.2-controlnet

mkdir -p $WORK/wan-beta/{data/input_faces,data/targets,cache,checkpoints,outputs,logs,hf_cache,models}

# Copy data
rsync -avh /path/to/input_faces/ $WORK/wan-beta/data/input_faces/
rsync -avh /path/to/targets/     $WORK/wan-beta/data/targets/

# Confirm the Wan model is at $WORK/wan-beta/models/Wan2.2-T2V-A14B-Diffusers
# (move from wherever you have it now)

# Pre-fetch the HED config (only if compute nodes have no internet)
huggingface-cli download TheDenk/wan2.2-t2v-a14b-controlnet-hed-v1 config.json \
    --local-dir $WORK/wan-beta/models/wan2.2-t2v-a14b-controlnet-hed-v1

# Python env
python -m venv $WORK/wan-beta/venv
source $WORK/wan-beta/venv/bin/activate
pip install -r requirements.txt
pip install -r requirements_train_beta.txt

# wandb auth (do once on login node; the API key persists in ~/.netrc)
wandb login
```

### sbatch templates

All three jobs share the same env block. Adjust `--partition`, `--account`, etc. for your cluster. Use `srun` to launch the python process so slurm captures resources properly.

#### `slurm/smoke_test.sbatch`
```bash
#!/bin/bash
#SBATCH --job-name=wan-beta-smoke
#SBATCH --partition=<your_gpu_partition>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail
cd $HOME/wan2.2-controlnet
source $WORK/wan-beta/venv/bin/activate

export HF_HOME=$WORK/wan-beta/hf_cache
export WAN_MODEL=$WORK/wan-beta/models/Wan2.2-T2V-A14B-Diffusers
export HED_CONFIG=$WORK/wan-beta/models/wan2.2-t2v-a14b-controlnet-hed-v1
export WANDB_MODE=${WANDB_MODE:-online}

srun python -m training.smoke_test_beta \
    --base_model_path $WAN_MODEL \
    --controlnet_config_repo $HED_CONFIG \
    --work_dir $WORK/wan-beta
```

#### `slurm/precompute.sbatch`
```bash
#!/bin/bash
#SBATCH --job-name=wan-beta-precompute
#SBATCH --partition=<your_gpu_partition>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail
cd $HOME/wan2.2-controlnet
source $WORK/wan-beta/venv/bin/activate

export HF_HOME=$WORK/wan-beta/hf_cache
export WAN_MODEL=$WORK/wan-beta/models/Wan2.2-T2V-A14B-Diffusers

srun python -m training.precompute_beta \
    --input_faces_dir $WORK/wan-beta/data/input_faces \
    --targets_dir    $WORK/wan-beta/data/targets \
    --output_dir     $WORK/wan-beta/cache \
    --base_model_path $WAN_MODEL \
    --height 512 --width 512 --num_frames 9
```

#### `slurm/train_beta.sbatch`
```bash
#!/bin/bash
#SBATCH --job-name=wan-beta-train
#SBATCH --partition=<your_gpu_partition>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail
cd $HOME/wan2.2-controlnet
source $WORK/wan-beta/venv/bin/activate

export HF_HOME=$WORK/wan-beta/hf_cache
export WAN_MODEL=$WORK/wan-beta/models/Wan2.2-T2V-A14B-Diffusers
export HED_CONFIG=$WORK/wan-beta/models/wan2.2-t2v-a14b-controlnet-hed-v1
export WANDB_MODE=${WANDB_MODE:-online}

srun python -m training.train_beta \
    --cache_dir       $WORK/wan-beta/cache \
    --base_model_path $WAN_MODEL \
    --controlnet_config_repo $HED_CONFIG \
    --output_dir      $WORK/wan-beta/checkpoints \
    --inference_output_dir $WORK/wan-beta/outputs \
    --card_path       training_cards/beta-001.md \
    --wandb_project   wan-controlnet-beta \
    --run_name        beta-001 \
    --num_frames 9 --height 512 --width 512 \
    --lr 1e-4 --weight_decay 0.01 --grad_clip 1.0 \
    --num_epochs 1 \
    --checkpoint_every 2000
```

### Notes
- All three sbatch files write `wan-beta-{smoke,precompute,train}-<jobid>.out/.err` to the directory you launched `sbatch` from. Move/symlink to `$WORK/wan-beta/logs/` if you want them centralized.
- `WANDB_MODE=offline` is the fallback if compute nodes can't reach wandb.ai; sync afterward with `wandb sync $HOME/wan2.2-controlnet/wandb/offline-run-*` from a login node.
- Adjust `$WORK` to your cluster's actual work-filesystem env var (`$SCRATCH`, `$PROJAPPL`, etc.) once you confirm Open Item §9.2.


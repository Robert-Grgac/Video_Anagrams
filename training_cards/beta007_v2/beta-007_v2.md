---
run_id: beta-007_v2
---

# Training Card — beta-007_v2

## 1. Goal
**Rerun of beta-007 after fixing a silent T5-conditioning-length bug.** The training recipe is unchanged from beta-007 (see [`training_cards/beta007/beta-007.md`](../beta007/beta-007.md)); only the prompt cache and the in-training negative-prompt encoder now pad to **226** instead of **512**, matching what the Wan 2.2 transformer was pretrained against.

### Why this rerun exists
- beta-007's in-training periodic + final-eval videos looked like pure noise, even though training loss descended (`loss_ema_final = 0.179`) and `controlnet_residual_norm` reached 0.44.
- Standalone inference on `beta-007_final.safetensors` produced **coherent prompt scenes** at `controlnet_weight=0.0` and progressively noisier outputs as weight increased — i.e., the base model worked fine when the pipeline encoded prompts itself, but the CN's learned residuals were inconsistent with what the pipeline emits at inference.
- Differential diagnosis isolated the cause: the pipeline's `_get_t5_prompt_embeds` defaults to `max_sequence_length=226` ([`wan_t2v_controlnet_pipeline.py:233`](../../wan_t2v_controlnet_pipeline.py#L233)), but `precompute_beta.py` was padding cached embeddings to **512**, and `train.py`'s on-the-fly negative-prompt encoding also padded to 512. The 286 extra zero-padding tokens dilute cross-attention to the real text tokens, silently corrupting conditioning — at both training and in-training eval. Training was internally consistent (everything at 512) so loss descended, but the CN's residuals are tied to a conditioning regime the pipeline never produces at inference.
- A one-shot validation submitted `beta-007_final.safetensors` with the cached 512-length positive + a fresh 512-length negative reproduced the noise → bug confirmed end-to-end.

## 2. Hypothesis & success criteria
- **Confirms (primary):**
  - Periodic-eval `eval/mse_avg` *descends* across the 30 rounds AND `eval/ssim_avg` *rises by ≥ 0.05* — i.e., the CN actually fits the training samples now that conditioning is right (beta-007 ΔSSIM was 0.0017, ~30× under the bar).
  - Standalone inference on `beta-007_v2_final.safetensors` with the v2 cache at `controlnet_weight=1.0` produces a recognizable scene **with face structure visible** (beta-007 at the same setup showed scene but no face).
  - `loss_ema_final ≤ 0.20` (matching beta-007's value within noise; the recipe itself was reasonable).
- **Rejects:**
  - Periodic-eval MSE is still flat / SSIM still flat → there is a second bug beyond conditioning length, OR 309 effective steps is genuinely too few for this recipe (cold-start dead zone is 1k–3k effective updates).
  - Standalone inference still shows no face structure at any weight → the recipe doesn't bootstrap a useful CN from cold start in 309 steps regardless of conditioning; warm-start from beta-001 (also re-trained against the v2 cache) becomes the right next step.

## 3. Setup
Identical to [beta-007](../beta007/beta-007.md) except:

### Data
- Prompt cache regenerated at `--max_seq_len 226` (was 512). Canny + latents caches reused unchanged — they are length-independent.
- Cache invalidation: delete `$WAN_BETA_CACHE/prompts/` and `$WAN_BETA_CACHE/prompts_negative.pt` before re-running precompute; the precompute script skips existing files for canny + latents.

### Code changes vs beta-007
- [`training/precompute_beta.py`](../../training/precompute_beta.py) — `max_seq_len` is now a CLI flag, default 226.
- [`training/beta007/train.py`](../../training/beta007/train.py) — module-level constant `WAN_T5_MAX_SEQ_LEN = 226`; neg-prompt encoder uses it; a runtime assertion at eval-stage cache load fails fast if any cached embed has a different seq length.

### Run metadata (auto)
- Status: `completed`
- Started: `2026-05-11T23:55:58+00:00`
- Finished: `2026-05-12T07:42:54+00:00`
- Wall time: `06:43:12`
- Git SHA: `8f4a4a1`

## 4. Results
*(machine fields auto-filled; qualitative fields human-filled after looking at wandb + inference videos)*
- Final loss (last effective step, raw): `0.146986`
- Final loss EMA (window 20): `<AUTO:final_loss_ema — MISSING>`
- High-phase avg loss (over all eff steps): `0.162409`
- Final-eval mean MSE (100 samples, EMA controlnet): `0.116258`
- Final-eval mean SSIM (100 samples, EMA controlnet): `0.143578`
- Final-eval wall time: `3374.3`s
- GPU peak memory: `70.4` GB
- cn_end_fraction (dynamic): `0.14`
- Loss curve descended? (Y/N, shape):
- Periodic-eval MSE trajectory shape (descending / U-shaped / flat / rising):
- Periodic-eval SSIM trajectory shape (rising / flat / falling):
- ΔSSIM step 10 → step 300 (success bar: ≥ 0.05):
- Periodic vs final-eval gap:
- Standalone inference at `controlnet_weight=1.0` (with v2 cache): scene? face structure?
- Standalone inference at `controlnet_weight=2.5` / `5.0`: face visible? compare to beta-001's threshold.
- Verdict (bug fix sufficient / still undertrained / second bug present):
- Next action:

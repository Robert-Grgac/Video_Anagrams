"""Standalone inference for a trained beta ControlNet.

Loads a saved ControlNet checkpoint, builds the full Wan 2.2 pipeline with
both experts (high-noise + low-noise), runs once on a (canny, prompt) pair
from the precomputed cache, and writes one mp4.

Uses pipe.enable_model_cpu_offload() so the two-expert pipeline fits on a
single 44GB A40 — both experts at bf16 ≈ 56GB if both stayed on GPU.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from training.utils import cast_respecting_fp32_modules, detect_boundary_ratio
from training.input_prompts import PROMPTS_BATCH_1, PROMPTS_BATCH_2

if TYPE_CHECKING:
    from wan_t2v_controlnet_pipeline import WanTextToVideoControlnetPipeline

ALL_PROMPTS = {**PROMPTS_BATCH_1, **PROMPTS_BATCH_2}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint_path", type=str, required=True,
                   help="Trained ControlNet .safetensors (e.g. .../beta-001_final.safetensors).")
    p.add_argument("--base_model_path", type=str, required=True,
                   help="Wan-AI Wan2.2-T2V-A14B-Diffusers snapshot dir.")
    p.add_argument("--controlnet_config_repo", type=str, required=True,
                   help="HED config snapshot dir (architecture only).")
    p.add_argument("--cache_dir", type=str, required=True,
                   help="Wan-beta precompute cache (for control input + slugs).")
    p.add_argument("--control_subdir", type=str, default="canny",
                   help="Subdir of cache_dir holding the (3, H, W) uint8 .pt "
                        "control inputs. Default 'canny' for the original "
                        "edge cache; set 'silhouette' for the option-H map.")
    p.add_argument("--output_path", type=str, required=True,
                   help="Output mp4 path.")
    p.add_argument("--face_idx", type=int, default=0,
                   help="Which face's Canny to condition on. Ignored when --face_idxs is set.")
    p.add_argument("--face_idxs", type=str, default=None,
                   help="Comma- or space-separated list of face indices, e.g. '0,25,50,75,99'. "
                        "When set, the pipeline is built ONCE and one mp4 per face is written; "
                        "output filenames get a '_face{idx}' suffix. Overrides --face_idx.")
    p.add_argument("--slug", type=str, default=None,
                   help="Slug name from PROMPTS_BATCH_*. If unset, picks the first slug "
                        "available for face_idx in the cache manifest.")
    p.add_argument("--negative_prompt", type=str,
                   default="bad quality, worst quality")
    p.add_argument("--num_frames", type=int, default=9)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--num_inference_steps", type=int, default=100)
    p.add_argument("--guidance_scale", type=float, default=5.0)
    p.add_argument("--controlnet_weight", type=float, default=1.0)
    p.add_argument("--weights", type=str, default=None,
                   help="Optional comma-separated list of controlnet_weight values. "
                        "If set, builds the pipeline once and writes one mp4 per weight; "
                        "output filenames get a '_w{weight}' suffix. Overrides --controlnet_weight.")
    p.add_argument("--controlnet_stride", type=int, default=3)
    # ControlNet was trained only against the high-noise expert (sigma >= 0.875).
    # Limit injection to that regime: with FlowMatch's roughly-linear sigma
    # schedule, current_sampling_percent < (1 - boundary_ratio) = 0.125 covers
    # exactly the steps where the high-noise expert is active.
    p.add_argument("--controlnet_guidance_start", type=float, default=0.0)
    p.add_argument("--controlnet_guidance_end", type=float, default=0.125)
    p.add_argument("--ends", type=str, default=None,
                   help="Optional comma-separated list of controlnet_guidance_end values. "
                        "Combined with --weights as a Cartesian product. "
                        "When set, output filenames get a '_w{weight}_e{end}' suffix. "
                        "Overrides --controlnet_guidance_end.")
    p.add_argument("--dynamic_cn_end", action="store_true",
                   help="Compute controlnet_guidance_end from the scheduler's sigma "
                        "trajectory (the step fraction at which sigma first drops below "
                        "boundary_ratio). Mirrors train_beta7._compute_cn_end_high_noise. "
                        "Overrides both --controlnet_guidance_end and --ends.")
    p.add_argument("--prompt_cache_path", type=str, default=None,
                   help="Path to a precomputed positive prompt embedding .pt file "
                        "(e.g. $HOME/cache/wan-beta/prompts/misty_morning.pt). When set, "
                        "the pipeline is called with prompt_embeds= instead of raw text, "
                        "reproducing the training-time eval conditioning. The negative "
                        "prompt is then encoded fresh at --prompt_embed_max_len (or the "
                        "loaded positive's seq length if unset) so both sides match.")
    p.add_argument("--prompt_embed_max_len", type=int, default=None,
                   help="Pad length for re-encoding the negative prompt when "
                        "--prompt_cache_path is set. Default: derive from the loaded "
                        "positive tensor's sequence dim. Match the precompute setting "
                        "(beta cache uses 512; pipeline default is 226).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fps", type=int, default=8)
    return p.parse_args()


def resolve_slug(cache_dir: Path, face_idx: int, slug_arg: str | None) -> str:
    if slug_arg is not None:
        if slug_arg not in ALL_PROMPTS:
            raise ValueError(f"Unknown slug '{slug_arg}'.")
        return slug_arg
    manifest = json.loads((cache_dir / "manifest.json").read_text())
    for rec in manifest:
        if rec["face_idx"] == face_idx:
            return rec["slug"]
    raise RuntimeError(f"No manifest entry found for face_idx={face_idx}.")


def load_canny_image(cache_dir: Path, face_idx: int,
                     control_subdir: str = "canny") -> Image.Image:
    canny_path = cache_dir / control_subdir / f"face_{face_idx}.pt"
    if not canny_path.exists():
        raise FileNotFoundError(canny_path)
    canny_u8 = torch.load(canny_path, map_location="cpu", weights_only=True)  # (3, H, W) uint8
    return Image.fromarray(canny_u8.permute(1, 2, 0).numpy())


def compute_dynamic_cn_end(base_model_path: str, num_inference_steps: int,
                           boundary_ratio: float) -> tuple[float, int]:
    """Step-fraction at which sigma first drops below boundary_ratio.

    Mirrors train_beta7._compute_cn_end_high_noise so inference uses the same
    high-noise-only CN gate the run was trained for.
    """
    from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
    sched = FlowMatchEulerDiscreteScheduler.from_pretrained(
        base_model_path, subfolder="scheduler",
    )
    sched.set_timesteps(num_inference_steps)
    sigmas = sched.sigmas[:-1].detach().cpu()
    below = (sigmas < boundary_ratio).nonzero(as_tuple=False)
    if below.numel() == 0:
        return 1.0, num_inference_steps
    first_low = int(below[0].item())
    return first_low / num_inference_steps, first_low


def save_video(frames_np: np.ndarray, path: Path, fps: int = 8) -> None:
    import imageio
    arr = (np.clip(frames_np, 0.0, 1.0) * 255).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(path), list(arr), fps=fps, codec="libx264")


def build_pipeline(args: argparse.Namespace) -> "WanTextToVideoControlnetPipeline":
    from diffusers import AutoencoderKLWan
    from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
    from transformers import AutoTokenizer, UMT5EncoderModel
    from safetensors.torch import load_file

    from wan_transformer import CustomWanTransformer3DModel
    from wan_controlnet import WanControlnet
    from wan_t2v_controlnet_pipeline import WanTextToVideoControlnetPipeline

    base = args.base_model_path

    print(f"[load] tokenizer + text_encoder ...")
    tokenizer = AutoTokenizer.from_pretrained(base, subfolder="tokenizer")
    text_encoder = UMT5EncoderModel.from_pretrained(
        base, subfolder="text_encoder", torch_dtype=torch.bfloat16,
    ).eval()

    print(f"[load] vae ...")
    vae = AutoencoderKLWan.from_pretrained(
        base, subfolder="vae", torch_dtype=torch.bfloat16,
    ).eval()

    print(f"[load] high-noise transformer ...")
    transformer = CustomWanTransformer3DModel.from_pretrained(
        base, subfolder="transformer", torch_dtype=torch.bfloat16,
    ).eval()

    # Load transformer_2 (low-noise expert) as the SAME custom subclass so it
    # accepts the controlnet_states kwarg the pipeline always passes. The class
    # only overrides forward(); state_dict keys match WanTransformer3DModel
    # exactly, so the checkpoint loads cleanly. With controlnet_states=None
    # (or unused stride misalignment), the residuals contribute nothing here.
    print(f"[load] low-noise transformer_2 ...")
    transformer_2 = CustomWanTransformer3DModel.from_pretrained(
        base, subfolder="transformer_2", torch_dtype=torch.bfloat16,
    ).eval()

    print(f"[load] scheduler ...")
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        base, subfolder="scheduler",
    )

    print(f"[load] controlnet config from {args.controlnet_config_repo} ...")
    config = WanControlnet.load_config(args.controlnet_config_repo)
    controlnet = WanControlnet.from_config(config)
    cast_respecting_fp32_modules(controlnet, torch.bfloat16)
    print(f"[load] controlnet weights from {args.checkpoint_path} ...")
    sd = load_file(args.checkpoint_path)
    missing, unexpected = controlnet.load_state_dict(sd, strict=False)
    if missing:
        print(f"[warn] missing keys when loading controlnet: {len(missing)}")
    if unexpected:
        print(f"[warn] unexpected keys when loading controlnet: {len(unexpected)}")
    controlnet.eval()

    boundary_ratio, src = detect_boundary_ratio(base, dict(transformer.config))
    print(f"[detect] boundary_ratio={boundary_ratio} ({src})")

    pipe = WanTextToVideoControlnetPipeline(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        transformer=transformer,
        transformer_2=transformer_2,
        vae=vae,
        controlnet=controlnet,
        scheduler=scheduler,
        boundary_ratio=boundary_ratio,
    )
    pipe.enable_model_cpu_offload()

    # Pin the ControlNet to GPU. The pipeline's prepare_controlnet_frames reads
    # self.controlnet.device at prep time; under model_cpu_offload that returns
    # "cpu", and accelerate's pre-forward hook does not migrate the resulting
    # input tensor → CPU/GPU device mismatch inside the first Conv3D. The
    # ControlNet is small (~1 GB at bf16) and runs every step, so pinning has
    # no memory cost and avoids the offload round-trip on each call.
    from accelerate.hooks import remove_hook_from_module
    remove_hook_from_module(pipe.controlnet, recurse=True)
    pipe.controlnet.to("cuda")
    return pipe


def parse_face_idxs(args: argparse.Namespace) -> list[int]:
    if args.face_idxs:
        # Accept comma- or whitespace-separated.
        raw = args.face_idxs.replace(",", " ").split()
        return [int(x) for x in raw if x.strip()]
    return [args.face_idx]


def main() -> int:
    args = parse_args()
    cache_dir = Path(args.cache_dir)

    face_idxs = parse_face_idxs(args)
    print(f"[input] face_idxs={face_idxs}")

    pipe = build_pipeline(args)

    if args.weights:
        weights = [float(w) for w in args.weights.split(",") if w.strip()]
    else:
        weights = [args.controlnet_weight]
    if args.dynamic_cn_end:
        cn_end, first_low = compute_dynamic_cn_end(
            args.base_model_path, args.num_inference_steps, pipe.boundary_ratio,
        )
        print(f"[dynamic_cn_end] boundary_ratio={pipe.boundary_ratio} "
              f"num_inference_steps={args.num_inference_steps} "
              f"first_low_idx={first_low} cn_end={cn_end:.4f}")
        ends = [cn_end]
    elif args.ends:
        ends = [float(e) for e in args.ends.split(",") if e.strip()]
    else:
        ends = [args.controlnet_guidance_end]
    multi_face = len(face_idxs) > 1
    sweep_active = (bool(args.weights) or bool(args.ends)
                    or args.dynamic_cn_end or multi_face)
    print(f"[sweep] {len(face_idxs)} face(s) x {len(weights)} weight(s) x "
          f"{len(ends)} end(s) = {len(face_idxs) * len(weights) * len(ends)} videos")

    # Prompt-cache mode uses one fixed positive embedding for every face. Warn
    # if that's mixed with multi-face, because per-face slug auto-resolution
    # would otherwise imply per-face prompts.
    pos_emb = None
    neg_emb = None
    if args.prompt_cache_path:
        pos_emb = torch.load(args.prompt_cache_path, map_location="cpu",
                             weights_only=True).to(torch.bfloat16)
        if pos_emb.dim() == 2:
            pos_emb = pos_emb.unsqueeze(0)
        max_len = args.prompt_embed_max_len or pos_emb.shape[1]
        print(f"[prompt-cache] loaded {args.prompt_cache_path} "
              f"shape={tuple(pos_emb.shape)}")
        print(f"[prompt-cache] encoding negative prompt "
              f"('{args.negative_prompt}') at max_sequence_length={max_len}")
        pos_emb = pos_emb.to(pipe._execution_device)
        neg_emb = pipe._get_t5_prompt_embeds(
            prompt=args.negative_prompt,
            max_sequence_length=max_len,
            device=pipe._execution_device,
            dtype=torch.bfloat16,
        )
        print(f"[prompt-cache] negative embed shape={tuple(neg_emb.shape)}")
        if multi_face:
            print(f"[warn] --prompt_cache_path is set AND multiple faces "
                  f"requested; the same prompt embedding will be used for all "
                  f"{len(face_idxs)} faces.")

    out_path = Path(args.output_path)
    from accelerate.hooks import remove_hook_from_module
    for face_idx in face_idxs:
        slug = resolve_slug(cache_dir, face_idx, args.slug)
        canny_img = load_canny_image(cache_dir, face_idx, args.control_subdir)

        if pos_emb is not None:
            prompt_kwargs = dict(prompt_embeds=pos_emb, negative_prompt_embeds=neg_emb)
            print(f"[face] face_idx={face_idx} slug='{slug}' (overridden by --prompt_cache_path)")
        else:
            prompt_text = ALL_PROMPTS[slug]
            prompt_kwargs = dict(prompt=prompt_text, negative_prompt=args.negative_prompt)
            print(f"[face] face_idx={face_idx} slug='{slug}' prompt={prompt_text!r}")

        for w in weights:
            for e in ends:
                # Diffusers' model_cpu_offload re-attaches an accelerate hook to the
                # controlnet at the start of each __call__; its pre_forward then
                # routes inputs to CPU and we get a CPU/GPU mismatch on the first
                # Conv3D. Strip the hook and pin the controlnet to GPU before every
                # iteration of the sweep.
                remove_hook_from_module(pipe.controlnet, recurse=True)
                pipe.controlnet.to("cuda")
                # Reseed so the only varying factors across videos are (face, weight, end).
                generator = torch.Generator().manual_seed(args.seed)
                out = pipe(
                    controlnet_frames=[canny_img] * args.num_frames,
                    **prompt_kwargs,
                    height=args.height,
                    width=args.width,
                    num_frames=args.num_frames,
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=args.guidance_scale,
                    controlnet_weight=w,
                    controlnet_stride=args.controlnet_stride,
                    controlnet_guidance_start=args.controlnet_guidance_start,
                    controlnet_guidance_end=e,
                    generator=generator,
                    output_type="np",
                )
                frames = out.frames[0]  # (T, H, W, 3) float in [0, 1]
                if not sweep_active:
                    target = out_path
                else:
                    parts = []
                    if multi_face:
                        parts.append(f"face{face_idx}")
                    if bool(args.weights) or len(weights) > 1:
                        parts.append(f"w{f'{w:.2f}'.replace('.', 'p')}")
                    if bool(args.ends) or args.dynamic_cn_end or len(ends) > 1:
                        estr = f"{e:.3f}".rstrip("0").rstrip(".").replace(".", "p")
                        parts.append(f"e{estr}")
                    suffix = "_" + "_".join(parts) if parts else ""
                    target = out_path.with_name(
                        f"{out_path.stem}{suffix}{out_path.suffix}"
                    )
                save_video(frames, target, fps=args.fps)
                print(f"[done] wrote {target}  (face={face_idx}, w={w}, end={e})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

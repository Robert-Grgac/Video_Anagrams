"""inference.py — generate one video from a face image + a scene prompt.

Usage:
    python inference.py <image_path> "<prompt>"

Reads the Wan 2.2 A14B model from ``models/wan2.2/``, the HED architecture
config from ``models/hed_config/``, and the trained ControlNet from
``models/controlnet/controlnet.safetensors``. Computes the deterministic
linear-FlowMatch inverse trajectory for the input face inline (no Euler ODE,
no precomputed inverts), encodes the prompt on the fly, runs the
``WanPTDCNPipeline``, and writes the result to ``./output.mp4``.

All published-recipe hyperparameters (100 inference steps, CFG=5.0,
controlnet_weight=1.0, the PI-heuristic schedule) are baked in as defaults
but can be overridden with optional flags. The image is resized to 528x528
(the inference resolution) before being fed to both the ControlNet and the
VAE inversion.
"""
from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from training.config import TrainConfig

DEFAULT_WAN_MODEL = REPO_ROOT / "models" / "wan2.2"
DEFAULT_HED_CONFIG = REPO_ROOT / "models" / "hed_config"
DEFAULT_CHECKPOINT = REPO_ROOT / "models" / "controlnet" / "controlnet.safetensors"
DEFAULT_OUTPUT = REPO_ROOT / "output.mp4"

# Inference resolution / frame count match the published pipeline. The PTD
# pipeline's heuristic-3 PI controller requires NUM_INFERENCE_STEPS in (100, 101).
HEIGHT = 528
WIDTH = 528
NUM_FRAMES = 61
NUM_INFERENCE_STEPS = 100


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the PTD + ControlNet pipeline on one face/prompt pair.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("image_path", type=str, help="Path to a face image (PNG/JPG).")
    p.add_argument("prompt", type=str, help="Scene prompt.")
    p.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT),
                   help="Output mp4 path.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--guidance_scale", type=float, default=5.0,
                   help="Classifier-free guidance scale.")
    p.add_argument("--controlnet_weight", type=float, default=1.0)
    p.add_argument("--controlnet_stride", type=int, default=3)
    p.add_argument("--direct_transfer_steps", type=int, default=45)
    p.add_argument("--decayed_transfer_steps", type=int, default=22)
    p.add_argument("--initial_blending_coeff", type=float, default=0.4)
    p.add_argument("--Kp", type=float, default=0.5,
                   help="Heuristic-3 PI proportional gain.")
    p.add_argument("--Ki", type=float, default=0.2,
                   help="Heuristic-3 PI integral gain.")
    p.add_argument("--max_blending_coeff_delta", type=float, default=0.05,
                   help="Per-step slew rate cap on the blending coefficient.")
    p.add_argument("--negative_prompt", type=str,
                   default=TrainConfig().negative_prompt)
    p.add_argument("--fps", type=int, default=8)
    return p.parse_args()


def check_assets(image_path: Path) -> None:
    if not (DEFAULT_WAN_MODEL / "model_index.json").exists():
        sys.exit(
            f"ERROR: Wan 2.2 A14B model not found at {DEFAULT_WAN_MODEL} "
            f"(missing model_index.json).\nDownload it with:\n"
            f"  huggingface-cli download Wan-AI/Wan2.2-T2V-A14B-Diffusers "
            f"--local-dir {DEFAULT_WAN_MODEL}"
        )
    if not (DEFAULT_HED_CONFIG / "config.json").exists():
        sys.exit(
            f"ERROR: HED config.json not found at "
            f"{DEFAULT_HED_CONFIG}/config.json (vendored in the repo)."
        )
    if not DEFAULT_CHECKPOINT.exists():
        sys.exit(
            f"ERROR: ControlNet checkpoint not found at {DEFAULT_CHECKPOINT}.\n"
            f"Train one with: python bootstrap.py"
        )
    if not image_path.exists():
        sys.exit(f"ERROR: image not found at {image_path}.")


def build_pipeline():
    from diffusers import AutoencoderKLWan
    # Wan 2.2 A14B's model_index/scheduler_config specifies
    # UniPCMultistepScheduler (flow_prediction + use_flow_sigmas). Using the
    # FlowMatchEulerDiscreteScheduler instead produces a different trajectory
    # and downcasts step() output to bf16, which silently flattens the
    # phase-substitute cosine curve.
    from diffusers.schedulers import UniPCMultistepScheduler
    from transformers import AutoTokenizer, UMT5EncoderModel
    from safetensors.torch import load_file

    from pipeline.wan_transformer import CustomWanTransformer3DModel
    from pipeline.wan_controlnet import WanControlnet
    from pipeline.PTD_CN_pipeline import WanPTDCNPipeline
    from utils.utils import cast_respecting_fp32_modules, detect_boundary_ratio

    base = str(DEFAULT_WAN_MODEL)

    print("[load] tokenizer + text_encoder ...")
    tokenizer = AutoTokenizer.from_pretrained(base, subfolder="tokenizer")
    text_encoder = UMT5EncoderModel.from_pretrained(
        base, subfolder="text_encoder", torch_dtype=torch.bfloat16,
    ).eval()

    print("[load] vae ...")
    vae = AutoencoderKLWan.from_pretrained(
        base, subfolder="vae", torch_dtype=torch.bfloat16,
    ).eval()

    print("[load] high-noise transformer ...")
    transformer = CustomWanTransformer3DModel.from_pretrained(
        base, subfolder="transformer", torch_dtype=torch.bfloat16,
    ).eval()

    print("[load] low-noise transformer_2 ...")
    transformer_2 = CustomWanTransformer3DModel.from_pretrained(
        base, subfolder="transformer_2", torch_dtype=torch.bfloat16,
    ).eval()

    print("[load] scheduler (UniPCMultistepScheduler) ...")
    scheduler = UniPCMultistepScheduler.from_pretrained(base, subfolder="scheduler")

    print(f"[load] controlnet config from {DEFAULT_HED_CONFIG} ...")
    config = WanControlnet.load_config(str(DEFAULT_HED_CONFIG))
    controlnet = WanControlnet.from_config(config)
    cast_respecting_fp32_modules(controlnet, torch.bfloat16)
    print(f"[load] controlnet weights from {DEFAULT_CHECKPOINT} ...")
    sd = load_file(str(DEFAULT_CHECKPOINT))
    missing, unexpected = controlnet.load_state_dict(sd, strict=False)
    if missing:
        print(f"[warn] missing keys when loading controlnet: {len(missing)}")
    if unexpected:
        print(f"[warn] unexpected keys when loading controlnet: {len(unexpected)}")
    controlnet.eval()

    boundary_ratio, src = detect_boundary_ratio(base, dict(transformer.config))
    print(f"[detect] boundary_ratio={boundary_ratio} ({src})")

    pipe = WanPTDCNPipeline(
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

    # Pin the CN to GPU — it runs every high-noise step, so accelerate's
    # offload churn would dominate latency. Strip its hook now; we re-strip
    # right before __call__ because accelerate re-attaches per call.
    from accelerate.hooks import remove_hook_from_module
    remove_hook_from_module(pipe.controlnet, recurse=True)
    pipe.controlnet.to("cuda")
    return pipe


@torch.no_grad()
def deterministic_ref_latents(pipe, face_img: Image.Image,
                              seed: int) -> torch.Tensor:
    """Build the per-step reference-latent trajectory inline.

    VAE-encodes the (already-resized) face as a static "video", normalizes
    with Wan's encode-side rule, then constructs
    ``z_t = sigma * z_face + (1 - sigma) * eps`` at each denoising-order
    sigma. No transformer forwards — this is the linear-FlowMatch formula,
    bit-identical to ``PTD_Pipeline.WanInversionPipeline.deterministic_invert``.
    Output is (NUM_INFERENCE_STEPS, 1, 16, T_lat, H_lat, W_lat).
    """
    device = pipe._execution_device

    img = pipe.video_processor.preprocess(face_img, height=HEIGHT, width=WIDTH)
    img = img.to(device=device, dtype=torch.float32)
    video = img.unsqueeze(2).repeat(1, 1, NUM_FRAMES, 1, 1)
    video = video.to(device=device, dtype=pipe.vae.dtype)
    z_face = pipe.vae.encode(video).latent_dist.mode()

    z_dim = pipe.vae.config.z_dim
    latents_mean = torch.tensor(
        pipe.vae.config.latents_mean, device=device, dtype=z_face.dtype,
    ).view(1, z_dim, 1, 1, 1)
    latents_std = torch.tensor(
        pipe.vae.config.latents_std, device=device, dtype=z_face.dtype,
    ).view(1, z_dim, 1, 1, 1)
    z_face = ((z_face - latents_mean) / latents_std).to(torch.float32)

    gen = torch.Generator(device=device).manual_seed(seed)
    eps = torch.randn(z_face.shape, generator=gen, device=device,
                      dtype=torch.float32)

    pipe.scheduler.set_timesteps(NUM_INFERENCE_STEPS, device=device)
    inv_sigmas = torch.flip(pipe.scheduler.sigmas, dims=[0]).to(
        device=device, dtype=torch.float32,
    )

    refs = [inv_sigmas[i] * z_face + (1.0 - inv_sigmas[i]) * eps
            for i in range(NUM_INFERENCE_STEPS)]
    ref_latents = torch.stack(refs, dim=0)
    print(f"[invert] deterministic ref_latents shape={tuple(ref_latents.shape)}")
    return ref_latents


def save_video(frames_np: np.ndarray, path: Path, fps: int) -> None:
    import imageio
    arr = (np.clip(frames_np, 0.0, 1.0) * 255).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(path), list(arr), fps=fps, codec="libx264")


def main() -> int:
    args = parse_args()

    image_path = Path(args.image_path).resolve()
    output_path = Path(args.output).resolve()
    check_assets(image_path)

    print(f"[input] image={image_path}")
    print(f"[input] prompt={args.prompt!r}")
    face_img = Image.open(image_path).convert("RGB").resize(
        (WIDTH, HEIGHT), Image.LANCZOS,
    )

    pipe = build_pipeline()
    ref_latents = deterministic_ref_latents(pipe, face_img, seed=args.seed)

    print(f"[infer] running PTD + ControlNet pipeline "
          f"({NUM_INFERENCE_STEPS} steps, cfg={args.guidance_scale}, "
          f"cn_weight={args.controlnet_weight}) ...")
    from accelerate.hooks import remove_hook_from_module
    remove_hook_from_module(pipe.controlnet, recurse=True)
    pipe.controlnet.to("cuda")

    generator = torch.Generator().manual_seed(args.seed)
    cn_frames = [face_img] * NUM_FRAMES if args.controlnet_weight > 0 else None
    out = pipe(
        controlnet_frames=cn_frames,
        ref_latents=ref_latents,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        height=HEIGHT, width=WIDTH,
        num_frames=NUM_FRAMES,
        num_inference_steps=NUM_INFERENCE_STEPS,
        guidance_scale=args.guidance_scale,
        controlnet_weight=args.controlnet_weight,
        controlnet_stride=args.controlnet_stride,
        direct_transfer_steps=args.direct_transfer_steps,
        decayed_transfer_steps=args.decayed_transfer_steps,
        initial_blending_coeff=args.initial_blending_coeff,
        Kp=args.Kp,
        Ki=args.Ki,
        max_blending_coeff_delta=args.max_blending_coeff_delta,
        generator=generator,
        output_type="np",
    )
    frames = out.frames[0]
    save_video(frames, output_path, fps=args.fps)
    print(f"[done] wrote {output_path}")

    del out, frames, ref_latents
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    sys.exit(main())

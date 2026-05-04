"""BETA precompute: extract Canny edges, VAE latents, and T5 embeddings.

Writes to ``<output_dir>/{canny,latents,prompts}/`` plus ``manifest.json``.
The training loop then reads only these small ``.pt`` files.

Decode-side denormalization in the inference pipeline
(``wan_t2v_controlnet_pipeline.py``) is::

    latents_std = 1.0 / vae.config.latents_std   # reassigned reciprocal
    latents     = z / latents_std + latents_mean # = z * vae.config.latents_std + mean

So the encode-side inverse used here is::

    z_norm = (vae.encode(x).latent_dist.mean - vae.config.latents_mean) / vae.config.latents_std

A round-trip self-test (load image -> encode -> normalize -> denormalize ->
decode -> compare to input) gates the whole script: wrong constants/sign
yields MSE >> 0.01 and the script aborts before producing corrupted data.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from training.input_prompts import PROMPTS_BATCH_1, PROMPTS_BATCH_2

ALL_PROMPTS: dict[str, str] = {**PROMPTS_BATCH_1, **PROMPTS_BATCH_2}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input_faces_dir", type=str, required=True)
    p.add_argument("--targets_dir", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--base_model_path", type=str, required=True)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--num_frames", type=int, default=9)
    p.add_argument("--canny_low", type=int, default=100)
    p.add_argument("--canny_high", type=int, default=200)
    p.add_argument("--negative_prompt", type=str, default="bad quality, worst quality")
    p.add_argument("--limit", type=int, default=None,
                   help="Only process the first N validated pairs (for smoke).")
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def discover_pairs(faces_dir: Path, targets_dir: Path) -> list[tuple[int, str, Path, Path]]:
    """Return list of (face_idx, slug, face_png_path, target_jpg_path)."""
    pairs: list[tuple[int, str, Path, Path]] = []
    skipped = 0
    for tgt in sorted(targets_dir.glob("face_*_*.jpg")):
        stem = tgt.stem  # face_{idx}_{slug}
        if not stem.startswith("face_"):
            skipped += 1
            continue
        body = stem[len("face_"):]
        if "_" not in body:
            skipped += 1
            continue
        idx_str, slug = body.split("_", 1)
        try:
            face_idx = int(idx_str)
        except ValueError:
            skipped += 1
            continue
        face_png = faces_dir / f"face_{face_idx}.png"
        if not face_png.exists():
            print(f"[skip] missing face PNG for {tgt.name}: {face_png}")
            skipped += 1
            continue
        if slug not in ALL_PROMPTS:
            print(f"[skip] slug '{slug}' from {tgt.name} not in PROMPTS_BATCH_*")
            skipped += 1
            continue
        pairs.append((face_idx, slug, face_png, tgt))
    print(f"[discover] {len(pairs)} valid pairs, {skipped} skipped")
    return pairs


def load_target_to_minus1_plus1(path: Path, height: int, width: int) -> torch.Tensor:
    """Load JPG target -> (3, H, W) float32 in [-1, 1]."""
    img = Image.open(path).convert("RGB")
    if img.size != (width, height):
        img = img.resize((width, height), Image.LANCZOS)
    arr = np.asarray(img, dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def make_canny_uint8(face_png: Path, height: int, width: int,
                     low: int, high: int) -> torch.Tensor:
    """Load face PNG -> 3-channel-replicated Canny edges (3, H, W) uint8."""
    img = Image.open(face_png).convert("RGB")
    if img.size != (width, height):
        img = img.resize((width, height), Image.LANCZOS)
    gray = np.asarray(img.convert("L"), dtype=np.uint8)
    edges = cv2.Canny(gray, low, high)
    canny_3 = np.stack([edges, edges, edges], axis=0)
    return torch.from_numpy(canny_3)


def encode_latent(vae, x_chw: torch.Tensor, num_frames: int,
                  device: torch.device,
                  latents_mean: torch.Tensor,
                  latents_std: torch.Tensor) -> torch.Tensor:
    """Encode a single still as a (C, T_lat, H_lat, W_lat) fp16 latent.

    The still is replicated to ``num_frames`` along the temporal axis before
    encoding, then mean-sampled and normalized using Wan's encode-side rule.
    """
    x = x_chw.unsqueeze(1).expand(-1, num_frames, -1, -1)        # (3, T, H, W)
    x = x.unsqueeze(0).to(device=device, dtype=vae.dtype)        # (1, 3, T, H, W)
    z = vae.encode(x).latent_dist.mean                           # (1, C, T_lat, H_lat, W_lat)
    z_norm = (z - latents_mean) / latents_std
    return z_norm.squeeze(0).to(torch.float16).cpu()


def latent_norm_round_trip_test(vae, sample_target: Path, height: int, width: int,
                                num_frames: int, device: torch.device,
                                latents_mean: torch.Tensor,
                                latents_std: torch.Tensor) -> float:
    """Hard gate: encode -> normalize -> denormalize -> decode -> MSE.

    Returns the measured MSE so the caller can log it. Aborts on failure.
    Threshold ``1e-2`` accommodates lossy VAE reconstruction; wrong-sign /
    wrong-constant errors typically yield MSE > 0.5.
    """
    x_chw = load_target_to_minus1_plus1(sample_target, height, width)
    x = x_chw.unsqueeze(1).expand(-1, num_frames, -1, -1)
    x = x.unsqueeze(0).to(device=device, dtype=vae.dtype)

    z = vae.encode(x).latent_dist.mean
    z_norm = (z - latents_mean) / latents_std
    z_unnorm = z_norm * latents_std + latents_mean
    x_rec = vae.decode(z_unnorm, return_dict=False)[0]

    mse = F.mse_loss(x_rec.float(), x.float()).item()
    print(f"[round-trip] VAE encode->decode MSE = {mse:.6f}")
    if not (mse < 1e-2):
        raise RuntimeError(
            f"VAE round-trip MSE {mse:.4f} exceeds 1e-2; latent normalization "
            "constants are likely wrong. Refusing to write corrupt cache."
        )
    return mse


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    (out_dir / "canny").mkdir(parents=True, exist_ok=True)
    (out_dir / "latents").mkdir(parents=True, exist_ok=True)
    (out_dir / "prompts").mkdir(parents=True, exist_ok=True)

    faces_dir = Path(args.input_faces_dir)
    targets_dir = Path(args.targets_dir)
    pairs = discover_pairs(faces_dir, targets_dir)
    if args.limit is not None:
        pairs = pairs[:args.limit]
        print(f"[limit] truncated to {len(pairs)} pairs")
    if not pairs:
        raise RuntimeError("No valid (face, target) pairs found.")

    device = torch.device(args.device)
    base = args.base_model_path

    # --- Load VAE ---
    print("[load] AutoencoderKLWan ...")
    from diffusers import AutoencoderKLWan
    vae_dtype = torch.float32 if device.type == "cpu" else torch.bfloat16
    vae = AutoencoderKLWan.from_pretrained(base, subfolder="vae",
                                           torch_dtype=vae_dtype)
    vae.eval().requires_grad_(False).to(device)

    z_dim = vae.config.z_dim
    latents_mean = torch.tensor(vae.config.latents_mean,
                                dtype=vae.dtype, device=device).view(1, z_dim, 1, 1, 1)
    latents_std = torch.tensor(vae.config.latents_std,
                               dtype=vae.dtype, device=device).view(1, z_dim, 1, 1, 1)

    # --- Round-trip gate ---
    sample_target = pairs[0][3]
    rt_mse = latent_norm_round_trip_test(
        vae, sample_target, args.height, args.width,
        args.num_frames, device, latents_mean, latents_std,
    )

    # --- Canny cache (per unique face) ---
    unique_faces = {idx: png for idx, _, png, _ in pairs}
    print(f"[canny] {len(unique_faces)} unique faces")
    for face_idx, face_png in tqdm(sorted(unique_faces.items()), desc="canny"):
        out_path = out_dir / "canny" / f"face_{face_idx}.pt"
        if out_path.exists():
            continue
        canny = make_canny_uint8(face_png, args.height, args.width,
                                 args.canny_low, args.canny_high)
        torch.save(canny, out_path)

    # --- Latent cache (per unique target) ---
    print(f"[latents] {len(pairs)} targets")
    with torch.no_grad():
        for face_idx, slug, _, tgt_jpg in tqdm(pairs, desc="latents"):
            out_path = out_dir / "latents" / f"face_{face_idx}_{slug}.pt"
            if out_path.exists():
                continue
            x_chw = load_target_to_minus1_plus1(tgt_jpg, args.height, args.width)
            z = encode_latent(vae, x_chw, args.num_frames, device,
                              latents_mean, latents_std)
            torch.save(z, out_path)

    # Free VAE before loading T5
    del vae
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --- T5 prompt embeddings ---
    print("[load] tokenizer + UMT5EncoderModel ...")
    from transformers import AutoTokenizer, UMT5EncoderModel
    tokenizer = AutoTokenizer.from_pretrained(base, subfolder="tokenizer")
    text_encoder_dtype = torch.float32 if device.type == "cpu" else torch.bfloat16
    text_encoder = UMT5EncoderModel.from_pretrained(
        base, subfolder="text_encoder", torch_dtype=text_encoder_dtype,
    )
    text_encoder.eval().requires_grad_(False).to(device)

    unique_slugs = sorted({slug for _, slug, _, _ in pairs})
    max_seq_len = 512

    def encode_text(prompt: str) -> torch.Tensor:
        ti = tokenizer([prompt], padding="max_length", max_length=max_seq_len,
                       truncation=True, add_special_tokens=True,
                       return_attention_mask=True, return_tensors="pt")
        ids = ti.input_ids.to(device)
        mask = ti.attention_mask.to(device)
        with torch.no_grad():
            emb = text_encoder(ids, mask).last_hidden_state  # (1, L, D)
        return emb.squeeze(0).to(torch.bfloat16).cpu()

    print(f"[prompts] encoding {len(unique_slugs)} unique slugs")
    for slug in tqdm(unique_slugs, desc="prompts"):
        out_path = out_dir / "prompts" / f"{slug}.pt"
        if out_path.exists():
            continue
        emb = encode_text(ALL_PROMPTS[slug])
        torch.save(emb, out_path)

    # Negative prompt for end-of-run inference smoke
    neg_path = out_dir / "prompts_negative.pt"
    if not neg_path.exists():
        neg_emb = encode_text(args.negative_prompt)
        torch.save(neg_emb, neg_path)

    del text_encoder, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --- Manifest ---
    manifest = []
    for face_idx, slug, _, _ in pairs:
        manifest.append({
            "face_idx": face_idx,
            "slug": slug,
            "canny_path": str(Path("canny") / f"face_{face_idx}.pt"),
            "latent_path": str(Path("latents") / f"face_{face_idx}_{slug}.pt"),
            "prompt_path": str(Path("prompts") / f"{slug}.pt"),
        })
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"[manifest] wrote {len(manifest)} entries to {manifest_path}")

    # --- Verification: 5 random samples decode back to expected shapes ---
    rng = np.random.default_rng(0)
    sample_idxs = rng.choice(len(manifest), size=min(5, len(manifest)),
                             replace=False)
    for i in sample_idxs:
        rec = manifest[int(i)]
        for key in ("canny_path", "latent_path", "prompt_path"):
            full = out_dir / rec[key]
            t = torch.load(full, map_location="cpu")
            print(f"  [verify] {rec[key]} dtype={t.dtype} shape={tuple(t.shape)}")

    # --- Disk usage report ---
    def du(p: Path) -> int:
        return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
    sizes = {sub: du(out_dir / sub) for sub in ("canny", "latents", "prompts")}
    total_gb = sum(sizes.values()) / (1024 ** 3)
    print(f"[disk] canny={sizes['canny']/1e6:.1f}MB "
          f"latents={sizes['latents']/1e9:.2f}GB "
          f"prompts={sizes['prompts']/1e6:.1f}MB "
          f"total={total_gb:.2f}GB")

    # --- Smoke-result sidecar: lets autofill_card.py pick up roundtrip MSE
    smoke_results_path = Path("training_cards") / "_precompute_meta.json"
    smoke_results_path.parent.mkdir(parents=True, exist_ok=True)
    smoke_results_path.write_text(json.dumps({
        "smoke_latent_roundtrip_mse": rt_mse,
        "cache_disk_gb": round(total_gb, 3),
        "pair_count": len(manifest),
        "timestamp": time.time(),
    }, indent=2))


if __name__ == "__main__":
    main()

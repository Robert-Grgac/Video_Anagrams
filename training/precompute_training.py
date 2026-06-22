"""Precompute the full training cache: silhouette + latents + prompt embeds.

Replaces the older separate ``precompute_silhouette.py``, ``precompute_beta.py``,
and ``precompute_raw_face.py`` scripts. Outputs the manifest the training
``PairDataset`` expects::

    <output_dir>/
        silhouette/face_{idx}.pt        # (3, H, W) uint8 — CN input
        latents/face_{idx}_{slug}.pt    # VAE-encoded target latent, fp16
        prompts/{slug}.pt               # T5 embedding, bf16
        prompts_negative.pt             # T5 embedding of the negative prompt
        manifest.json                   # records read by PairDataset

The silhouette stage runs on CPU (mediapipe); the latent + prompt stages run
on GPU (Wan VAE + UMT5). A VAE encode→decode round-trip gate runs before any
latents are written — wrong normalization constants would otherwise silently
corrupt training.

Resolution and the negative prompt come from ``training.config.TrainConfig``
so the cache stays in lock-step with the training recipe.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from training.config import TrainConfig
from training.face_silhouette import make_option_h_uint8
from utils.prompts import PROMPTS_BATCH_1, PROMPTS_BATCH_2

ALL_PROMPTS: dict[str, str] = {**PROMPTS_BATCH_1, **PROMPTS_BATCH_2}

WAN_T5_MAX_SEQ_LEN = 226


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input_faces_dir", type=str, required=True,
                   help="Dir of face_{idx}.png input faces.")
    p.add_argument("--targets_dir", type=str, required=True,
                   help="Dir of face_{idx}_{slug}.jpg PTDiffusion target images.")
    p.add_argument("--output_dir", type=str, required=True,
                   help="Top-level cache dir. Subdirs silhouette/, latents/, "
                        "prompts/ + manifest.json are written here.")
    p.add_argument("--base_model_path", type=str, required=True,
                   help="Wan-AI Wan2.2-T2V-A14B-Diffusers snapshot dir.")
    p.add_argument("--overwrite", action="store_true",
                   help="Re-render even if a per-stage output file exists.")
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def discover_pairs(faces_dir: Path,
                   targets_dir: Path) -> list[tuple[int, str, Path, Path]]:
    pairs: list[tuple[int, str, Path, Path]] = []
    skipped = 0
    for tgt in sorted(targets_dir.glob("face_*_*.jpg")):
        stem = tgt.stem
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


# ---- silhouette stage (CPU) -------------------------------------------------

def run_silhouette_stage(pairs, out_dir: Path, height: int, width: int,
                         overwrite: bool) -> None:
    sil_dir = out_dir / "silhouette"
    sil_dir.mkdir(parents=True, exist_ok=True)
    unique_faces = {idx: png for idx, _, png, _ in pairs}
    print(f"[silhouette] {len(unique_faces)} unique faces -> {sil_dir}")

    failures: list[str] = []
    written = skipped = 0
    for face_idx, face_png in tqdm(sorted(unique_faces.items()), desc="silhouette"):
        out_path = sil_dir / f"face_{face_idx}.pt"
        if out_path.exists() and not overwrite:
            skipped += 1
            continue
        t = make_option_h_uint8(face_png, height, width)
        if t is None:
            failures.append(face_png.name)
            continue
        torch.save(t, out_path)
        written += 1
    print(f"[silhouette] wrote {written}, skipped {skipped}, failed {len(failures)}")
    if failures:
        for f in failures:
            print(f"  [fail] mediapipe could not detect a face in {f}")
        raise RuntimeError(
            f"{len(failures)} silhouette failure(s); fix the input PNGs before "
            "continuing — partial caches will desync the manifest."
        )


# ---- latents stage (GPU) ----------------------------------------------------

def load_target_to_minus1_plus1(path: Path, height: int, width: int) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    if img.size != (width, height):
        img = img.resize((width, height), Image.LANCZOS)
    arr = np.asarray(img, dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def encode_latent(vae, x_chw: torch.Tensor, num_frames: int,
                  device: torch.device,
                  latents_mean: torch.Tensor,
                  latents_std: torch.Tensor) -> torch.Tensor:
    x = x_chw.unsqueeze(1).expand(-1, num_frames, -1, -1)
    x = x.unsqueeze(0).to(device=device, dtype=vae.dtype)
    z = vae.encode(x).latent_dist.mean
    z_norm = (z - latents_mean) / latents_std
    return z_norm.squeeze(0).to(torch.float16).cpu()


def vae_round_trip_gate(vae, sample_target: Path, height: int, width: int,
                        num_frames: int, device: torch.device,
                        latents_mean: torch.Tensor,
                        latents_std: torch.Tensor) -> float:
    """Encode→normalize→denormalize→decode→MSE. Aborts if MSE > 1e-2."""
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


def run_latents_stage(pairs, out_dir: Path, base_model_path: str,
                      height: int, width: int, num_frames: int,
                      device: torch.device, overwrite: bool) -> tuple[float, int]:
    lat_dir = out_dir / "latents"
    lat_dir.mkdir(parents=True, exist_ok=True)

    print("[load] AutoencoderKLWan ...")
    from diffusers import AutoencoderKLWan
    vae_dtype = torch.float32 if device.type == "cpu" else torch.bfloat16
    vae = AutoencoderKLWan.from_pretrained(base_model_path, subfolder="vae",
                                           torch_dtype=vae_dtype)
    vae.eval().requires_grad_(False).to(device)

    z_dim = vae.config.z_dim
    latents_mean = torch.tensor(vae.config.latents_mean,
                                dtype=vae.dtype, device=device).view(1, z_dim, 1, 1, 1)
    latents_std = torch.tensor(vae.config.latents_std,
                               dtype=vae.dtype, device=device).view(1, z_dim, 1, 1, 1)

    rt_mse = vae_round_trip_gate(vae, pairs[0][3], height, width, num_frames,
                                 device, latents_mean, latents_std)

    print(f"[latents] {len(pairs)} targets -> {lat_dir}")
    written = skipped = 0
    with torch.no_grad():
        for face_idx, slug, _, tgt_jpg in tqdm(pairs, desc="latents"):
            out_path = lat_dir / f"face_{face_idx}_{slug}.pt"
            if out_path.exists() and not overwrite:
                skipped += 1
                continue
            x_chw = load_target_to_minus1_plus1(tgt_jpg, height, width)
            z = encode_latent(vae, x_chw, num_frames, device, latents_mean, latents_std)
            torch.save(z, out_path)
            written += 1

    del vae
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"[latents] wrote {written}, skipped {skipped}")
    return rt_mse, len(pairs)


# ---- prompts stage (GPU) ----------------------------------------------------

def run_prompts_stage(pairs, out_dir: Path, base_model_path: str,
                      negative_prompt: str, device: torch.device,
                      overwrite: bool) -> None:
    pr_dir = out_dir / "prompts"
    pr_dir.mkdir(parents=True, exist_ok=True)

    print("[load] tokenizer + UMT5EncoderModel ...")
    from transformers import AutoTokenizer, UMT5EncoderModel
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, subfolder="tokenizer")
    enc_dtype = torch.float32 if device.type == "cpu" else torch.bfloat16
    text_encoder = UMT5EncoderModel.from_pretrained(
        base_model_path, subfolder="text_encoder", torch_dtype=enc_dtype,
    )
    text_encoder.eval().requires_grad_(False).to(device)

    # Match the inference pipeline's prompt encoder byte-for-byte: same cleaning,
    # same zero-replacement of T5 padding positions. Asymmetric pad handling
    # between cached positives and live negatives silently breaks CFG.
    from pipeline.wan_t2v_controlnet_pipeline import prompt_clean

    def encode_text(prompt: str) -> torch.Tensor:
        cleaned = prompt_clean(prompt)
        ti = tokenizer([cleaned], padding="max_length", max_length=WAN_T5_MAX_SEQ_LEN,
                       truncation=True, add_special_tokens=True,
                       return_attention_mask=True, return_tensors="pt")
        ids = ti.input_ids.to(device)
        mask = ti.attention_mask.to(device)
        seq_lens = mask.gt(0).sum(dim=1).long()
        with torch.no_grad():
            emb = text_encoder(ids, mask).last_hidden_state
        emb = emb.to(torch.bfloat16)
        emb_list = [u[:v] for u, v in zip(emb, seq_lens)]
        emb = torch.stack(
            [torch.cat([u, u.new_zeros(WAN_T5_MAX_SEQ_LEN - u.size(0), u.size(1))])
             for u in emb_list], dim=0,
        )
        return emb.squeeze(0).cpu()

    unique_slugs = sorted({slug for _, slug, _, _ in pairs})
    print(f"[prompts] {len(unique_slugs)} unique slugs -> {pr_dir} "
          f"(T5 max_seq_len={WAN_T5_MAX_SEQ_LEN})")
    written = skipped = 0
    for slug in tqdm(unique_slugs, desc="prompts"):
        out_path = pr_dir / f"{slug}.pt"
        if out_path.exists() and not overwrite:
            skipped += 1
            continue
        torch.save(encode_text(ALL_PROMPTS[slug]), out_path)
        written += 1

    neg_path = out_dir / "prompts_negative.pt"
    if overwrite or not neg_path.exists():
        torch.save(encode_text(negative_prompt), neg_path)
        print(f"[prompts] wrote negative embed -> {neg_path}")
    print(f"[prompts] wrote {written}, skipped {skipped}")

    del text_encoder, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---- manifest ----------------------------------------------------------------

def write_manifest(pairs, out_dir: Path) -> Path:
    manifest = [
        {
            "face_idx": face_idx,
            "slug": slug,
            "control_path": str(Path("silhouette") / f"face_{face_idx}.pt"),
            "latent_path": str(Path("latents") / f"face_{face_idx}_{slug}.pt"),
            "prompt_path": str(Path("prompts") / f"{slug}.pt"),
        }
        for face_idx, slug, _, _ in pairs
    ]
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"[manifest] wrote {len(manifest)} entries to {manifest_path}")
    return manifest_path


# ---- main --------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    cfg = TrainConfig()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = discover_pairs(Path(args.input_faces_dir), Path(args.targets_dir))
    if not pairs:
        raise RuntimeError("No valid (face, target) pairs found.")

    device = torch.device(args.device)

    t0 = time.time()
    run_silhouette_stage(pairs, out_dir, cfg.height, cfg.width, args.overwrite)
    rt_mse, _ = run_latents_stage(pairs, out_dir, args.base_model_path,
                                  cfg.height, cfg.width, cfg.num_frames,
                                  device, args.overwrite)
    run_prompts_stage(pairs, out_dir, args.base_model_path,
                      cfg.negative_prompt, device, args.overwrite)
    write_manifest(pairs, out_dir)

    def du(p: Path) -> int:
        return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
    sizes = {sub: du(out_dir / sub) for sub in ("silhouette", "latents", "prompts")
             if (out_dir / sub).exists()}
    total_gb = sum(sizes.values()) / (1024 ** 3)
    print(f"[disk] silhouette={sizes.get('silhouette', 0)/1e6:.1f}MB "
          f"latents={sizes.get('latents', 0)/1e9:.2f}GB "
          f"prompts={sizes.get('prompts', 0)/1e6:.1f}MB "
          f"total={total_gb:.2f}GB")
    print(f"[done] precompute finished in {time.time() - t0:.1f}s; "
          f"VAE round-trip MSE = {rt_mse:.6f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""BETA dataset: loads only precomputed cache files."""
from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset


class BetaPairDataset(Dataset):
    """Yield (canny, latent, prompt_embeds) tuples from the precompute cache.

    Manifest entries store paths *relative* to ``cache_dir``; the constructor
    resolves them once so ``__getitem__`` does plain ``torch.load`` calls.
    """

    def __init__(self, cache_dir: str | Path, num_frames: int):
        self.cache_dir = Path(cache_dir)
        self.num_frames = int(num_frames)
        manifest_path = self.cache_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"manifest.json not found in {self.cache_dir}; "
                "run precompute_beta first."
            )
        self.records = json.loads(manifest_path.read_text())

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        r = self.records[idx]
        canny_u8 = torch.load(self.cache_dir / r["canny_path"], map_location="cpu")
        canny = canny_u8.float() / 127.5 - 1.0
        canny = canny.unsqueeze(1).expand(-1, self.num_frames, -1, -1).contiguous()
        canny = canny.to(torch.bfloat16)

        latent = torch.load(self.cache_dir / r["latent_path"], map_location="cpu")
        latent = latent.to(torch.bfloat16)

        prompt = torch.load(self.cache_dir / r["prompt_path"], map_location="cpu")
        prompt = prompt.to(torch.bfloat16)

        return {
            "canny": canny,
            "latent": latent,
            "prompt_embeds": prompt,
            "face_idx": r["face_idx"],
            "slug": r["slug"],
        }

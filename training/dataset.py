"""Training dataset: loads precomputed silhouette + latent + prompt-embed triples."""
from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

CONTROL_SUBDIR = "silhouette"


class PairDataset(Dataset):
    """Yield (control, latent, prompt_embeds) tuples from the precompute cache.

    Manifest entries store paths relative to ``cache_dir``; the constructor
    resolves them once so ``__getitem__`` does plain ``torch.load`` calls.
    """

    def __init__(self, cache_dir: str | Path, num_frames: int):
        self.cache_dir = Path(cache_dir)
        self.num_frames = int(num_frames)
        manifest_path = self.cache_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"manifest.json not found in {self.cache_dir}; "
                "run precompute_training.py first."
            )
        self.records = json.loads(manifest_path.read_text())

    def control_path(self, record: dict) -> Path:
        name = Path(record["control_path"]).name
        return self.cache_dir / CONTROL_SUBDIR / name

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        r = self.records[idx]
        control_u8 = torch.load(self.control_path(r), map_location="cpu")
        control = control_u8.float() / 127.5 - 1.0
        control = control.unsqueeze(1).expand(-1, self.num_frames, -1, -1).contiguous()
        control = control.to(torch.bfloat16)

        latent = torch.load(self.cache_dir / r["latent_path"], map_location="cpu")
        latent = latent.to(torch.bfloat16)

        prompt = torch.load(self.cache_dir / r["prompt_path"], map_location="cpu")
        prompt = prompt.to(torch.bfloat16)

        return {
            "control": control,
            "latent": latent,
            "prompt_embeds": prompt,
            "face_idx": r["face_idx"],
            "slug": r["slug"],
        }

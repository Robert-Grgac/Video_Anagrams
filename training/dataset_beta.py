"""BETA dataset: loads only precomputed cache files."""
from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset


class BetaPairDataset(Dataset):
    """Yield (control, latent, prompt_embeds) tuples from the precompute cache.

    Manifest entries store paths *relative* to ``cache_dir``; the constructor
    resolves them once so ``__getitem__`` does plain ``torch.load`` calls.

    ``control_subdir`` selects which precomputed control modality is loaded.
    The manifest's ``canny_path`` field stores something like
    ``"canny/face_0.pt"``; the dataset takes only the basename and re-routes
    the load to ``<cache_dir>/<control_subdir>/face_0.pt``. The on-disk
    schema (3, H, W) uint8 is identical across modalities, so the rest of
    the loop is unchanged. Default is ``"canny"`` for backward compat.

    The yielded dict still uses the key ``"canny"`` for the loaded control
    tensor — to avoid touching every training-loop call site. Treat it as
    the generic control input regardless of modality.
    """

    def __init__(self, cache_dir: str | Path, num_frames: int,
                 control_subdir: str = "canny"):
        self.cache_dir = Path(cache_dir)
        self.num_frames = int(num_frames)
        self.control_subdir = str(control_subdir)
        manifest_path = self.cache_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"manifest.json not found in {self.cache_dir}; "
                "run precompute_beta first."
            )
        self.records = json.loads(manifest_path.read_text())

    def control_path(self, record: dict) -> Path:
        """Resolve a record's control-modality file path under control_subdir."""
        name = Path(record["canny_path"]).name
        return self.cache_dir / self.control_subdir / name

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        r = self.records[idx]
        canny_u8 = torch.load(self.control_path(r), map_location="cpu")
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

"""Prompt-axis metric: ViCLIP video<->prompt cosine similarity (GPU).

Primary backend = InternVideo ViCLIP-L-14 (vendored module under
analysis/third_party/viclip/ + ViClip-InternVid-10M-FLT.pth). Per video: sample
ViCLIP's standard 8 frames, encode video + the prompt text (ALL_PROMPTS[slug]),
cosine similarity -> `viclip`.

Fallback backend (`--backend openclip`): frame-averaged open_clip ViT-L-14
CLIPScore (documented limitation, §6.1/§10).
See docs/IMPLEMENTATION_PLAN_EVAL.md §6.1.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.eval_common import (  # noqa: E402
    ALL_PROMPTS,
    add_common_args,
    iter_videos,
    read_frames,
    resolve_conditions,
    write_rows,
    write_summary,
)

METRIC = "viclip"


def _resolve_viclip_ckpt() -> str:
    cands = []
    if os.environ.get("VICLIP_CKPT"):
        cands.append(Path(os.environ["VICLIP_CKPT"]))
    here = Path(__file__).parent / "third_party" / "viclip"
    cands.append(here / "ViClip-InternVid-10M-FLT.pth")
    cands.append(Path.home() / "checkpoints" / "viclip" / "ViClip-InternVid-10M-FLT.pth")
    for c in cands:
        if c.exists():
            return str(c)
    raise FileNotFoundError(
        "ViCLIP checkpoint not found. Set VICLIP_CKPT or place "
        "ViClip-InternVid-10M-FLT.pth under analysis/third_party/viclip/ "
        f"(tried: {[str(c) for c in cands]})"
    )


class ViClipScorer:
    """Wraps the vendored ViCLIP module. cosine(video, prompt) in [-1,1]."""

    def __init__(self, device: str):
        self.device = device
        sys.path.insert(0, str(Path(__file__).parent / "third_party"))
        from viclip import ViCLIP, frames2tensor  # noqa: E402
        from viclip.simple_tokenizer import SimpleTokenizer  # noqa: E402

        self._frames2tensor = frames2tensor
        ckpt = _resolve_viclip_ckpt()
        print(f"[viclip] loading {ckpt}")
        tokenizer = SimpleTokenizer()
        self.model = ViCLIP(tokenizer=tokenizer, size="l", pretrain=ckpt)
        self.model = self.model.to(device).eval()
        self.tokenizer = tokenizer
        self._text_cache: dict[str, torch.Tensor] = {}

    def _text_feat(self, prompt: str) -> torch.Tensor:
        # get_text_features L2-normalizes; cache per unique prompt.
        return self.model.get_text_features(prompt, self.tokenizer, self._text_cache)

    def score(self, frames_rgb: np.ndarray, prompt: str) -> float:
        # frames2tensor expects cv2-style BGR (it flips ::-1 internally), so
        # hand it BGR to recover the RGB ImageNet normalization it was built for.
        # ascontiguousarray: cv2.resize rejects the negative-stride ::-1 view.
        frames_bgr = [np.ascontiguousarray(f[:, :, ::-1]) for f in frames_rgb]
        tube = self._frames2tensor(frames_bgr, fnum=8, device=torch.device(self.device))
        vid_feat = self.model.get_vid_features(tube)  # [1, D], L2-normed
        txt_feat = self._text_feat(prompt)  # [1, D], L2-normed
        return float((vid_feat @ txt_feat.T).squeeze().item())


class OpenClipScorer:
    """Fallback: frame-averaged open_clip ViT-L-14 CLIPScore."""

    def __init__(self, device: str, n_frames: int = 8):
        import open_clip

        self.device = device
        self.n_frames = n_frames
        pretrained = os.environ.get("OPENCLIP_PRETRAINED", "openai")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-L-14", pretrained=pretrained
        )
        self.model = self.model.to(device).eval()
        self.tokenizer = open_clip.get_tokenizer("ViT-L-14")
        self._text_cache: dict[str, torch.Tensor] = {}

    def _text_feat(self, prompt: str) -> torch.Tensor:
        if prompt not in self._text_cache:
            tok = self.tokenizer([prompt]).to(self.device)
            with torch.no_grad():
                f = self.model.encode_text(tok).float()
                f /= f.norm(dim=-1, keepdim=True)
            self._text_cache[prompt] = f
        return self._text_cache[prompt]

    def score(self, frames_rgb: np.ndarray, prompt: str) -> float:
        from PIL import Image

        n = len(frames_rgb)
        idxs = np.linspace(0, n - 1, self.n_frames).round().astype(int)
        batch = torch.stack(
            [self.preprocess(Image.fromarray(frames_rgb[i])) for i in idxs]
        ).to(self.device)
        with torch.no_grad():
            feats = self.model.encode_image(batch).float()
            feats /= feats.norm(dim=-1, keepdim=True)
            vid_feat = feats.mean(dim=0, keepdim=True)
            vid_feat /= vid_feat.norm(dim=-1, keepdim=True)
        txt_feat = self._text_feat(prompt)
        return float((vid_feat @ txt_feat.T).squeeze().item())


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    add_common_args(p)
    p.add_argument("--backend", default="viclip", choices=["viclip", "openclip", "auto"])
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    conds = resolve_conditions(args)

    backend = args.backend
    if backend == "auto":
        try:
            scorer = ViClipScorer(args.device)
            backend = "viclip"
        except Exception as e:
            print(f"[auto] ViCLIP unavailable ({e}); falling back to open_clip")
            scorer = OpenClipScorer(args.device)
            backend = "openclip"
    elif backend == "viclip":
        scorer = ViClipScorer(args.device)
    else:
        scorer = OpenClipScorer(args.device)
    print(f"[backend] {backend}")

    rows = []
    for cond, cdir in conds.items():
        n = 0
        for path, face_idx, slug in iter_videos(cdir):
            prompt = ALL_PROMPTS[slug]
            frames = read_frames(path)
            s = scorer.score(frames, prompt)
            rows.append(
                {"condition": cond, "face_idx": face_idx, "slug": slug, "viclip": s}
            )
            n += 1
        print(f"[{cond}] {n} videos")

    if not rows:
        raise SystemExit("no rows produced")

    write_rows(METRIC, rows, args.results_dir)
    import pandas as pd

    write_summary(
        METRIC, pd.read_csv(Path(args.results_dir) / f"{METRIC}.csv"),
        args.results_dir, ["viclip"],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

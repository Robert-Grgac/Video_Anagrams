"""Face structural inputs for the ControlNet — alternatives to canny edges.

Two silhouette extractors used by ``precompute_training.py``:

- ``make_binary_silhouette_uint8`` (option A) — filled FACE_OVAL polygon, white
  on black. Pure shape, no interior features.
- ``make_option_h_uint8`` (option H) — filled FACE_OVAL at a gray intermediate
  value PLUS eye / eyebrow / lips / nose contour lines drawn in white. Combines
  silhouette and interior structure in one map.

Both return a (3, H, W) uint8 ``torch.Tensor`` (channels replicated) or ``None``
if MediaPipe can't find a face. The downstream dataset normalises to [-1, 1].

Uses MediaPipe FaceMesh on CPU; no GPU needed. Must be imported from an env
that has ``mediapipe`` installed (e.g. the dedicated ``silhouette`` env).
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import torch
from PIL import Image


_FACE_MESH = None


def _get_face_mesh():
    global _FACE_MESH
    if _FACE_MESH is None:
        _FACE_MESH = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.3,
        )
    return _FACE_MESH


def _load_rgb(face_png: Path, height: int, width: int) -> np.ndarray:
    img = Image.open(face_png).convert("RGB")
    if img.size != (width, height):
        img = img.resize((width, height), Image.LANCZOS)
    return np.asarray(img, dtype=np.uint8)


def _detect_landmarks(rgb: np.ndarray) -> np.ndarray | None:
    """Return (N, 2) int32 pixel coords for the first detected face, or None."""
    res = _get_face_mesh().process(rgb)
    if not res.multi_face_landmarks:
        return None
    h, w = rgb.shape[:2]
    pts = np.array(
        [(lm.x * w, lm.y * h) for lm in res.multi_face_landmarks[0].landmark],
        dtype=np.float32,
    )
    return pts.astype(np.int32)


def _ordered_loop(edges: set) -> list[int]:
    """Chain an undirected closed-loop edge set into an ordered vertex list."""
    adj: dict[int, list[int]] = defaultdict(list)
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
    start = min(adj)
    ordered = [start]
    prev = None
    cur = start
    for _ in range(len(adj)):
        nbrs = [n for n in adj[cur] if n != prev]
        if not nbrs:
            break
        nxt = nbrs[0]
        if nxt == start:
            break
        ordered.append(nxt)
        prev = cur
        cur = nxt
    return ordered


_FACE_OVAL_ORDER: list[int] | None = None


def _face_oval_indices() -> list[int]:
    global _FACE_OVAL_ORDER
    if _FACE_OVAL_ORDER is None:
        _FACE_OVAL_ORDER = _ordered_loop(
            set(mp.solutions.face_mesh.FACEMESH_FACE_OVAL)
        )
    return _FACE_OVAL_ORDER


# Interior contour groups exposed by mediapipe's face_mesh constants. NOSE may
# not exist on older mediapipe builds; the loop guards via getattr.
_INTERIOR_CONTOUR_NAMES = (
    "FACEMESH_LEFT_EYE",
    "FACEMESH_RIGHT_EYE",
    "FACEMESH_LEFT_EYEBROW",
    "FACEMESH_RIGHT_EYEBROW",
    "FACEMESH_LIPS",
    "FACEMESH_NOSE",
)


def _draw_edges(canvas: np.ndarray, pts: np.ndarray, edges, thickness: int,
                color: int) -> None:
    for a, b in edges:
        cv2.line(canvas, tuple(pts[a]), tuple(pts[b]),
                 color, thickness, cv2.LINE_AA)


def make_binary_silhouette_uint8(face_png: Path, height: int,
                                 width: int) -> torch.Tensor | None:
    rgb = _load_rgb(face_png, height, width)
    pts = _detect_landmarks(rgb)
    if pts is None:
        return None
    oval = pts[_face_oval_indices()]
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [oval], 255)
    return torch.from_numpy(np.stack([mask, mask, mask], axis=0))


def make_option_h_uint8(face_png: Path, height: int, width: int,
                        fill_value: int = 128, line_value: int = 255,
                        line_thickness: int = 2) -> torch.Tensor | None:
    rgb = _load_rgb(face_png, height, width)
    pts = _detect_landmarks(rgb)
    if pts is None:
        return None
    canvas = np.zeros((height, width), dtype=np.uint8)
    oval = pts[_face_oval_indices()]
    cv2.fillPoly(canvas, [oval], fill_value)

    fm = mp.solutions.face_mesh
    for name in _INTERIOR_CONTOUR_NAMES:
        edges = getattr(fm, name, None)
        if edges is None:
            continue
        _draw_edges(canvas, pts, edges,
                    thickness=line_thickness, color=line_value)
    return torch.from_numpy(np.stack([canvas, canvas, canvas], axis=0))

"""Shared helpers used across all beta training/smoke/inference scripts.

Extracted from beta-001's `train.py` and beta-005's `train.py` so that no
per-beta script needs to import from another beta's directory.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn


def cast_respecting_fp32_modules(model: nn.Module, dtype: torch.dtype) -> None:
    """Cast every parameter to ``dtype`` except those whose qualified name
    matches a substring in ``model._keep_in_fp32_modules``.

    Diffusers' built-in loaders honor that list; constructing a model from a
    bare config does not, so a blanket ``.to(bf16)`` would silently demote
    norms / sinusoidal time embeds / scale_shift tables.
    """
    keep = list(getattr(model, "_keep_in_fp32_modules", []) or [])
    skipped: list[str] = []
    casted: list[str] = []
    for name, param in model.named_parameters():
        if any(k in name for k in keep):
            skipped.append(name)
        else:
            param.data = param.data.to(dtype)
            casted.append(name)
    for name, buf in model.named_buffers():
        if any(k in name for k in keep):
            continue
        buf.data = buf.data.to(dtype)
    print(f"[cast] {len(casted)} params -> {dtype}; "
          f"{len(skipped)} kept fp32 (e.g. {skipped[:3]})")


def detect_boundary_ratio(base_model_path: str | Path,
                          transformer_config: dict,
                          override: Optional[float] = None) -> tuple[float, str]:
    """Return (boundary_ratio, source_string).

    Priority: explicit ``override`` -> ``transformer.config.boundary_ratio``
    -> ``model_index.json`` of the base pipeline -> default ``0.5`` (upper 50%).
    """
    if override is not None:
        return float(override), "cli_override"
    for key in ("boundary_ratio", "boundary_sigma"):
        v = transformer_config.get(key)
        if v is not None:
            return float(v), f"transformer.config.{key}"
    mi = Path(base_model_path) / "model_index.json"
    if mi.exists():
        try:
            data = json.loads(mi.read_text())
            v = data.get("boundary_ratio")
            if v is not None:
                return float(v), "model_index.json.boundary_ratio"
        except Exception:
            pass
    return 0.5, "fallback_upper_50pct"


def mean_residual_l2(residuals) -> float:
    if residuals is None:
        return 0.0
    if isinstance(residuals, (list, tuple)):
        vals = [r.detach().float().pow(2).mean().sqrt().item() for r in residuals]
        return float(np.mean(vals))
    return float(residuals.detach().float().pow(2).mean().sqrt().item())


def _collate_keep_meta(samples):
    """Tensor-stack tensor fields, list-collect scalar fields."""
    out = {}
    for k in samples[0]:
        if torch.is_tensor(samples[0][k]):
            out[k] = torch.stack([s[k] for s in samples], dim=0)
        else:
            out[k] = [s[k] for s in samples]
    return out


def _save_safetensors(model: nn.Module, path: Path) -> None:
    from safetensors.torch import save_file
    sd = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(sd, str(path))


def _save_state_dict(state_dict: dict, path: Path) -> None:
    from safetensors.torch import save_file
    sd = {k: v.detach().cpu().contiguous() for k, v in state_dict.items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(sd, str(path))


def _format_seconds(s: float) -> str:
    s = int(s)
    h, r = divmod(s, 3600)
    m, sec = divmod(r, 60)
    return f"{h:02d}:{m:02d}:{sec:02d}"


def _maybe_force_native_attention(model: nn.Module, label: str) -> None:
    """If the active GPU is Blackwell-class (compute_cap >= 12), force the
    diffusers attention dispatcher to NATIVE for this model. The flash-attn
    package can be import-clean on wan22-bw without the SM 12.0 kernels being
    available, in which case ``flash_attn_func`` is None at call time and the
    first forward crashes (see beta-004 attempt 1)."""
    if not torch.cuda.is_available():
        return
    try:
        major, _minor = torch.cuda.get_device_capability(0)
    except Exception:
        return
    forced = os.environ.get("WAN_FORCE_NATIVE_ATTN", "").strip() == "1"
    if major < 12 and not forced:
        return
    try:
        model.set_attention_backend("native")
        print(f"[attn] {label}: forced NATIVE backend (compute_cap={major}.x)")
    except Exception as e:
        print(f"[attn] {label}: set_attention_backend('native') failed: {e}",
              file=sys.stderr)


def _build_controlnet_from_checkpoint(controlnet_config_repo: str,
                                      checkpoint_path: Path):
    from safetensors.torch import load_file
    from wan_controlnet import WanControlnet
    config = WanControlnet.load_config(controlnet_config_repo)
    cn = WanControlnet.from_config(config)
    cast_respecting_fp32_modules(cn, torch.bfloat16)
    sd = load_file(str(checkpoint_path))
    missing, unexpected = cn.load_state_dict(sd, strict=False)
    if missing:
        print(f"[load-ckpt] WARN missing keys: {len(missing)} (e.g. {missing[:2]})")
    if unexpected:
        print(f"[load-ckpt] WARN unexpected keys: {len(unexpected)} (e.g. {unexpected[:2]})")
    cn.eval()
    return cn

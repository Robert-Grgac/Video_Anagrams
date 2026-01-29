from __future__ import annotations
import torch
from typing import Sequence


def cfg(
    uncond: torch.Tensor,
    cond: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """
    Applies classifier-free guidance.

    Args:
        uncond (Tensor):
            Model output conditioned on the "unconditional" prompt.
        cond (Tensor):
            Model output conditioned on the target prompt.
        scale (float):
            Guidance scale (γ). If scale == 1, this returns cond.

    Returns:
        Tensor:
            Guided model output.
    """
    if scale == 1.0:
        return cond
    return uncond + scale * (cond - uncond)


def mean_reduce(tensors: Sequence[torch.Tensor]) -> torch.Tensor:
    """
    Computes the arithmetic mean of a sequence of tensors.

    Args:
        tensors (Sequence[Tensor]):
            List or tuple of tensors with identical shape.

    Returns:
        Tensor:
            Element-wise mean.
    """
    if len(tensors) == 0:
        raise ValueError("mean_reduce() received an empty sequence.")
    return torch.stack(tensors, dim=0).mean(dim=0)


def mix_joint(
    out_a: torch.Tensor,
    out_b: torch.Tensor,
) -> torch.Tensor:
    """
    Joint-stage mixing (MatchDiffusion-style).

    This function assumes:
        out_a = CFG(z, prompt A)
        out_b = CFG(z, prompt B)

    It returns:
        0.5 * (out_a + out_b)
    """
    return mean_reduce([out_a, out_b])


def mix_anagram(
    view_outputs: Sequence[torch.Tensor],
) -> torch.Tensor:
    """
    Anagram-stage mixing (Visual Anagrams-style).

    This function assumes that:
        - Each tensor in view_outputs is already inverse-aligned
          into the same latent coordinate system.
        - Each tensor corresponds to a different (view, prompt) pair.

    It returns:
        mean(view_outputs)
    """
    return mean_reduce(view_outputs)

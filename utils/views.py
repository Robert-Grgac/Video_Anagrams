from __future__ import annotations
import torch


class ViewTransform:
    """
    Base class for all view transforms.

    A view is a deterministic, invertible transform v(x) applied to latent tensors.
    Subclasses must implement:
        - forward(x): apply the view
        - inverse(x): undo the view
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


    def assert_inverse_ok(self, x: torch.Tensor, atol: float = 1e-6):
        """
        Checks that inverse(forward(x)) ≈ x.

        This should be called during development / debugging,
        not during production inference.
        """
        x_rec = self.inverse(self.forward(x))
        if not torch.allclose(x, x_rec, atol=atol):
            max_err = (x - x_rec).abs().max().item()
            raise AssertionError(
                f"{self.__class__.__name__}: inverse check failed "
                f"(max error = {max_err})"
            )

    def assert_norm_preserved(self, x: torch.Tensor, rtol: float = 1e-5):
        """
        Checks that ||x||₂ ≈ ||v(x)||₂.

        This is a proxy test for orthogonality / noise-statistics preservation.
        """
        x_norm = torch.norm(x)
        v_norm = torch.norm(self.forward(x))
        if not torch.allclose(x_norm, v_norm, rtol=rtol):
            raise AssertionError(
                f"{self.__class__.__name__}: norm not preserved "
                f"(||x||={x_norm.item()}, ||v(x)||={v_norm.item()})"
            )


class IdentityView(ViewTransform):
    """
    Identity view: v(x) = x.

    Used for prompt A in the anagram pipeline.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        return x


class HorizontalFlipView(ViewTransform):
    """
    Horizontal flip over width axis.

    Latent shape: (B, C, T, H, W)
    Width axis = -1
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Flip width dimension
        return torch.flip(x, dims=[-1])

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        # Horizontal flip is self-inverse
        return torch.flip(x, dims=[-1])

class Rotate180View(ViewTransform):
    """
    Rotate 180° (flip both height and width axes).

    Latent shape: (B, C, T, H, W)
    Height axis = -2
    Width axis = -1
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.rot90(x, k=2, dims=[-2, -1])  

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        # Rotate 180° is self-inverse
        return torch.rot90(x, k=2, dims=[-2, -1])


class VerticalFlipView(ViewTransform):
    """
    Vertical flip over height axis.

    Latent shape: (B, C, T, H, W)
    Height axis = -2
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.flip(x, dims=[-2])

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        # Self-inverse
        return torch.flip(x, dims=[-2])


class TimeReverseView(ViewTransform):
    """
    Temporal reversal.

    Reverses the latent sequence in time.
    Latent shape: (B, C, T, H, W)
    Time axis = -3
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.flip(x, dims=[-3])

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        # Self-inverse
        return torch.flip(x, dims=[-3])

class OuterCircleRotate180View(ViewTransform):
    """
    "Outer circle" view (permutation-based, invertible):

    - Inner circle (radius r) stays fixed.
    - Outer region (everything outside the circle) is rotated 180° about the center.

    This is a pure permutation (no interpolation), so it's:
      * deterministic
      * exactly invertible (self-inverse)
      * norm/noise-statistics preserving (up to floating-point reduction noise in norm())
    """

    def __init__(self, inner_radius_ratio: float = 3 / 8):
        """
        Args:
            inner_radius_ratio:
                Radius of the inner circle as a fraction of min(H, W).
                Visual Anagrams used r = 3/8 * im_size for their inner-circle demo.
        """
        if inner_radius_ratio <= 0:
            raise ValueError("inner_radius_ratio must be > 0.")
        self.inner_radius_ratio = float(inner_radius_ratio)

        # Cache masks per (H, W, device) to avoid recomputing every call.
        # Value is a 2D boolean tensor of shape (H, W) on that device.
        self._mask_cache: dict[tuple[int, int, torch.device], torch.Tensor] = {}

    def _get_inner_mask(self, H: int, W: int, device: torch.device) -> torch.Tensor:
        key = (H, W, device)
        mask = self._mask_cache.get(key, None)
        if mask is not None:
            return mask

        # invariant under 180° rotation (flip H and W).
        yy = torch.arange(H, device=device) - (H - 1) / 2.0
        xx = torch.arange(W, device=device) - (W - 1) / 2.0
        Y, X = torch.meshgrid(yy, xx, indexing="ij")

        r = self.inner_radius_ratio * float(min(H, W))
        mask = (Y * Y + X * X) < (r * r)  # (H, W) bool

        self._mask_cache[key] = mask
        return mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, T, H, W) or (B, C, H, W)

        Returns:
            Tensor of same shape as x.
        """
        if x.ndim < 4:
            raise ValueError(f"Expected at least 4D tensor, got shape {tuple(x.shape)}")

        H, W = x.shape[-2], x.shape[-1]
        inner_mask_2d = self._get_inner_mask(H, W, x.device)  # (H, W), bool

        # Broadcast mask to x's rank: -> (1,1,H,W) or (1,1,1,H,W)
        mask = inner_mask_2d
        while mask.ndim < x.ndim:
            mask = mask.unsqueeze(0)

        # 180° rotation = flip H and W
        x_rot = torch.flip(x, dims=[-2, -1])

        # Keep inner circle from x, take outer ring from rotated x.
        return torch.where(mask, x, x_rot)

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        # Self-inverse because rotating the outer ring by 180° twice restores it.
        return self.forward(x)
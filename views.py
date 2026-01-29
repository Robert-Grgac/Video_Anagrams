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

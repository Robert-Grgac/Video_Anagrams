import torch
import pytest

from video_anagrams.views import (
    IdentityView,
    HorizontalFlipView,
    VerticalFlipView,
    TimeReverseView,
)

@pytest.fixture
def latents():
    # Fake WAN latent: (B, C, T, H, W)
    return torch.randn(2, 4, 5, 8, 8)


@pytest.mark.parametrize("view_cls", [
    IdentityView,
    HorizontalFlipView,
    VerticalFlipView,
    TimeReverseView,
])
def test_inverse_property(view_cls, latents):
    view = view_cls()
    view.assert_inverse_ok(latents)


@pytest.mark.parametrize("view_cls", [
    IdentityView,
    HorizontalFlipView,
    VerticalFlipView,
    TimeReverseView,
])
def test_norm_preservation(view_cls, latents):
    view = view_cls()
    view.assert_norm_preserved(latents)


@pytest.mark.parametrize("view_cls", [
    IdentityView,
    HorizontalFlipView,
    VerticalFlipView,
    TimeReverseView,
])
def test_shape_preserved(view_cls, latents):
    view = view_cls()
    y = view.forward(latents)
    assert y.shape == latents.shape

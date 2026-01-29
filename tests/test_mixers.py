import torch
from video_anagrams.mixers import cfg, mean_reduce, mix_joint, mix_anagram


def test_cfg_identity():
    x = torch.randn(4, 3)
    assert torch.allclose(cfg(x, x, scale=5.0), x)


def test_cfg_scale_one():
    u = torch.randn(4, 3)
    c = torch.randn(4, 3)
    assert torch.allclose(cfg(u, c, scale=1.0), c)


def test_cfg_formula():
    u = torch.randn(4, 3)
    c = torch.randn(4, 3)
    s = 2.5
    out = cfg(u, c, s)
    expected = u + s * (c - u)
    assert torch.allclose(out, expected)


def test_mean_reduce():
    a = torch.ones(2, 2)
    b = torch.zeros(2, 2)
    out = mean_reduce([a, b])
    assert torch.allclose(out, 0.5 * torch.ones(2, 2))


def test_mix_joint():
    a = torch.randn(3, 4)
    b = torch.randn(3, 4)
    out = mix_joint(a, b)
    assert torch.allclose(out, 0.5 * (a + b))


def test_mix_anagram_multi_view():
    xs = [torch.randn(3, 4) for _ in range(3)]
    out = mix_anagram(xs)
    expected = sum(xs) / len(xs)
    assert torch.allclose(out, expected)

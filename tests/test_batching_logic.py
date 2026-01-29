import torch


def test_joint_stage_batching_shapes():
    B, C, T, H, W = 1, 4, 5, 8, 8
    z = torch.randn(B, C, T, H, W)

    # simulate joint batching
    latent_batch = torch.cat([z, z, z], dim=0)
    assert latent_batch.shape[0] == 3 * B

    # simulate model output
    fake_out = torch.randn_like(latent_batch)

    u, a, b = fake_out.chunk(3, dim=0)
    assert u.shape == a.shape == b.shape == z.shape


def test_anagram_stage_batching_shapes():
    B, C, T, H, W = 1, 4, 5, 8, 8
    z = torch.randn(B, C, T, H, W)

    z1 = z
    z2 = torch.flip(z, dims=[-1])

    latent_batch = torch.cat([z1, z1, z2, z2], dim=0)
    assert latent_batch.shape[0] == 4 * B

    fake_out = torch.randn_like(latent_batch)

    u1, c1, u2, c2 = fake_out.chunk(4, dim=0)
    assert u1.shape == c1.shape == u2.shape == c2.shape == z.shape

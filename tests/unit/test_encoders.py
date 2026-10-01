import numpy as np
import pytest

torch = pytest.importorskip("torch")

from rat_usv.representations.encoders import Encoder, augment, nt_xent  # noqa: E402
from rat_usv.representations.vae import VAE  # noqa: E402


def test_vae_and_encoder_shapes():
    x = torch.rand(2, 1, 128, 128)
    vae = VAE()
    mu, logvar = vae.encode(x)
    assert mu.shape == (2, 32) and vae.decode(mu).shape == (2, 1, 128, 128)
    assert Encoder()(x).shape == (2, 32)


def test_augment_never_moves_energy_across_frequency():
    x = torch.zeros(4, 1, 128, 128)
    x[:, :, 40, 30:90] = 1.0  # a flat tone in one frequency row
    y = augment(x, torch.Generator().manual_seed(0))
    row_energy = y[:, 0].sum(dim=2)  # per frequency row
    others = torch.cat([row_energy[:, :40], row_energy[:, 41:]], dim=1)
    assert torch.all(row_energy[:, 40] > 10 * others.mean(dim=1))


def test_nt_xent_prefers_matching_pairs():
    a = torch.nn.functional.normalize(torch.randn(8, 16), dim=1)
    assert nt_xent(a, a).item() < nt_xent(a, a[torch.randperm(8)]).item()
    assert np.isfinite(nt_xent(a, a).item())

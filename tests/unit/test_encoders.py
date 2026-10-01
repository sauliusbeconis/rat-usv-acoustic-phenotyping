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


def test_explorer_audio_is_slowed_sixteenfold(tmp_path):
    import base64
    import io
    import wave

    import soundfile as sf

    from rat_usv.visualization.explorer import SLOWDOWN, _audio

    fs = 214_285
    sf.write(tmp_path / "x.wav", np.zeros(fs, dtype=np.int16), fs, subtype="PCM_16")
    w = wave.open(io.BytesIO(base64.b64decode(_audio(tmp_path / "x.wav", 0.5, 0.6))))
    assert w.getframerate() == fs // SLOWDOWN
    assert abs(w.getnframes() / fs - 0.14) < 0.002  # call + 20 ms each side, original samples

"""AVA-style convolutional VAE (Goffinet et al. 2021, eLife 10:e67855) for 128x128 call images.

Encoder/decoder follow AVA's 7-layer convolutional design and fixed-precision Gaussian
likelihood (precision 10). Difference from AVA: a diagonal (not low-rank) posterior.
Requires the `ml` dependency group (torch).
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn

LATENT = 32
PRECISION = 10.0
BATCH = 128


class VAE(nn.Module):
    def __init__(self, latent: int = LATENT) -> None:
        super().__init__()
        chans = [
            (1, 8, 1),
            (8, 8, 2),
            (8, 16, 1),
            (16, 16, 2),
            (16, 24, 1),
            (24, 24, 2),
            (24, 32, 1),
        ]
        layers: list[nn.Module] = []
        for cin, cout, stride in chans:
            layers += [nn.BatchNorm2d(cin), nn.Conv2d(cin, cout, 3, stride, 1), nn.ReLU()]
        self.encoder = nn.Sequential(*layers, nn.Flatten())  # 32 x 16 x 16 = 8192
        self.enc_fc = nn.Sequential(
            nn.Linear(8192, 1024), nn.ReLU(), nn.Linear(1024, 256), nn.ReLU()
        )
        self.mu = nn.Linear(256, latent)
        self.logvar = nn.Linear(256, latent)
        self.dec_fc = nn.Sequential(
            nn.Linear(latent, 64), nn.ReLU(), nn.Linear(64, 256), nn.ReLU(),
            nn.Linear(256, 1024), nn.ReLU(), nn.Linear(1024, 8192), nn.ReLU(),
        )  # fmt: skip
        up = [(32, 24, 1), (24, 24, 2), (24, 16, 1), (16, 16, 2), (16, 8, 1), (8, 8, 2), (8, 1, 1)]
        dec: list[nn.Module] = []
        for i, (cin, cout, stride) in enumerate(up):
            dec += [
                nn.BatchNorm2d(cin),
                nn.ConvTranspose2d(cin, cout, 3, stride, 1, output_padding=stride - 1),
            ]
            if i < len(up) - 1:
                dec.append(nn.ReLU())
        self.decoder = nn.Sequential(*dec)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.enc_fc(self.encoder(x))
        return self.mu(h), self.logvar(h).clamp(-10, 10)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.dec_fc(z).view(-1, 32, 16, 16))

    def loss(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        recon = self.decode(z)
        rec = 0.5 * PRECISION * ((recon - x) ** 2).flatten(1).sum(1).mean()
        kl = 0.5 * (mu**2 + logvar.exp() - 1 - logvar).sum(1).mean()
        return rec + kl, rec, kl


@dataclass
class TrainResult:
    model: VAE
    history: list[dict[str, float]]


def _tensor(images: NDArray[np.uint8], idx: NDArray[np.int64]) -> torch.Tensor:
    return torch.from_numpy(images[idx].astype(np.float32) / 255.0).unsqueeze(1)


def train_vae(
    images: NDArray[np.uint8], train_idx: NDArray[np.int64], epochs: int, seed: int, device: str
) -> TrainResult:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = VAE().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    # Training images stay on the device as uint8; batches are converted there.
    data = torch.from_numpy(np.ascontiguousarray(images[np.sort(train_idx)])).to(device)
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)
    history = []
    for epoch in range(epochs):
        model.train()
        order = torch.from_numpy(rng.permutation(len(data))).to(device)
        totals = torch.zeros(3, device=device)
        batches = 0
        for start in range(0, len(order), BATCH):
            x = data[order[start : start + BATCH]].unsqueeze(1).float().div_(255.0)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                loss, rec, kl = model.loss(x)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            totals += torch.stack([loss.detach(), rec.detach(), kl.detach()]).float()
            batches += 1
        means = totals.cpu().numpy() / batches
        history.append(dict(zip(("epoch", "loss", "rec", "kl"), [epoch, *means], strict=True)))
    return TrainResult(model, history)


@torch.no_grad()
def embed(
    model: VAE, images: NDArray[np.uint8], device: str, batch: int = 512
) -> NDArray[np.float32]:
    model.eval()
    out = []
    idx = np.arange(len(images))
    for start in range(0, len(idx), batch):
        mu, _ = model.encode(_tensor(images, idx[start : start + batch]).to(device))
        out.append(mu.cpu().numpy())
    return np.concatenate(out).astype(np.float32)


def config() -> dict[str, Any]:
    return {"architecture": "AVA-style conv VAE, diagonal posterior", "latent": LATENT,
            "precision": PRECISION, "optimizer": "Adam lr=1e-3", "batch": BATCH,
            "mixed_precision": "bfloat16 autocast on CUDA"}  # fmt: skip

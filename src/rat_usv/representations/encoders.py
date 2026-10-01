"""M4 rungs 2 and 3: supervised CNN and contrastive (SimCLR-style) encoders.

Both share the VAE's convolutional encoder and a 32-d embedding, so the rungs differ only in
the training objective. Augmentations for the contrastive rung change time position, timing
gaps, gain and noise, never frequency (frequency is the biological signal under study).
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray
from torch import nn

from rat_usv.representations.vae import BATCH, LATENT, VAE


class Encoder(nn.Module):
    """VAE encoder trunk (conv stack + fc) to a LATENT-d embedding."""

    def __init__(self) -> None:
        super().__init__()
        trunk = VAE()
        self.conv, self.fc = trunk.encoder, trunk.enc_fc
        self.out = nn.Linear(256, LATENT)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(self.fc(self.conv(x)))


class AgeCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = Encoder()
        self.head = nn.Linear(LATENT, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        return z, self.head(F.relu(z)).squeeze(1)


class SimCLR(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = Encoder()
        self.projection = nn.Sequential(nn.Linear(LATENT, 64), nn.ReLU(), nn.Linear(64, 64))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        return z, F.normalize(self.projection(z), dim=1)


def augment(x: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    """Time roll (+-12%), up to two time masks, gain 0.7-1.3, Gaussian noise; no frequency change."""
    n, _, _, width = x.shape
    shifts = torch.randint(-15, 16, (n, 1), generator=gen, device=x.device)
    cols = (torch.arange(width, device=x.device).view(1, -1) - shifts) % width
    x = torch.gather(x, 3, cols.view(n, 1, 1, width).expand_as(x))
    for _ in range(2):
        start = torch.randint(0, width - 10, (n,), generator=gen, device=x.device)
        length = torch.randint(0, 10, (n,), generator=gen, device=x.device)
        cols = torch.arange(width, device=x.device).view(1, -1)
        mask = (cols >= start.view(-1, 1)) & (cols < (start + length).view(-1, 1))
        x = x.masked_fill(mask.view(n, 1, 1, width), 0.0)
    gain = 0.7 + 0.6 * torch.rand((n, 1, 1, 1), generator=gen, device=x.device)
    noise = 0.03 * torch.randn(x.shape, generator=gen, device=x.device)
    return (x * gain + noise).clamp(0, 1)


def nt_xent(a: torch.Tensor, b: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    z = torch.cat([a, b])
    sim = z @ z.t() / temperature
    sim.fill_diagonal_(float("-inf"))
    n = a.shape[0]
    target = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)]).to(a.device)
    return F.cross_entropy(sim, target)


@dataclass
class EncoderResult:
    encoder: Encoder
    history: list[dict[str, float]]


def train_encoder(
    kind: str,
    images: NDArray[np.uint8],
    train_idx: NDArray[np.int64],
    ages: NDArray[np.float32],
    epochs: int,
    seed: int,
    device: str,
) -> EncoderResult:
    """kind: 'cnn' (regress pup age from a call) or 'ssl' (contrastive, no labels)."""
    torch.manual_seed(seed)
    gen = torch.Generator(device=device).manual_seed(seed)
    model: nn.Module = AgeCNN() if kind == "cnn" else SimCLR()
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    order_idx = np.sort(train_idx)
    data = torch.from_numpy(np.ascontiguousarray(images[order_idx])).to(device)
    target = torch.from_numpy(ages[order_idx].astype(np.float32)).to(device)
    batch = BATCH * 2 if kind == "ssl" else BATCH
    history = []
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(data), generator=gen, device=device)
        total, batches = torch.zeros((), device=device), 0
        for start in range(0, len(perm) - 1, batch):
            idx = perm[start : start + batch]
            x = data[idx].unsqueeze(1).float().div_(255.0)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                if kind == "cnn":
                    _, pred = model(x)
                    loss = F.smooth_l1_loss(pred.float(), target[idx])
                else:
                    _, pa = model(augment(x, gen))
                    _, pb = model(augment(x, gen))
                    loss = nt_xent(pa.float(), pb.float())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += loss.detach().float()
            batches += 1
        history.append({"epoch": epoch, "loss": float(total.cpu()) / max(batches, 1)})
    return EncoderResult(model.encoder, history)  # type: ignore[arg-type]


@torch.no_grad()
def embed_encoder(
    encoder: Encoder, images: NDArray[np.uint8], device: str, batch: int = 512
) -> NDArray[np.float32]:
    encoder.eval()
    out = []
    for start in range(0, len(images), batch):
        x = (
            torch.from_numpy(np.asarray(images[start : start + batch]))
            .to(device)
            .unsqueeze(1)
            .float()
            / 255.0
        )
        out.append(encoder(x).float().cpu().numpy())
    return np.concatenate(out).astype(np.float32)


def config(kind: str) -> dict[str, Any]:
    base = {
        "backbone": "VAE encoder trunk",
        "embedding": LATENT,
        "optimizer": "Adam lr=1e-3 wd=1e-5",
    }
    if kind == "cnn":
        return base | {
            "objective": "smooth-L1 regression of pup age per call (training litters only)"
        }
    return base | {"objective": "NT-Xent (temperature 0.1), two augmented views",
                   "augmentations": "time roll +-15 px, 2 time masks <10 px, gain 0.7-1.3, noise sd 0.03; no frequency change"}  # fmt: skip

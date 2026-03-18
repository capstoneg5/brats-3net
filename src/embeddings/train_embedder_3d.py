# src/embeddings/train_embedder_3d.py
from __future__ import annotations

import argparse
import glob
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ---------------------------
# Augmentations (3D + multi-channel)
# ---------------------------
def rand_flip(x: torch.Tensor) -> torch.Tensor:
    # x: [C, D, H, W]
    for dim in [1, 2, 3]:
        if torch.rand(1).item() < 0.5:
            x = torch.flip(x, dims=[dim])
    return x

def rand_noise(x: torch.Tensor, sigma: float = 0.05) -> torch.Tensor:
    if torch.rand(1).item() < 0.8:
        x = x + sigma * torch.randn_like(x)
    return x

def rand_intensity(x: torch.Tensor) -> torch.Tensor:
    # per-channel scale/shift
    if torch.rand(1).item() < 0.8:
        scale = (0.9 + 0.2 * torch.rand(x.shape[0], 1, 1, 1, device=x.device))
        shift = (0.05 * torch.randn(x.shape[0], 1, 1, 1, device=x.device))
        x = x * scale + shift
    return x

def normalize_per_channel(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    # z-score per channel
    c = x.shape[0]
    x2 = x.view(c, -1)
    mean = x2.mean(dim=1).view(c, 1, 1, 1)
    std = x2.std(dim=1).view(c, 1, 1, 1)
    return (x - mean) / (std + eps)

def make_view(x: torch.Tensor) -> torch.Tensor:
    x = normalize_per_channel(x)
    x = rand_flip(x)
    x = rand_intensity(x)
    x = rand_noise(x)
    return x


# ---------------------------
# Dataset
# ---------------------------
class CubeDataset(Dataset):
    def __init__(self, cubes_dir: str | Path):
        cubes_dir = Path(cubes_dir)
        self.paths = sorted([Path(p) for p in glob.glob(str(cubes_dir / "*.npy"))])
        if not self.paths:
            raise RuntimeError(f"No .npy cubes found in: {cubes_dir}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        p = self.paths[idx]
        arr = np.load(p)  # expected [C, D, H, W]
        x = torch.from_numpy(arr).float()
        v1 = make_view(x.clone())
        v2 = make_view(x.clone())
        return v1, v2


# ---------------------------
# Model: 3D Encoder + projection head (SimCLR)
# ---------------------------
class Encoder3D(nn.Module):
    def __init__(self, in_ch: int = 4, embed_dim: int = 768):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv3d(in_ch, 32, 3, padding=1), nn.BatchNorm3d(32), nn.ReLU(inplace=True),
            nn.MaxPool3d(2),  # 16
            nn.Conv3d(32, 64, 3, padding=1), nn.BatchNorm3d(64), nn.ReLU(inplace=True),
            nn.MaxPool3d(2),  # 8
            nn.Conv3d(64, 128, 3, padding=1), nn.BatchNorm3d(128), nn.ReLU(inplace=True),
            nn.MaxPool3d(2),  # 4
            nn.Conv3d(128, 256, 3, padding=1), nn.BatchNorm3d(256), nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(256, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, D, H, W]
        h = self.backbone(x)
        h = self.pool(h).flatten(1)
        z = self.fc(h)
        z = F.normalize(z, dim=1)
        return z

class ProjectionHead(nn.Module):
    def __init__(self, embed_dim: int = 768, proj_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, proj_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        p = self.net(z)
        p = F.normalize(p, dim=1)
        return p


# ---------------------------
# InfoNCE loss (SimCLR)
# ---------------------------
def info_nce_loss(p1: torch.Tensor, p2: torch.Tensor, temperature: float = 0.2) -> torch.Tensor:
    # p1, p2: [B, D]
    b = p1.size(0)
    p = torch.cat([p1, p2], dim=0)  # [2B, D]
    sim = (p @ p.T) / temperature   # [2B, 2B]
    mask = torch.eye(2 * b, device=p.device).bool()
    sim = sim.masked_fill(mask, -1e9)

    # positives: i <-> i+B
    pos = torch.cat([torch.arange(b, 2*b), torch.arange(0, b)]).to(p.device)
    logits = sim
    labels = pos
    loss = F.cross_entropy(logits, labels)
    return loss


@dataclass
class TrainCfg:
    cubes_dir: Path
    out_dir: Path
    epochs: int
    batch_size: int
    lr: float
    embed_dim: int
    proj_dim: int
    temperature: float
    device: str


def train(cfg: TrainCfg) -> Path:
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    ds = CubeDataset(cfg.cubes_dir)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, num_workers=2, pin_memory=False, drop_last=True)

    encoder = Encoder3D(in_ch=4, embed_dim=cfg.embed_dim).to(cfg.device)
    projector = ProjectionHead(embed_dim=cfg.embed_dim, proj_dim=cfg.proj_dim).to(cfg.device)

    opt = torch.optim.AdamW(list(encoder.parameters()) + list(projector.parameters()), lr=cfg.lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.device.startswith("cuda")))

    best = 1e9
    best_path = cfg.out_dir / "embedder3d_best.pt"

    for ep in range(1, cfg.epochs + 1):
        encoder.train(); projector.train()
        losses = []

        for v1, v2 in dl:
            v1 = v1.to(cfg.device, non_blocking=True)
            v2 = v2.to(cfg.device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(cfg.device.startswith("cuda"))):
                z1 = encoder(v1)
                z2 = encoder(v2)
                p1 = projector(z1)
                p2 = projector(z2)
                loss = info_nce_loss(p1, p2, temperature=cfg.temperature)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            losses.append(loss.item())

        avg = float(np.mean(losses)) if losses else 0.0
        print(f"Epoch {ep:02d} | loss={avg:.4f}")

        # lower is better
        if avg < best:
            best = avg
            torch.save(
                {
                    "encoder": encoder.state_dict(),
                    "projector": projector.state_dict(),
                    "embed_dim": cfg.embed_dim,
                },
                best_path,
            )
            print(f"✅ Saved BEST -> {best_path} (loss={best:.4f})")

    print("✅ Training complete.")
    return best_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cubes_dir", required=True)
    ap.add_argument("--out_dir", default="artifacts/checkpoints")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--embed_dim", type=int, default=768)
    ap.add_argument("--proj_dim", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    args = ap.parse_args()

    cfg = TrainCfg(
        cubes_dir=Path(args.cubes_dir),
        out_dir=Path(args.out_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        embed_dim=args.embed_dim,
        proj_dim=args.proj_dim,
        temperature=args.temperature,
        device=args.device,
    )
    train(cfg)


if __name__ == "__main__":
    main()
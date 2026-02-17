# brats-3net/models/segmentation/train_unet3d.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# ✅ Make brats-3net importable when you run from repo root:
# python brats-3net/models/segmentation/train_unet3d.py ...
sys.path.append("brats-3net")

# ✅ Your actual module contains these classes (you verified)
from models.segmentation.unet3d import SegmentationTrainer, UNet


# -------------------------
# Dataset: train.pt / val.pt
# -------------------------
class PtDataset(Dataset):
    """
    Expects torch.save dict with keys:
      X: (N,4,128,128,128) float32
      Y: (N,128,128,128) uint8  (labels 0..3)
      metas: optional list[dict]
    """

    def __init__(self, pt_path: str):
        d = torch.load(pt_path, map_location="cpu")
        self.X = d["X"]
        self.Y = d["Y"]
        self.metas = d.get("metas", None)

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def __getitem__(self, idx: int):
        x = self.X[idx]  # (4,D,H,W) float32
        y = self.Y[idx].long()  # (D,H,W) labels {0,1,2,4}

        # Remap BraTS label 4 -> class index 3 for CrossEntropy with num_classes=4
        # (classes become 0,1,2,3)
        y = torch.where(y == 4, 3, y)

        return x, y


def dice_score(pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> float:
    """
    pred: (B,D,H,W) long
    target: (B,D,H,W) long
    """
    eps = 1e-6
    d = 0.0
    for c in range(1, num_classes):  # skip background
        p = (pred == c).float()
        t = (target == c).float()
        inter = (p * t).sum()
        denom = p.sum() + t.sum() + eps
        d += (2.0 * inter / denom).item()
    return d / max(1, (num_classes - 1))


# -------------------------
# Try to use SegmentationTrainer if compatible
# -------------------------
def try_trainer(args) -> bool:
    """
    Returns True if trainer was used successfully, else False.
    This avoids you guessing trainer constructor signature.
    """
    try:
        trainer = SegmentationTrainer(
            train_pt=args.train_pt,
            val_pt=args.val_pt,
            out_dir=args.out_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            num_classes=args.num_classes,
            device=args.device,
        )
        trainer.train()
        return True
    except TypeError:
        # signature mismatch -> fallback to manual loop
        return False
    except AttributeError:
        # trainer missing .train() -> fallback
        return False


# -------------------------
# Manual training loop (always works)
# -------------------------
def train_manual(args) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_path = out_dir / "unet3d_best.pt"
    last_path = out_dir / "unet3d_last.pt"

    train_ds = PtDataset(args.train_pt)
    val_ds = PtDataset(args.val_pt)

    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=1,
    )

    device = torch.device(args.device)

    # ✅ Use UNet (this exists in your file)
    # Your UNet signature might differ; this is the common pattern:
    model = UNet(
        spatial_dims=3,
        in_channels=4,
        out_channels=args.num_classes,
        channels=(8, 16, 32, 64, 128),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)

    ce = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_dice = -1.0

    for epoch in range(1, args.epochs + 1):
        # ---- train ----
        model.train()
        train_loss = 0.0

        for x, y in train_dl:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            logits = model(x)              # (B,C,D,H,W)
            loss = ce(logits, y)           # y: (B,D,H,W)
            loss.backward()
            opt.step()
            train_loss += loss.item()

        train_loss /= max(1, len(train_dl))

        # ---- val ----
        model.eval()
        val_loss = 0.0
        val_dice = 0.0

        with torch.no_grad():
            for x, y in val_dl:
                x = x.to(device)
                y = y.to(device)
                logits = model(x)
                loss = ce(logits, y)
                val_loss += loss.item()

                pred = torch.argmax(logits, dim=1)  # (B,D,H,W)
                val_dice += dice_score(pred, y, args.num_classes)

        val_loss /= max(1, len(val_dl))
        val_dice /= max(1, len(val_dl))

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_dice={val_dice:.4f}"
        )

        # save last
        torch.save(
            {
                "model": model.state_dict(),
                "epoch": epoch,
                "val_dice": val_dice,
            },
            last_path,
        )

        # save best
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "best_dice": best_dice,
                },
                best_path,
            )
            print(f"✅ Saved BEST -> {best_path} (dice={best_dice:.4f})")

    print("✅ Training complete.")
    print("Best checkpoint:", best_path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_pt", required=True)
    ap.add_argument("--val_pt", required=True)

    ap.add_argument("--out_dir", default="brats-3net/artifacts/checkpoints")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--num_classes", type=int, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    # If you want to force manual loop:
    ap.add_argument("--force_manual", action="store_true")

    args = ap.parse_args()

    # ✅ First try your built-in trainer (if it matches), else fallback
    used_trainer = False
    if not args.force_manual:
        used_trainer = try_trainer(args)

    if used_trainer:
        print("✅ Trained using SegmentationTrainer")
    else:
        print("ℹ️ SegmentationTrainer not used (signature mismatch). Using manual loop with UNet.")
        train_manual(args)


if __name__ == "__main__":
    main()
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

# Make "models..." importable
import sys
sys.path.append("brats-3net")
from models.segmentation.unet3d import UNet  # your UNet class


def load_model(ckpt_path: str, num_classes: int, device: torch.device) -> torch.nn.Module:
    ckpt = torch.load(ckpt_path, map_location="cpu")

    # IMPORTANT: must match training architecture
    model = UNet(
        spatial_dims=3,
        in_channels=4,
        out_channels=num_classes,
        channels=(8, 16, 32, 64, 128),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)

    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    return model


def maybe_map_back_to_brats(labels_0_3: np.ndarray) -> np.ndarray:
    """
    Training uses labels {0,1,2,3} where 3 means original BraTS label 4.
    Convert back so output aligns with BraTS convention: {0,1,2,4}
    """
    out = labels_0_3.copy()
    out[out == 3] = 4
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", required=True, help="Path to train.pt or val.pt etc")
    ap.add_argument("--processed_root", required=True, help="Root where patient folders exist")
    ap.add_argument("--split", required=True, choices=["train", "val"], help="Split folder name under processed_root")
    ap.add_argument("--ckpt", required=True, help="Path to unet3d_best.pt")
    ap.add_argument("--num_classes", type=int, default=4, help="4 for labels 0..3 (3==ET)")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    ap.add_argument("--write_brats_labels", action="store_true", help="Also save pred_seg_brats.npy with 3->4 mapping")
    args = ap.parse_args()

    device = torch.device(args.device)

    d: Dict = torch.load(args.pt, map_location="cpu")
    X = d["X"]  # (N,4,128,128,128) float32
    metas: List[Dict] = d.get("metas", [])

    if not metas:
        raise RuntimeError("No metas found in pt file. Need metas with patient_id.")

    split_dir = Path(args.processed_root) / args.split
    if not split_dir.exists():
        raise FileNotFoundError(f"Processed split not found: {split_dir}")

    model = load_model(args.ckpt, args.num_classes, device)

    written = 0
    with torch.no_grad():
        for i in range(int(X.shape[0])):
            x = X[i].unsqueeze(0).to(device)  # (1,4,D,H,W)

            logits = model(x)                 # (1,C,D,H,W)
            pred = torch.argmax(logits, dim=1).squeeze(0).to("cpu")  # (D,H,W)

            patient_id = metas[i].get("patient_id")
            if not patient_id:
                raise RuntimeError(f"Missing patient_id in metas[{i}]")

            out_dir = split_dir / patient_id
            out_dir.mkdir(parents=True, exist_ok=True)

            pred_np = pred.numpy().astype("uint8")
            np.save(out_dir / "pred_seg.npy", pred_np)

            if args.write_brats_labels:
                brats = maybe_map_back_to_brats(pred_np)
                np.save(out_dir / "pred_seg_brats.npy", brats.astype("uint8"))

            written += 1

            if written % 10 == 0:
                print(f"✅ wrote {written}/{X.shape[0]}")

    print(f"\n✅ Done. Wrote pred_seg.npy for {written} patients into: {split_dir}")


if __name__ == "__main__":
    main()
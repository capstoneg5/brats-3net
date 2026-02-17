"""
Build PyTorch .pt datasets from *processed* BraTS .npy folders.

Input (your current structure):
  brats-3net/artifacts/processed/<split>/<patient>/
      t1.npy, t1ce.npy, t2.npy, flair.npy, (optional) seg.npy, metadata.json

Output:
  brats-3net/artifacts/datasets/<split>.pt
  brats-3net/artifacts/datasets/<split>_index.json

Each sample:
  x: float32 tensor (4, D, H, W)
  y: uint8 tensor  (D, H, W)  (if seg exists; else zeros)
  meta: dict (patient_id, paths, etc.)

Run:
  python brats-3net/scripts/build_train_pt_from_processed.py --split train
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm


MODS = ["t1", "t1ce", "t2", "flair"]


@dataclass
class SampleIndex:
    patient_id: str
    x_paths: Dict[str, str]
    seg_path: Optional[str]
    shape: Tuple[int, int, int]


def load_patient(patient_dir: Path) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    # load modalities -> (4, D, H, W)
    vols = []
    for m in MODS:
        p = patient_dir / f"{m}.npy"
        if not p.exists():
            raise FileNotFoundError(f"Missing {p}")
        v = np.load(p).astype("float32")  # (D,H,W)
        vols.append(v)

    x = np.stack(vols, axis=0)  # (4,D,H,W)
    x_t = torch.from_numpy(x)

    # seg optional -> (D,H,W)
    seg_p = patient_dir / "seg.npy"
    if seg_p.exists():
        y = np.load(seg_p)
        # ensure integer labels
        if y.dtype != np.uint8:
            y = y.astype("uint8")
    else:
        # create empty seg
        y = np.zeros(vols[0].shape, dtype="uint8")

    y_t = torch.from_numpy(y)

    # metadata optional
    meta_p = patient_dir / "metadata.json"
    meta = {}
    if meta_p.exists():
        try:
            meta = json.loads(meta_p.read_text())
        except Exception:
            meta = {}

    meta.update(
        {
            "patient_id": patient_dir.name,
            "has_seg": bool(seg_p.exists()),
        }
    )

    return x_t, y_t, meta


def build_split(
    processed_root: Path,
    split: str,
    out_dir: Path,
    max_patients: Optional[int] = None,
) -> None:
    split_dir = processed_root / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Processed split not found: {split_dir}")

    patient_dirs = sorted([p for p in split_dir.iterdir() if p.is_dir()])
    if max_patients:
        patient_dirs = patient_dirs[: max_patients]

    if not patient_dirs:
        raise RuntimeError(f"No patient folders found in {split_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)

    xs: List[torch.Tensor] = []
    ys: List[torch.Tensor] = []
    metas: List[Dict] = []
    index: List[SampleIndex] = []

    for pdir in tqdm(patient_dirs, desc=f"Building {split}.pt"):
        x, y, meta = load_patient(pdir)

        # sanity checks
        if x.ndim != 4 or x.shape[0] != 4:
            raise RuntimeError(f"Bad x shape for {pdir.name}: {tuple(x.shape)}")
        if y.shape != x.shape[1:]:
            raise RuntimeError(f"Bad y shape for {pdir.name}: y={tuple(y.shape)} x={tuple(x.shape)}")

        xs.append(x)
        ys.append(y)
        metas.append(meta)

        x_paths = {m: str((pdir / f"{m}.npy").as_posix()) for m in MODS}
        seg_path = str((pdir / "seg.npy").as_posix()) if (pdir / "seg.npy").exists() else None

        index.append(
            SampleIndex(
                patient_id=pdir.name,
                x_paths=x_paths,
                seg_path=seg_path,
                shape=tuple(int(i) for i in x.shape[1:]),
            )
        )

    # Stack tensors -> big tensors
    X = torch.stack(xs, dim=0)  # (N,4,D,H,W)
    Y = torch.stack(ys, dim=0)  # (N,D,H,W)

    # Save .pt
    pt_path = out_dir / f"{split}.pt"
    torch.save(
        {
            "X": X,          # float32
            "Y": Y,          # uint8
            "metas": metas,  # list[dict]
        },
        pt_path,
    )

    # Save index json
    idx_path = out_dir / f"{split}_index.json"
    idx_payload = [
        {
            "patient_id": it.patient_id,
            "x_paths": it.x_paths,
            "seg_path": it.seg_path,
            "shape": list(it.shape),
        }
        for it in index
    ]
    idx_path.write_text(json.dumps(idx_payload, indent=2))

    print(f"\n✅ Saved: {pt_path}")
    print(f"✅ Saved: {idx_path}")
    print(f"✅ X shape: {tuple(X.shape)}  dtype={X.dtype}")
    print(f"✅ Y shape: {tuple(Y.shape)}  dtype={Y.dtype}")
    print(f"✅ patients: {len(index)}")


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--processed_root", type=Path, default=Path("brats-3net/artifacts/processed"))
    ap.add_argument("--out_dir", type=Path, default=Path("brats-3net/artifacts/datasets"))
    ap.add_argument("--split", choices=["train", "val", "test"], required=True)
    ap.add_argument("--max_patients", type=int, default=None)
    args = ap.parse_args()

    build_split(
        processed_root=args.processed_root,
        split=args.split,
        out_dir=args.out_dir,
        max_patients=args.max_patients,
    )


if __name__ == "__main__":
    main()
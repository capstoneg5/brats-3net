"""
Minimal working BraTS preprocessing
----------------------------------
Produces:
  • normalized volumes per modality
  • resampled / cropped / padded tensors
  • metadata.json per patient

Dependencies:
  pip install nibabel numpy scipy tqdm
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple, Literal, cast, Any

import nibabel as nib
import numpy as np
import vol
from scipy.ndimage import zoom
from tqdm import tqdm


# ---------------------------------------------------------
# Utility functions
# ---------------------------------------------------------

def load_nifti(path: Path) -> np.ndarray:
    """Load NIfTI as float32 numpy array."""
    img = nib.load(str(path))
    return img.get_fdata().astype("float32")


def pick_nii(patient_dir: Path, suffix: str) -> Path:
    p1 = patient_dir / f"{patient_dir.name}_{suffix}.nii.gz"
    if p1.exists():
        return p1
    p2 = patient_dir / f"{patient_dir.name}_{suffix}.nii"
    if p2.exists():
        return p2
    raise FileNotFoundError(f"Missing file for {patient_dir.name}_{suffix} (.nii/.nii.gz)")

def zscore_normalize(volume: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Z-score normalize using non-zero (brain) region."""
    if mask is None:
        mask = volume > 0

    vox = volume[mask]
    if vox.size == 0:
        return volume

    mean = float(vox.mean())
    std = float(vox.std() + 1e-8)

    out = (volume - mean) / std
    out[~mask] = 0
    return out.astype("float32")


def resample_to_shape(
    volume: np.ndarray,
    target_shape: Tuple[int, int, int],
    order: Literal[0,1,2,3,4,5] = 1,
) -> np.ndarray:
    factors = tuple(float(t) / float(s) for t, s in zip(target_shape, volume.shape))
    return zoom(volume, zoom=factors, order=order).astype("float32")


def center_crop_or_pad(volume: np.ndarray, target_shape: Tuple[int, int, int]) -> np.ndarray:
    """Center crop or zero-pad to exact shape."""
    out = np.zeros(target_shape, dtype="float32")

    src = np.array(volume.shape)
    tgt = np.array(target_shape)

    min_shape = np.minimum(src, tgt)

    src_start = ((src - min_shape) // 2).astype(int)
    tgt_start = ((tgt - min_shape) // 2).astype(int)

    out[
        tgt_start[0]:tgt_start[0] + min_shape[0],
        tgt_start[1]:tgt_start[1] + min_shape[1],
        tgt_start[2]:tgt_start[2] + min_shape[2],
    ] = volume[
        src_start[0]:src_start[0] + min_shape[0],
        src_start[1]:src_start[1] + min_shape[1],
        src_start[2]:src_start[2] + min_shape[2],
    ]

    return out


# ---------------------------------------------------------
# Main preprocessing per patient
# ---------------------------------------------------------

def preprocess_patient(
    patient_dir: Path,
    out_dir: Path,
    target_shape: Tuple[int, int, int] = (128, 128, 128),
) -> None:
    """Preprocess one BraTS patient."""

    modalities = {
        "t1": pick_nii(patient_dir, "t1"),
        "t1ce": pick_nii(patient_dir, "t1ce"),
        "t2": pick_nii(patient_dir, "t2"),
        "flair": pick_nii(patient_dir, "flair"),
    }

    try:
        seg_path = pick_nii(patient_dir, "seg")
    except FileNotFoundError:
        seg_path = None

    patient_out = out_dir / patient_dir.name
    patient_out.mkdir(parents=True, exist_ok=True)

    metadata: Dict[str, Any] = {"modalities": {}, "target_shape": list(target_shape)}

    for name, path in modalities.items():
        vol = load_nifti(path)
        vol = zscore_normalize(vol)
        vol = resample_to_shape(vol, target_shape)
        vol = center_crop_or_pad(vol, target_shape)

        np.save(patient_out / f"{name}.npy", vol)

        orig_shape = list(vol.shape)

        metadata["modalities"][name] = {
            "original_shape": orig_shape,
            "final_shape": list(vol.shape),
        }

    # segmentation mask (optional for val/test)
    if seg_path is not None and seg_path.exists():
        seg = load_nifti(seg_path)
        seg = resample_to_shape(seg, target_shape, order=0)  # nearest for labels
        seg = center_crop_or_pad(seg, target_shape)
        np.save(patient_out / "seg.npy", seg.astype("uint8"))
        metadata["has_seg"] = True
    else:
        metadata["has_seg"] = False

    with (patient_out / "metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2)


# ---------------------------------------------------------
# Split-level preprocessing
# ---------------------------------------------------------

def preprocess_split(
    data_root: Path,
    out_dir: Path,
    max_patients: int | None = None,
) -> None:
    """Run preprocessing for all patients in a split."""

    patients = sorted([p for p in data_root.iterdir() if p.is_dir()])

    if max_patients:
        patients = patients[:max_patients]

    print(f"Found {len(patients)} patients")

    for p in tqdm(patients, desc="Preprocessing"):
        preprocess_patient(p, out_dir)

    print("✅ Preprocessing complete")


# ---------------------------------------------------------
# CLI usage
# ---------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--max_patients", type=int, default=None)

    args = parser.parse_args()

    preprocess_split(args.data_root, args.out_dir, args.max_patients)

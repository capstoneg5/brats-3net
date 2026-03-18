from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
from scipy.ndimage import label as cc_label


@dataclass
class LesionRecord:
    lesion_id: str
    patient_id: str
    lesion_type: str  # WT / TC / ET (for bookkeeping; cube is usually WT component)
    bbox: List[int]   # [z0,z1,y0,y1,x0,x1]
    centroid_zyx: List[float]

    wt_vox: int
    tc_vox: int
    et_vox: int
    tumor_volume_voxels: int

    tc_pct: float
    et_pct: float

    cube_shape: List[int]
    modalities: List[str]


def _bbox_from_mask(mask: np.ndarray) -> Tuple[int, int, int, int, int, int]:
    """Returns z0,z1,y0,y1,x0,x1 inclusive-exclusive bbox."""
    idx = np.argwhere(mask)
    z0, y0, x0 = idx.min(axis=0)
    z1, y1, x1 = idx.max(axis=0) + 1
    return int(z0), int(z1), int(y0), int(y1), int(x0), int(x1)


def _centroid_zyx(mask: np.ndarray) -> Tuple[float, float, float]:
    idx = np.argwhere(mask)
    c = idx.mean(axis=0)
    return float(c[0]), float(c[1]), float(c[2])


def _crop_with_margin(
    arr: np.ndarray,
    bbox: Tuple[int, int, int, int, int, int],
    margin: int,
    target_shape: Tuple[int, int, int],
) -> np.ndarray:
    """
    Crop arr to bbox±margin and then center-pad/crop to target_shape.
    arr shape is (Z,Y,X).
    """
    z0, z1, y0, y1, x0, x1 = bbox
    z0 = max(0, z0 - margin)
    y0 = max(0, y0 - margin)
    x0 = max(0, x0 - margin)
    z1 = min(arr.shape[0], z1 + margin)
    y1 = min(arr.shape[1], y1 + margin)
    x1 = min(arr.shape[2], x1 + margin)

    cropped = arr[z0:z1, y0:y1, x0:x1]

    # center crop/pad to target_shape
    out = np.zeros(target_shape, dtype=arr.dtype)
    src = np.array(cropped.shape)
    tgt = np.array(target_shape)
    min_shape = np.minimum(src, tgt)

    src_start = ((src - min_shape) // 2).astype(int)
    tgt_start = ((tgt - min_shape) // 2).astype(int)

    out[
        tgt_start[0]:tgt_start[0] + min_shape[0],
        tgt_start[1]:tgt_start[1] + min_shape[1],
        tgt_start[2]:tgt_start[2] + min_shape[2],
    ] = cropped[
        src_start[0]:src_start[0] + min_shape[0],
        src_start[1]:src_start[1] + min_shape[1],
        src_start[2]:src_start[2] + min_shape[2],
    ]
    return out


def extract_lesions_for_split(
    processed_root: Path,
    split: str,
    out_dir: Path,
    use_gt_masks: bool,
    cube_shape: Tuple[int, int, int] = (32, 32, 32),
    margin: int = 4,
    max_patients: Optional[int] = None,
) -> Path:
    """
    Reads:
      artifacts/processed/<split>/<patient>/{t1,t1ce,t2,flair,seg}.npy

    Writes:
      out_dir/cubes/<lesion_id>.npy   (shape = (C,Z,Y,X) where C=4 modalities)
      out_dir/lesions.jsonl          (one record per lesion)
    """
    split_dir = processed_root / split
    processed_patients = [p for p in split_dir.iterdir() if p.is_dir()]
    # ✅ filter patients having seg.npy if GT masks are requested
    if use_gt_masks:
        patients = [p for p in processed_patients if (p / "seg.npy").exists()]
    else:
        # if using predicted masks, you would check pred mask path instead
        patients = processed_patients

    if max_patients:
        patients = patients[:max_patients]

    print(f"Total processed: {len(processed_patients)}")
    print(f"Using patients: {len(patients)}")

    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)
    cubes_dir = out_dir / "cubes"
    cubes_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "lesions.jsonl"

    patients = sorted([p for p in split_dir.iterdir() if p.is_dir()])
    if max_patients is not None:
        patients = patients[:max_patients]

    lesion_idx = 0
    written = 0

    with jsonl_path.open("w", encoding="utf-8") as f:
        for pdir in patients:
            patient_id = pdir.name

            seg_path = pdir / "seg.npy"
            if not seg_path.exists():
                # skip if no GT seg (e.g., test split)
                continue

            seg = np.load(seg_path).astype(np.int16)  # (Z,Y,X)

            # WT / TC / ET masks
            wt_mask = (seg > 0)                    # 1,2,4
            tc_mask = (seg == 1) | (seg == 4)      # 1,4
            et_mask = (seg == 4)                   # 4

            # connected components on WT (each component = one lesion)
            labeled, ncomp = cc_label(wt_mask.astype(np.uint8))
            if ncomp == 0:
                continue

            # load modalities once
            t1 = np.load(pdir / "t1.npy").astype(np.float32)
            t1ce = np.load(pdir / "t1ce.npy").astype(np.float32)
            t2 = np.load(pdir / "t2.npy").astype(np.float32)
            flair = np.load(pdir / "flair.npy").astype(np.float32)

            for comp_id in range(1, ncomp + 1):
                comp_mask = (labeled == comp_id)
                vox_wt = int(comp_mask.sum())
                if vox_wt < 50:  # ignore tiny noise components
                    continue

                bbox = _bbox_from_mask(comp_mask)
                cz, cy, cx = _centroid_zyx(comp_mask)

                # within this WT component, compute TC/ET voxels
                tc_vox = int((tc_mask & comp_mask).sum())
                et_vox = int((et_mask & comp_mask).sum())
                tumor_vox = vox_wt

                tc_pct = float(tc_vox / (tumor_vox + 1e-8))
                et_pct = float(et_vox / (tumor_vox + 1e-8))

                # crop each modality to the lesion bbox
                c_t1 = _crop_with_margin(t1, bbox, margin, cube_shape)
                c_t1ce = _crop_with_margin(t1ce, bbox, margin, cube_shape)
                c_t2 = _crop_with_margin(t2, bbox, margin, cube_shape)
                c_flair = _crop_with_margin(flair, bbox, margin, cube_shape)

                cube = np.stack([c_t1, c_t1ce, c_t2, c_flair], axis=0)  # (4,Z,Y,X)

                lesion_idx += 1
                lesion_id = f"{patient_id}_lesion{lesion_idx}"

                np.save(cubes_dir / f"{lesion_id}.npy", cube)

                rec = LesionRecord(
                    lesion_id=lesion_id,
                    patient_id=patient_id,
                    lesion_type="WT_component",
                    bbox=list(bbox),
                    centroid_zyx=[cz, cy, cx],
                    wt_vox=vox_wt,
                    tc_vox=tc_vox,
                    et_vox=et_vox,
                    tumor_volume_voxels=tumor_vox,
                    tc_pct=tc_pct,
                    et_pct=et_pct,
                    cube_shape=list(cube.shape),
                    modalities=["t1", "t1ce", "t2", "flair"],
                )

                f.write(json.dumps(asdict(rec)) + "\n")
                written += 1

    print(f"✅ Extracted lesions: {written}")
    print(f"✅ Cubes saved to: {cubes_dir}")
    print(f"✅ Metadata saved to: {jsonl_path}")
    return jsonl_path
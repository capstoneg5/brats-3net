# src/rag/multimodal_router.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

import numpy as np
import torch

# Your model wrapper
from models.segmentation.unet3d import UNet3DSegmenter

try:
    import nibabel as nib
except Exception:
    nib = None  # handled later

# Overlay rendering (headless safe)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass
class RouteResult:
    kind: str  # "nifti" | "text" | "unknown"
    rag_text: str
    structured: Dict[str, Any]
    overlay_paths: List[str]


# ---------------------------
# File / modality utilities
# ---------------------------
def _is_nifti(p: Path) -> bool:
    name = p.name.lower()
    return name.endswith(".nii") or name.endswith(".nii.gz")


def _guess_role(p: Path) -> str:
    """
    Best-effort role inference from filename.
    Typical BraTS: t1, t1ce, t2, flair, seg
    """
    n = p.name.lower()
    if "seg" in n or "mask" in n:
        return "seg"
    if "t1ce" in n or "t1c" in n or "t1gd" in n:
        return "t1ce"
    if "flair" in n:
        return "flair"
    if "t2" in n:
        return "t2"
    if "t1" in n:
        return "t1"
    return "image"


def _find_best_image(mods: Dict[str, Path]) -> Optional[Path]:
    for k in ("t1ce", "flair", "t2", "t1", "image"):
        if k in mods:
            return mods[k]
    # fallback: any non-seg nifti
    for k, p in mods.items():
        if k != "seg":
            return p
    return None


def _find_seg(mods: Dict[str, Path]) -> Optional[Path]:
    return mods.get("seg")


# ---------------------------
# NIfTI loading + spacing
# ---------------------------
def _as_3d(x: np.ndarray) -> np.ndarray:
    """
    Ensure we have a 3D volume.
    If 4D, take first volume (common in some exports).
    """
    if x.ndim == 3:
        return x
    if x.ndim == 4:
        return x[..., 0]
    raise ValueError(f"Expected 3D/4D NIfTI array, got shape {x.shape}")


def _load_nifti_with_spacing(path: Path) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    if nib is None:
        raise RuntimeError("nibabel is not installed. Install it: pip install nibabel")

    # 👇 Cast makes Pylance happy
    img = cast("nib.Nifti1Image", nib.load(str(path)))

    data = np.asarray(img.get_fdata(), dtype=np.float32)
    data = _as_3d(data)

    aff = np.asarray(img.affine, dtype=np.float64)
    sx = float(np.linalg.norm(aff[:3, 0]))
    sy = float(np.linalg.norm(aff[:3, 1]))
    sz = float(np.linalg.norm(aff[:3, 2]))
    zooms3: Tuple[float, float, float] = (sx, sy, sz)

    return data, zooms3


# ---------------------------
# Normalization + overlays
# ---------------------------
def _normalize_img(x: np.ndarray) -> np.ndarray:
    x = np.nan_to_num(x)
    lo, hi = np.percentile(x, (1, 99))
    if hi <= lo:
        return np.zeros_like(x, dtype=np.float32)
    y = (x - lo) / (hi - lo)
    return np.clip(y, 0.0, 1.0).astype(np.float32)


def _zscore(vol: np.ndarray) -> np.ndarray:
    vol = np.nan_to_num(vol)
    m = float(vol.mean())
    s = float(vol.std()) + 1e-6
    return ((vol - m) / s).astype(np.float32)


def _render_overlay_slices(
    image: np.ndarray,
    mask: Optional[np.ndarray],
    out_dir: Path,
    prefix: str,
) -> List[str]:
    out_dir.mkdir(parents=True, exist_ok=True)

    img_n = _normalize_img(image)
    ci, cj, ck = [s // 2 for s in img_n.shape[:3]]

    slices = {
        "axial": (img_n[:, :, ck], None if mask is None else mask[:, :, ck]),
        "coronal": (img_n[:, cj, :], None if mask is None else mask[:, cj, :]),
        "sagittal": (img_n[ci, :, :], None if mask is None else mask[ci, :, :]),
    }

    saved: List[str] = []
    for name, (img2d, m2d) in slices.items():
        fig = plt.figure(figsize=(5, 5), dpi=150)
        ax = fig.add_subplot(1, 1, 1)
        ax.imshow(np.rot90(img2d), cmap="gray")
        ax.axis("off")

        if m2d is not None:
            overlay = np.rot90((m2d > 0).astype(np.float32))
            ax.imshow(overlay, alpha=0.35)

        fp = out_dir / f"{prefix}_{name}.png"
        fig.tight_layout(pad=0)
        fig.savefig(fp, bbox_inches="tight", pad_inches=0)
        plt.close(fig)
        saved.append(str(fp))

    return saved


# ---------------------------
# Lesion statistics
# ---------------------------
def _mask_stats(mask: np.ndarray, zooms_xyz: Tuple[float, float, float]) -> Dict[str, Any]:
    z_x, z_y, z_z = zooms_xyz
    voxel_volume_mm3 = z_x * z_y * z_z

    lesion = mask > 0
    voxels = int(np.count_nonzero(lesion))
    volume_mm3 = voxels * voxel_volume_mm3
    volume_ml = volume_mm3 / 1000.0

    if voxels == 0:
        return {
            "lesion_voxels": 0,
            "voxel_volume_mm3": float(voxel_volume_mm3),
            "lesion_volume_mm3": 0.0,
            "lesion_volume_ml": 0.0,
            "bbox_ijk": None,
            "centroid_ijk": None,
            "approx_max_diameter_mm": None,
            "labels_present": [],
        }

    coords = np.argwhere(lesion)
    mins = coords.min(axis=0).tolist()
    maxs = coords.max(axis=0).tolist()
    centroid = coords.mean(axis=0).tolist()

    di = (maxs[0] - mins[0] + 1) * z_x
    dj = (maxs[1] - mins[1] + 1) * z_y
    dk = (maxs[2] - mins[2] + 1) * z_z
    approx_max_diameter_mm = float(np.sqrt(di * di + dj * dj + dk * dk))

    return {
        "lesion_voxels": voxels,
        "voxel_volume_mm3": float(voxel_volume_mm3),
        "lesion_volume_mm3": float(volume_mm3),
        "lesion_volume_ml": float(volume_ml),
        "bbox_ijk": {"min": mins, "max": maxs},
        "centroid_ijk": [float(c) for c in centroid],
        "approx_max_diameter_mm": approx_max_diameter_mm,
        "labels_present": sorted([int(v) for v in np.unique(mask) if v != 0]),
    }


def _make_rag_text(structured: Dict[str, Any]) -> str:
    if not structured.get("has_lesion"):
        return (
            "Imaging-derived stats: No lesion voxels detected in the mask. "
            "If unexpected, confirm mask alignment / correct case."
        )

    vol_ml = structured["lesion_stats"]["lesion_volume_ml"]
    diam_mm = structured["lesion_stats"].get("approx_max_diameter_mm")
    bbox = structured["lesion_stats"].get("bbox_ijk")

    return (
        "Imaging-derived stats (educational): "
        f"Lesion volume ≈ {vol_ml:.2f} mL. "
        f"Approx max extent (bbox diagonal) ≈ {float(diam_mm):.1f} mm. "
        f"Bounding box (voxel coords) min={bbox['min']} max={bbox['max']}. "
        "Computed from segmentation + voxel spacing; not a diagnosis."
    )


# ---------------------------
# Pred mask inference
# ---------------------------
def _load_state_dict_any(ckpt: Any) -> Dict[str, Any]:
    """
    Accepts common checkpoint formats and returns a state_dict-like dict.
    """
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        return ckpt["state_dict"]
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        return ckpt["model_state_dict"]
    if isinstance(ckpt, dict):
        return ckpt
    raise ValueError("Unexpected checkpoint format (expected dict-like checkpoint).")


def _strip_module_prefix(state: Dict[str, Any]) -> Dict[str, Any]:
    fixed: Dict[str, Any] = {}
    for k, v in state.items():
        fixed[k.replace("module.", "")] = v
    return fixed


def _run_segmentation_inference(
    image_paths: Dict[str, Path],
    checkpoint: Optional[Path],
) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    """
    Runs UNet3D on BraTS-style 4-channel MRI:
      channels = [t1, t1ce, t2, flair]
    Returns:
      pred_mask: (D,H,W) uint8 labels
      zooms3: (sx,sy,sz) mm spacing (from the chosen reference modality)
    """
    required = ["t1", "t1ce", "t2", "flair"]
    missing = [k for k in required if k not in image_paths]
    if missing:
        raise ValueError(f"Missing modalities for pred_mask: {missing}. Found: {list(image_paths.keys())}")

    if checkpoint is None:
        raise ValueError("mask_source='pred_mask' requires a checkpoint path.")

    # Load 4 modalities
    t1, zooms3 = _load_nifti_with_spacing(image_paths["t1"])
    t1ce, z2 = _load_nifti_with_spacing(image_paths["t1ce"])
    t2, z3 = _load_nifti_with_spacing(image_paths["t2"])
    flair, z4 = _load_nifti_with_spacing(image_paths["flair"])

    # Hard shape check (production must fail fast)
    shape = t1.shape
    if t1ce.shape != shape or t2.shape != shape or flair.shape != shape:
        raise ValueError(
            f"Modality shape mismatch: "
            f"t1={t1.shape}, t1ce={t1ce.shape}, t2={t2.shape}, flair={flair.shape}"
        )

    # Normalize channels
    x = np.stack([_zscore(t1), _zscore(t1ce), _zscore(t2), _zscore(flair)], axis=0).astype(np.float32)  # (C,D,H,W)

    # Build model wrapper
    seg = UNet3DSegmenter(in_channels=4, out_channels=4)

    ckpt = torch.load(str(checkpoint), map_location=seg.device)
    state = _strip_module_prefix(_load_state_dict_any(ckpt))

    # strict=False to tolerate minor wrapper naming differences
    seg.model.load_state_dict(state, strict=False)
    seg.model.eval()

    # inference_mode is faster + safer than no_grad
    with torch.inference_mode():
        pred_mask, _probs = seg.predict(x)

    pred_mask_u8 = np.asarray(pred_mask, dtype=np.uint8)

    # Use spacing from reference modality (t1). (We keep others only for mismatch warnings)
    zooms3_typed: Tuple[float, float, float] = (float(zooms3[0]), float(zooms3[1]), float(zooms3[2]))
    return pred_mask_u8, zooms3_typed


# ---------------------------
# Main router
# ---------------------------
def route_for_chat(
    uploaded_paths: List[Path],
    out_dir: Path,
    case_id: str = "chat_case",
    mask_source: str = "gt_seg",  # "gt_seg" | "pred_mask"
    checkpoint: Optional[Path] = None,
) -> RouteResult:
    """
    Multimodal router for chat:
      - If NIfTI present: produce lesion stats + overlay PNGs + RAG text
      - Else: kind="text"

    mask_source:
      - "gt_seg": use provided seg/mask if present
      - "pred_mask": run UNet3D using checkpoint (requires t1/t1ce/t2/flair)
    """
    paths = [Path(p) for p in uploaded_paths if Path(p).exists()]

    nifti_files = [p for p in paths if _is_nifti(p)]
    if not nifti_files:
        return RouteResult(kind="text", rag_text="", structured={"note": "No NIfTI files provided."}, overlay_paths=[])

    if nib is None:
        return RouteResult(
            kind="nifti",
            rag_text="NIfTI files detected, but nibabel is not installed in this environment.",
            structured={"error": "missing_dependency", "dependency": "nibabel"},
            overlay_paths=[],
        )

    # Map roles -> first file per role
    mods: Dict[str, Path] = {}
    for p in nifti_files:
        mods.setdefault(_guess_role(p), p)

    img_path = _find_best_image(mods)
    if img_path is None:
        return RouteResult(
            kind="nifti",
            rag_text="NIfTI files found but no image volume could be selected.",
            structured={"error": "no_image_volume", "roles": list(mods.keys())},
            overlay_paths=[],
        )

    image, zooms = _load_nifti_with_spacing(img_path)

    mask: Optional[np.ndarray] = None
    mask_info: Dict[str, Any] = {"mask_source": mask_source}

    if mask_source == "gt_seg":
        seg_path = _find_seg(mods)
        if seg_path is not None:
            mask, zooms_m = _load_nifti_with_spacing(seg_path)
            if mask.shape != image.shape:
                mask_info["warning"] = "seg_shape_mismatch"
                mask_info["image_shape"] = list(image.shape)
                mask_info["seg_shape"] = list(mask.shape)
                # If mismatched, don't compute stats to avoid lying
                mask = None
            elif tuple(np.round(zooms_m, 6)) != tuple(np.round(zooms, 6)):
                mask_info["warning"] = "seg_spacing_mismatch"
                mask_info["image_zooms"] = zooms
                mask_info["seg_zooms"] = zooms_m
        else:
            mask_info["warning"] = "gt_seg_requested_but_not_found"

    elif mask_source == "pred_mask":
        try:
            mask, zooms_inf = _run_segmentation_inference(mods, checkpoint=checkpoint)
            if tuple(np.round(zooms_inf, 6)) != tuple(np.round(zooms, 6)):
                mask_info["warning"] = "pred_mask_spacing_mismatch"
                mask_info["image_zooms"] = zooms
                mask_info["pred_zooms"] = zooms_inf
            if mask.shape != image.shape:
                mask_info["warning"] = "pred_mask_shape_mismatch"
                mask_info["image_shape"] = list(image.shape)
                mask_info["pred_shape"] = list(mask.shape)
                mask = None
        except Exception as e:
            mask_info["error"] = f"pred_mask_failed: {type(e).__name__}: {e}"
            mask = None

    else:
        mask_info["warning"] = "unknown_mask_source"

    structured: Dict[str, Any] = {
        "case_id": case_id,
        "image_path": str(img_path),
        "image_role": _guess_role(img_path),
        "image_shape_ijk": list(image.shape[:3]),
        "zooms_xyz_mm": list(zooms),
        "mask_info": mask_info,
    }

    if mask is None:
        structured["has_lesion"] = False
        structured["lesion_stats"] = {"lesion_voxels": 0, "note": "No usable mask available; lesion stats not computed."}
    else:
        stats = _mask_stats(mask, zooms_xyz=zooms)
        structured["has_lesion"] = stats["lesion_voxels"] > 0
        structured["lesion_stats"] = stats

    overlay_dir = out_dir / "overlays" / case_id
    overlay_paths = _render_overlay_slices(image=image, mask=mask, out_dir=overlay_dir, prefix=case_id)

    rag_text = _make_rag_text(structured)
    return RouteResult(kind="nifti", rag_text=rag_text, structured=structured, overlay_paths=overlay_paths)
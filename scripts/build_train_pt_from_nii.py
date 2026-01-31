from pathlib import Path
from typing import List, Optional, Tuple
import torch
import nibabel as nib
import numpy as np
from tqdm import tqdm

# 🔽 RAW TRAINING DATA LOCATION
TRAIN_RAW = Path("data/train")

# 🔽 OUTPUT PROCESSED DATA
OUT_DIR = Path("data/processed/train")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODALITIES = ["flair", "t1", "t1ce", "t2"]

# If True -> stop on first missing file
# If False -> skip that patient and continue
STRICT = False


def load_nii(path: Path) -> np.ndarray:
    """Load NIfTI -> float32 numpy. Keeps raw values."""
    return nib.load(str(path)).get_fdata().astype(np.float32)


def to_dhw(arr_hwd: np.ndarray) -> np.ndarray:
    """
    nibabel returns [H, W, D] for BraTS.
    Convert -> [D, H, W]
    """
    if arr_hwd.ndim != 3:
        raise ValueError(f"Expected 3D array, got {arr_hwd.shape}")
    return np.transpose(arr_hwd, (2, 0, 1))


def find_one(pdir: Path, patient_id: str, mod: str) -> Path:
    mod_l = mod.lower()

    # 1) Normal modality search (strict patterns)
    patterns = [
        f"*_{mod}.nii.gz",
        f"*_{mod}.nii",
        f"*{mod}*.nii.gz",
        f"*{mod}*.nii",
    ]

    for pat in patterns:
        hits = [h for h in pdir.rglob(pat) if h.is_file()]
        if hits:
            hits.sort(key=lambda x: (patient_id not in x.name, len(x.name)))
            return hits[0]

    # 2) ✅ Special handling for segmentation: CASE-INSENSITIVE fallback
    if mod_l in ("seg", "mask", "label", "labels"):
        nii_files = [h for h in pdir.rglob("*.nii*") if h.is_file()]
        seg_keywords = ("seg", "segm", "segmentation", "mask", "label", "gt")

        seg_hits = [h for h in nii_files if any(k in h.name.lower() for k in seg_keywords)]
        if seg_hits:
            # Prefer ones that look closest to BraTS naming if present
            seg_hits.sort(key=lambda x: (patient_id not in x.name, len(x.name)))
            return seg_hits[0]

    # 3) If still not found, raise with sample
    sample = sorted([x.name for x in pdir.rglob("*.nii*")])[:20]
    raise FileNotFoundError(
        f"Missing '{mod}' for {patient_id} in {pdir}\n"
        f"Found NIfTI files (sample): {sample}"
    )


def build_patient_pt(pdir: Path) -> Tuple[str, dict]:
    """
    Build one patient sample:
      image: float32 [4, D, H, W]
      seg:   int16   [D, H, W]  (keeps 0/1/2/4)
      mask:  uint8   [D, H, W]  (seg>0)
    """
    pid = pdir.name

    # -------- modalities -> [4,D,H,W] float32 --------
    vols = []
    for mod in MODALITIES:
        mod_path = find_one(pdir, pid, mod)
        vol = to_dhw(load_nii(mod_path)).astype(np.float32)
        vols.append(vol)

    volume_4ch = np.stack(vols, axis=0).astype(np.float32)

    # -------- segmentation -> [D,H,W] int16 (KEEP MULTICLASS) --------
    seg_path = find_one(pdir, pid, "seg")
    seg = to_dhw(load_nii(seg_path))

    # BraTS seg labels are typically 0,1,2,4 — keep them!
    seg_int = np.rint(seg).astype(np.int16)  # safe convert float->int
    mask = (seg_int > 0).astype(np.uint8)

    payload = {
        "image": torch.from_numpy(volume_4ch),  # float32
        "seg": torch.from_numpy(seg_int),       # int16 multiclass
        "mask": torch.from_numpy(mask),         # uint8 binary
        "patient_id": pid,
    }
    return pid, payload


def main():
    patient_dirs = sorted([p for p in TRAIN_RAW.iterdir() if p.is_dir()])
    if not patient_dirs:
        raise RuntimeError(f"No training folders found in {TRAIN_RAW.resolve()}")

    saved = 0
    skipped = 0
    errors = []

    for pdir in tqdm(patient_dirs, desc="Building train .pt"):
        try:
            pid, payload = build_patient_pt(pdir)
            out_path = OUT_DIR / f"{pid}.pt"
            torch.save(payload, out_path)
            saved += 1
        except Exception as e:
            skipped += 1
            msg = f"[SKIP] {pdir.name}: {e}"
            errors.append(msg)
            if STRICT:
                raise
            # optional: print occasionally
            if skipped <= 5:
                print(msg)

    print(f"\n✅ Saved   : {saved}")
    print(f"⚠️ Skipped : {skipped}")
    print(f"📁 Output  : {OUT_DIR.resolve()}")

    # Show last few errors (helpful)
    if errors:
        print("\n--- Sample skip reasons (up to 5) ---")
        for line in errors[:5]:
            print(line)


if __name__ == "__main__":
    main()

from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
import monai
from torch.serialization import safe_globals

from models.embeddings.lesion_embedder_3d import LesionEmbeddingPipeline3D
from src.graph.knowledge_graph import ClinicalKnowledgeGraph

MODALITIES = ["t1", "t2", "flair", "t1ce"]


def _to_numpy_3d(x) -> np.ndarray:
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    x = np.asarray(x)

    if x.ndim == 4 and x.shape[0] == 1:
        x = x[0]
    if x.ndim != 3:
        raise ValueError(f"Expected 3D [D,H,W], got {x.shape}")
    return x.astype(np.float32)


def load_volume_mask_and_seg(pt_path: Path):
    """
    Returns:
      volume_4ch : [4, D, H, W]
      mask       : binary tumor mask (for embedding)
      seg        : original BraTS labels (0,1,2,4)
    """

    with safe_globals([monai.data.meta_tensor.MetaTensor]):
        data = torch.load(pt_path, map_location="cpu", weights_only=False)

    # ---------------------------
    # Load volume
    # ---------------------------
    if "image" in data:  # train format
        img = data["image"]
        if torch.is_tensor(img):
            img = img.detach().cpu().numpy()

        if img.ndim == 5 and img.shape[0] == 1:
            img = img[0]   # [4,D,H,W]

        volume_4ch = img.astype(np.float32)

    else:  # val format
        vols = []
        for k in MODALITIES:
            vols.append(_to_numpy_3d(data[k]))
        volume_4ch = np.stack(vols, axis=0).astype(np.float32)

    # ---------------------------
    # Load segmentation
    # ---------------------------
    if "seg" not in data:
        return volume_4ch, None, None

    seg = _to_numpy_3d(data["seg"]).astype(np.int16)
    mask = (seg > 0).astype(np.uint8)

    return volume_4ch, mask, seg


def safe_props(meta: Dict[str, Any]) -> Dict[str, Any]:
    props = dict(meta)
    if "type" in props:
        props["embedding_type"] = props.pop("type")
    return props


def brats_region_stats(seg: np.ndarray) -> Dict[str, float | int]:
    n1 = int((seg == 1).sum())
    n2 = int((seg == 2).sum())
    n4 = int((seg == 4).sum())
    wt = n1 + n2 + n4
    tc = n1 + n4
    et = n4
    return {
        "ncr_net_vox": n1,
        "edema_vox": n2,
        "enhancing_vox": n4,
        "wt_vox": wt,
        "tc_vox": tc,
        "et_vox": et,
        "et_pct": float(et / (wt + 1e-8)),
        "tc_pct": float(tc / (wt + 1e-8)),
    }


def main():
    processed_dir = Path("data/processed/train")  # or val
    out_path = Path("artifacts/kg_3d.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pt_files = sorted(processed_dir.glob("*.pt"))
    if not pt_files:
        raise FileNotFoundError(f"No .pt files found in: {processed_dir.resolve()}")

    kg = ClinicalKnowledgeGraph()
    pipeline = LesionEmbeddingPipeline3D(cube_size=(64, 64, 64))

    total = 0
    lesions_added = 0
    masks_missing = 0
    masks_empty = 0

    for pt_path in pt_files:
        patient_id = pt_path.stem
        total += 1

        volume_4ch, mask, seg = load_volume_mask_and_seg(pt_path)

        p_node = kg.add_patient(patient_id, metadata={"split": processed_dir.name})

        mri_node = kg.add_mri_volume(
            patient_node=p_node,
            modality="MULTI_4CH",
            embedding=None,
            metadata={
                "shape_4ch": list(volume_4ch.shape),
                "cube_size": [64, 64, 64],
                "source_file": str(pt_path),
            },
        )

        if mask is None:
            masks_missing += 1
        elif mask.sum() == 0:
            masks_empty += 1
        else:
            out = pipeline.embed_from_volume_and_mask(
                volume_4ch=volume_4ch,
                mask=mask,
                patient_id=patient_id,
                margin=5,
            )
            if out is not None:
                props = safe_props(out.meta)

                if seg is not None:
                    props.update(brats_region_stats(seg))

                # ✅ CREATE LESION NODE
                lesion_id = kg.add_lesion(
                    mri_node=mri_node,
                    lesion_type="tumor",
                    embedding=out.embedding,
                    properties=props,
                )

                # ✅ ADD SUBREGION NODES
                if seg is not None:
                    kg.add_lesion_region(
                        lesion_id,
                        "WT",
                        props["wt_vox"],
                    )

                    kg.add_lesion_region(
                        lesion_id,
                        "TC",
                        props["tc_vox"],
                        props["tc_pct"],
                    )

                    kg.add_lesion_region(
                        lesion_id,
                        "ET",
                        props["et_vox"],
                        props["et_pct"],
                    )

                lesions_added += 1

        if total % 25 == 0:
            print(
                f"Processed {total}/{len(pt_files)} | lesions: {lesions_added} | "
                f"mask_missing: {masks_missing} | mask_empty: {masks_empty}"
            )

    kg.save(out_path)
    print(f"✅ 3D Knowledge Graph saved to: {out_path}")
    print(f"Patients processed: {total}")
    print(f"Lesions embedded:   {lesions_added}")
    print(f"Masks missing:      {masks_missing}")
    print(f"Masks empty:        {masks_empty}")


if __name__ == "__main__":
    main()

# training/evaluate_segmentation.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from loguru import logger

from config import paths, training_config, model_config
from ingestion.data_ingestion import BraTSDataIngestion
from preprocessing.mri_preprocessing import MRIPreprocessor, MODALITY_KEYS
from models.segmentation.unet3d import UNet3DSegmenter


def remap_brats_labels(seg: torch.Tensor) -> torch.Tensor:
    if seg.ndim == 4:
        seg0 = seg[0]
    else:
        seg0 = seg
    seg0 = seg0.long()
    seg0 = torch.where(seg0 == 4, torch.tensor(3, device=seg0.device), seg0)
    seg0 = torch.clamp(seg0, 0, 3)
    return seg0.unsqueeze(0)


class BraTSEvalDataset(Dataset):
    def __init__(self, ingestion: BraTSDataIngestion, patient_ids: List[str], preprocessor: MRIPreprocessor):
        self.ingestion = ingestion
        self.patient_ids = patient_ids
        self.preprocessor = preprocessor

    def __len__(self) -> int:
        return len(self.patient_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        pid = self.patient_ids[idx]
        sample = self.ingestion.load_patient_data(pid)

        processed = self.preprocessor.preprocess_patient(
            modalities=sample.modalities,
            segmentation=sample.segmentation,
            augment=False,
        )

        img = torch.cat([processed[k] for k in MODALITY_KEYS], dim=0).float()  # [4,D,H,W]
        seg = processed.get("seg", None)
        if seg is None:
            raise ValueError(f"Validation requires segmentation for patient {pid}")

        seg = remap_brats_labels(seg).long()  # [1,D,H,W]
        return {"image": img, "label": seg, "patient_id": pid}


def _collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    images = torch.stack([b["image"] for b in batch], dim=0)  # [B,4,D,H,W]
    labels = torch.stack([b["label"] for b in batch], dim=0)  # [B,1,D,H,W]
    patient_ids = [b["patient_id"] for b in batch]
    return {"image": images, "label": labels, "patient_id": patient_ids}


def load_checkpoint(model: UNet3DSegmenter, ckpt_path: Path) -> None:
    ckpt = torch.load(ckpt_path, map_location=model.device)
    model.model.load_state_dict(ckpt["model_state_dict"])
    logger.info(f"Loaded checkpoint: {ckpt_path} (epoch={ckpt.get('epoch')})")


@torch.no_grad()
def evaluate(model: UNet3DSegmenter, loader: DataLoader) -> Tuple[float, float]:
    model.model.eval()

    total_loss = 0.0
    total_dice = 0.0
    n = 0

    for batch in loader:
        out = model.validate_step(batch)  # uses DiceLoss + DiceMetric inside :contentReference[oaicite:4]{index=4}
        total_loss += out["loss"]
        total_dice += out["dice"]
        n += 1

    return total_loss / max(n, 1), total_dice / max(n, 1)


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    # Use validation dataset
    val_ing = BraTSDataIngestion(data_root=paths.DATA_ROOT_VAL)
    val_ids = val_ing.discover_patients()

    if not val_ids:
        raise RuntimeError("No validation patients found. Check DATA_ROOT_VAL in config.py")

    preprocessor = MRIPreprocessor(target_size=model_config.IMG_SIZE)
    val_ds = BraTSEvalDataset(val_ing, val_ids, preprocessor)

    val_loader = DataLoader(
        val_ds,
        batch_size=training_config.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=(device == "cuda"),
        collate_fn=_collate_fn,
    )

    model = UNet3DSegmenter(
        in_channels=model_config.SEGMENTATION_IN_CHANNELS,
        out_channels=model_config.SEGMENTATION_OUT_CHANNELS,
        device=device,
    )
    model.setup_training(lr=training_config.LEARNING_RATE)

    ckpt_path = paths.SEGMENTATION_MODEL_DIR / "unet3d_best.pt"
    if not ckpt_path.exists():
        ckpt_path = paths.SEGMENTATION_MODEL_DIR / "unet3d_last.pt"

    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"No checkpoint found in {paths.SEGMENTATION_MODEL_DIR}. "
            "Train first to create unet3d_best.pt or unet3d_last.pt"
        )

    load_checkpoint(model, ckpt_path)

    val_loss, val_dice = evaluate(model, val_loader)
    logger.info(f"✅ Evaluation results | val_loss={val_loss:.4f}, val_dice={val_dice:.4f}")


if __name__ == "__main__":
    main()

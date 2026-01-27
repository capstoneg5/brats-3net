# training/train_segmentation.py
from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from loguru import logger

from config import paths, training_config, model_config
from ingestion.data_ingestion import BraTSDataIngestion
from preprocessing.mri_preprocessing import MRIPreprocessor, MODALITY_KEYS
from models.segmentation.unet3d import UNet3DSegmenter, SegmentationTrainer


# -------------------------
# Reproducibility
# -------------------------
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # safe if no cuda
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -------------------------
# BraTS label remap
# BraTS labels: 0,1,2,4 -> map to 0,1,2,3
# -------------------------
def remap_brats_labels(seg: torch.Tensor) -> torch.Tensor:
    # seg expected shape: [1, D, H, W] or [D,H,W]
    if seg.ndim == 4:
        seg0 = seg[0]
    else:
        seg0 = seg

    seg0 = seg0.long()
    seg0 = torch.where(seg0 == 4, torch.tensor(3, device=seg0.device), seg0)
    # Ensure values are only 0..3
    seg0 = torch.clamp(seg0, 0, 3)
    return seg0.unsqueeze(0)  # [1, D,H,W]


# -------------------------
# Dataset
# -------------------------
class BraTSSegmentationDataset(Dataset):
    """
    Loads BraTS patient volumes, applies preprocessing (MONAI),
    returns:
      image: [4, D, H, W]
      label: [1, D, H, W] with values in {0,1,2,3}
    """

    def __init__(
        self,
        ingestion: BraTSDataIngestion,
        patient_ids: List[str],
        preprocessor: MRIPreprocessor,
        augment: bool = False,
    ) -> None:
        self.ingestion = ingestion
        self.patient_ids = patient_ids
        self.preprocessor = preprocessor
        self.augment = augment

    def __len__(self) -> int:
        return len(self.patient_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        pid = self.patient_ids[idx]
        sample = self.ingestion.load_patient_data(pid)

        # preprocess_patient expects dict of np arrays keyed by modalities + optional seg
        processed = self.preprocessor.preprocess_patient(
            modalities=sample.modalities,
            segmentation=sample.segmentation,
            augment=self.augment,
        )

        # processed has keys: t1,t2,flair, t1ce and seg (all torch tensors)
        # Each modality after EnsureChannelFirst is typically [1, D, H, W]
        img = torch.cat([processed[k] for k in MODALITY_KEYS], dim=0).float()  # [4,D,H,W]

        if "seg" not in processed:
            raise ValueError(f"Training requires segmentation mask for patient {pid}")

        seg = processed["seg"]
        seg = remap_brats_labels(seg).long()  # [1,D,H,W]

        return {"image": img, "label": seg, "patient_id": pid}


def _collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    # Custom collate to keep patient_id list too
    images = torch.stack([b["image"] for b in batch], dim=0)  # [B,4,D,H,W]
    labels = torch.stack([b["label"] for b in batch], dim=0)  # [B,1,D,H,W]
    patient_ids = [b["patient_id"] for b in batch]
    return {"image": images, "label": labels, "patient_id": patient_ids}


# -------------------------
# Checkpointing
# -------------------------
def save_checkpoint(
    ckpt_path: Path,
    model: UNet3DSegmenter,
    epoch: int,
    best_val_loss: float,
) -> None:
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "model_state_dict": model.model.state_dict(),
            "optimizer_state_dict": model.optimizer.state_dict() if model.optimizer else None,
            "scheduler_state_dict": model.scheduler.state_dict() if model.scheduler else None,
            "config": {
                "in_channels": model_config.SEGMENTATION_IN_CHANNELS,
                "out_channels": model_config.SEGMENTATION_OUT_CHANNELS,
                "img_size": model_config.IMG_SIZE,
            },
        },
        ckpt_path,
    )


def main() -> None:
    set_seed(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    # Ingestion
    train_ing = BraTSDataIngestion(data_root=paths.DATA_ROOT_TRAIN)
    val_ing = BraTSDataIngestion(data_root=paths.DATA_ROOT_VAL)

    train_ids = train_ing.discover_patients()
    val_ids = val_ing.discover_patients()

    if len(train_ids) == 0:
        raise RuntimeError("No training patients found. Check DATA_ROOT_TRAIN in config.py")

    if len(val_ids) == 0:
        logger.warning("No validation patients found. Validation will be skipped.")
        val_ids = []

    # Preprocessor (uses config IMG_SIZE + augmentation flags)
    preprocessor = MRIPreprocessor(target_size=model_config.IMG_SIZE)

    # Datasets & loaders
    train_ds = BraTSSegmentationDataset(train_ing, train_ids, preprocessor, augment=True)
    val_ds = BraTSSegmentationDataset(val_ing, val_ids, preprocessor, augment=False) if val_ids else None

    train_loader = DataLoader(
        train_ds,
        batch_size=training_config.BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=(device == "cuda"),
        collate_fn=_collate_fn,
    )

    val_loader = None
    if val_ds is not None:
        val_loader = DataLoader(
            val_ds,
            batch_size=training_config.BATCH_SIZE,
            shuffle=False,
            num_workers=2,
            pin_memory=(device == "cuda"),
            collate_fn=_collate_fn,
        )

    # Model
    model = UNet3DSegmenter(
        in_channels=model_config.SEGMENTATION_IN_CHANNELS,
        out_channels=model_config.SEGMENTATION_OUT_CHANNELS,
        device=device,
    )
    model.setup_training(lr=training_config.LEARNING_RATE)

    trainer = SegmentationTrainer(model, train_loader, val_loader)

    # Training loop
    best_val_loss = float("inf")
    patience = training_config.PATIENCE
    min_delta = training_config.MIN_DELTA
    patience_counter = 0

    ckpt_best = paths.SEGMENTATION_MODEL_DIR / "unet3d_best.pt"
    ckpt_last = paths.SEGMENTATION_MODEL_DIR / "unet3d_last.pt"

    logger.info(
        f"Starting training | epochs={training_config.NUM_EPOCHS} "
        f"| batch={training_config.BATCH_SIZE} | lr={training_config.LEARNING_RATE}"
    )

    for epoch in range(1, training_config.NUM_EPOCHS + 1):
        train_loss, train_dice = trainer.train_epoch()

        if val_loader is not None and (epoch % training_config.VAL_INTERVAL == 0 or epoch == 1):
            val_loss, val_dice = trainer.validate_epoch()

            # scheduler step on val loss
            if model.scheduler is not None:
                model.scheduler.step(val_loss)

            logger.info(
                f"[Epoch {epoch:03d}] "
                f"train_loss={train_loss:.4f}, train_dice={train_dice:.4f} | "
                f"val_loss={val_loss:.4f}, val_dice={val_dice:.4f}"
            )

            # checkpoint (best)
            improved = (best_val_loss - val_loss) > min_delta
            if improved:
                best_val_loss = val_loss
                patience_counter = 0
                save_checkpoint(ckpt_best, model, epoch, best_val_loss)
                logger.info(f"✅ Saved BEST checkpoint -> {ckpt_best} (val_loss={best_val_loss:.4f})")
            else:
                patience_counter += 1
                logger.info(f"No improvement. patience={patience_counter}/{patience}")

            # early stopping
            if patience_counter >= patience:
                logger.warning("⏹ Early stopping triggered.")
                break
        else:
            logger.info(
                f"[Epoch {epoch:03d}] train_loss={train_loss:.4f}, train_dice={train_dice:.4f} | val=skipped"
            )

        # checkpoint (last)
        save_checkpoint(ckpt_last, model, epoch, best_val_loss)

    logger.info("Training complete.")


if __name__ == "__main__":
    main()

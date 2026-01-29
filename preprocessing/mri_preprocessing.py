from typing import Dict, Tuple, Optional
import numpy as np
import torch
import SimpleITK as sITK
from loguru import logger

from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    CropForegroundd,
    Resized,
    NormalizeIntensityd,
    RandRotate90d,
    RandFlipd,
    RandAffined,
    RandGaussianNoised,
    ToTensord,
)

from config import model_config, training_config


MODALITY_KEYS = ["t1", "t2", "flair", "t1ce"]
ALL_KEYS = MODALITY_KEYS + ["seg"]


class MRIPreprocessor:
    """
    Preprocessing pipeline for BraTS MRI data
    """

    def __init__(self, target_size: Tuple[int, int, int] = None) -> None:
        self.target_size = target_size or model_config.IMG_SIZE
        logger.info(f"Initializing MRI Preprocessor | Target size: {self.target_size}")

        self.preprocessing_transforms = self._build_preprocessing_pipeline()
        self.augmentation_transforms = self._build_augmentation_pipeline()

    # --------------------------------------------------
    # MONAI PIPELINES
    # --------------------------------------------------

    def _build_preprocessing_pipeline(self) -> Compose:
        return Compose([
            EnsureChannelFirstd(keys=ALL_KEYS, channel_dim="no_channel", allow_missing_keys=True),
            Orientationd(keys=ALL_KEYS, axcodes="RAS", allow_missing_keys=True),
            Spacingd(
                keys=ALL_KEYS,
                pixdim=(1.0, 1.0, 1.0),
                mode=("bilinear", "bilinear", "bilinear", "bilinear", "nearest"),
                allow_missing_keys=True,
            ),
            CropForegroundd(
                keys=ALL_KEYS,
                source_key="t1",
                margin=10,
                allow_missing_keys=True,
            ),
            Resized(
                keys=ALL_KEYS,
                spatial_size=self.target_size,
                mode=("trilinear", "trilinear", "trilinear", "trilinear", "nearest"),
                allow_missing_keys=True,
            ),
            NormalizeIntensityd(
                keys=MODALITY_KEYS,
                nonzero=True,
                channel_wise=True,
            ),
            ToTensord(keys=ALL_KEYS, allow_missing_keys=True),
        ])

    @staticmethod
    def _build_augmentation_pipeline() -> Optional[Compose]:
        if not training_config.USE_AUGMENTATION:
            return None

        return Compose([
            RandRotate90d(keys=ALL_KEYS, prob=0.5, spatial_axes=(0, 2)),
            RandFlipd(keys=ALL_KEYS, prob=0.5, spatial_axis=0),
            RandFlipd(keys=ALL_KEYS, prob=0.5, spatial_axis=1),
            RandFlipd(keys=ALL_KEYS, prob=0.5, spatial_axis=2),
            RandAffined(
                keys=ALL_KEYS,
                prob=0.3,
                rotate_range=training_config.ROTATION_RANGE,
                scale_range=training_config.SCALE_RANGE,
                mode=("bilinear", "bilinear", "bilinear", "bilinear", "nearest"),
            ),
            RandGaussianNoised(
                keys=MODALITY_KEYS,
                prob=0.2,
                mean=0.0,
                std=0.1,
            ),
        ])

    # --------------------------------------------------
    # CORE OPERATIONS
    # --------------------------------------------------

    @staticmethod
    def normalize_intensity(image: np.ndarray) -> np.ndarray:
        mask = image > 0
        mean = image[mask].mean()
        std = image[mask].std()
        image[mask] = (image[mask] - mean) / (std + 1e-8)
        return image.astype(np.float32)

    @staticmethod
    def skull_strip(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        sITK_img = sITK.GetImageFromArray(image)

        otsu_filter = sITK.OtsuThresholdImageFilter()
        otsu_filter.SetInsideValue(0)
        otsu_filter.SetOutsideValue(1)

        mask = otsu_filter.Execute(sITK_img)

        mask = sITK.BinaryMorphologicalClosing(mask, [5, 5, 5])
        mask = sITK.BinaryMorphologicalOpening(mask, [3, 3, 3])

        mask_np = sITK.GetArrayFromImage(mask).astype(np.float32)
        return image * mask_np, mask_np

    # --------------------------------------------------
    # PATIENT LEVEL PIPELINE
    # --------------------------------------------------

    def preprocess_patient(
        self,
        modalities: Dict[str, np.ndarray],
        segmentation: Optional[np.ndarray] = None,
        augment: bool = False,
    ) -> Dict[str, torch.Tensor]:

        data = {}

        for key in MODALITY_KEYS:
            data[key] = self.normalize_intensity(modalities[key])

        if segmentation is not None:
            data["seg"] = segmentation.astype(np.uint8)

        data = self.preprocessing_transforms(data)

        if augment and self.augmentation_transforms:
            data = self.augmentation_transforms(data)

        return data


# --------------------------------------------------
# DEBUG / VS CODE RUN
# --------------------------------------------------

def main() -> None:
    from ingestion.data_ingestion import BraTSDataIngestion
    from config import paths

    # Process training data
    print("Processing Training Data for Preprocessing:")
    ingestion_train = BraTSDataIngestion(data_root=paths.DATA_ROOT_TRAIN)
    patient_ids_train = ingestion_train.discover_patients()

    if patient_ids_train:
        preprocessor = MRIPreprocessor()
        processed_count = 0
        for patient_id in patient_ids_train:
            sample = ingestion_train.load_patient_data(patient_id)
            output = preprocessor.preprocess_patient(
                sample.modalities,
                sample.segmentation,
                augment=False,
            )
            processed_count += 1
            if processed_count % 50 == 0:
                print(f"Processed {processed_count}/{len(patient_ids_train)} training patients")
        print(f"Total training patients preprocessed: {processed_count}")

    # Process validation data
    print("\nProcessing Validation Data for Preprocessing:")
    ingestion_val = BraTSDataIngestion(data_root=paths.DATA_ROOT_VAL)
    patient_ids_val = ingestion_val.discover_patients()

    if patient_ids_val:
        preprocessor = MRIPreprocessor()
        processed_count = 0
        for patient_id in patient_ids_train:
            sample = ingestion_train.load_patient_data(patient_id)

            processed = preprocessor.preprocess_patient(
                sample.modalities,
                sample.segmentation,
                augment=False,
            )

            save_path = (
                    paths.PROJECT_ROOT
                    / "data"
                    / "processed"
                    / "train"
                    / f"{patient_id}.pt"
            )
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(processed, save_path)
            processed_count += 1
            if processed_count % 25 == 0:
                print(f"Processed {processed_count}/{len(patient_ids_val)} validation patients")
        print(f"Total validation patients preprocessed: {processed_count}")


if __name__ == "__main__":
    main()

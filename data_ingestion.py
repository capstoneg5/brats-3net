import os
import nibabel as nib
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import json
from loguru import logger
from tqdm import tqdm

# Import your config file (ensure it's in the same folder or in PYTHONPATH)
from config import paths, BRATS_MODALITIES, BRATS_LABELS

# =========================
# MRI DATA CONTAINER
# =========================

@dataclass
class MRIData:
    """Container for MRI scan data"""
    patient_id: str
    modalities: Dict[str, np.ndarray]
    segmentation: Optional[np.ndarray] = None
    metadata: Optional[Dict] = None

# =========================
# BraTS Data Ingestion
# =========================

class BraTSDataIngestion:
    """
    Data ingestion pipeline for BraTS dataset
    Loads multi-sequence MRI volumes and segmentation masks
    """

    def __init__(self, data_root: Path = None):
        """
        Initialize data ingestion
        Args:
            data_root: Root directory containing BraTS dataset
        """
        if data_root is None:
            self.data_root = paths.DATA_ROOT_TRAIN
        else:
            self.data_root = data_root

        logger.info(f"Initializing BraTS data ingestion from: {self.data_root}")

        if not self.data_root.exists():
            raise FileNotFoundError(
                f"Data directory not found: {self.data_root}\n"
                f"Please update the DATA_ROOT path in config.py"
            )

    # -------------------------
    # Discover patient folders
    # -------------------------
    def discover_patients(self) -> List[str]:
        patient_dirs = []
        for item in self.data_root.iterdir():
            if item.is_dir() and item.name.startswith('BraTS'):
                patient_dirs.append(item.name)
        logger.info(f"Found {len(patient_dirs)} patient directories")
        return sorted(patient_dirs)

    # -------------------------
    # Load NIfTI file
    # -------------------------
    def load_nifti(self, filepath: Path) -> np.ndarray:
        try:
            nii = nib.load(str(filepath))
            data = nii.get_fdata()
            return data.astype(np.float32)
        except Exception as e:
            logger.error(f"Error loading {filepath}: {e}")
            raise

    # -------------------------
    # Load patient data
    # -------------------------
    def load_patient_data(self, patient_id: str) -> MRIData:
        patient_dir = self.data_root / patient_id

        if not patient_dir.exists():
            raise FileNotFoundError(f"Patient directory not found: {patient_dir}")

        modalities = {}
        for modality in BRATS_MODALITIES:
            filepath = patient_dir / f"{patient_id}_{modality}.nii.gz"
            if not filepath.exists():
                filepath = patient_dir / f"{patient_id}_{modality}.nii"

            if filepath.exists():
                modalities[modality] = self.load_nifti(filepath)
                logger.debug(f"Loaded {modality} for {patient_id}: {modalities[modality].shape}")
            else:
                logger.warning(f"Modality {modality} not found for {patient_id}")

        # Load segmentation
        seg_filepath = patient_dir / f"{patient_id}_seg.nii.gz"
        if not seg_filepath.exists():
            seg_filepath = patient_dir / f"{patient_id}_seg.nii"

        segmentation = None
        if seg_filepath.exists():
            segmentation = self.load_nifti(seg_filepath)
            logger.debug(f"Loaded segmentation for {patient_id}: {segmentation.shape}")

        metadata = {
            'patient_id': patient_id,
            'modalities_available': list(modalities.keys()),
            'has_segmentation': segmentation is not None,
            'image_shape': list(modalities[BRATS_MODALITIES[0]].shape) if modalities else None
        }

        return MRIData(
            patient_id=patient_id,
            modalities=modalities,
            segmentation=segmentation,
            metadata=metadata
        )

    # -------------------------
    # Validate data
    # -------------------------
    def validate_data(self, mri_data: MRIData) -> bool:
        shapes = [arr.shape for arr in mri_data.modalities.values()]
        if len(set(shapes)) > 1:
            logger.error(f"Inconsistent shapes for {mri_data.patient_id}: {shapes}")
            return False

        if mri_data.segmentation is not None:
            if mri_data.segmentation.shape != shapes[0]:
                logger.error(f"Segmentation shape mismatch for {mri_data.patient_id}: "
                             f"seg={mri_data.segmentation.shape}, img={shapes[0]}")
                return False

        for modality, data in mri_data.modalities.items():
            if np.isnan(data).any() or np.isinf(data).any():
                logger.error(f"NaN/Inf found in {modality} for {mri_data.patient_id}")
                return False

        return True

    # -------------------------
    # Dataset statistics
    # -------------------------
    def get_dataset_statistics(self, patient_ids: List[str]) -> Dict:
        stats = {
            'total_patients': len(patient_ids),
            'modality_counts': {mod: 0 for mod in BRATS_MODALITIES},
            'segmentation_count': 0,
            'image_shapes': [],
            'intensity_stats': {mod: {'mean': [], 'std': []} for mod in BRATS_MODALITIES}
        }

        for patient_id in tqdm(patient_ids[:50], desc="Computing statistics"):
            try:
                data = self.load_patient_data(patient_id)

                for modality in BRATS_MODALITIES:
                    if modality in data.modalities:
                        stats['modality_counts'][modality] += 1
                        arr = data.modalities[modality]
                        stats['intensity_stats'][modality]['mean'].append(arr.mean().item())
                        stats['intensity_stats'][modality]['std'].append(arr.std().item())

                if data.segmentation is not None:
                    stats['segmentation_count'] += 1

                if data.modalities:
                    shape = list(data.modalities[BRATS_MODALITIES[0]].shape)
                    stats['image_shapes'].append(shape)

            except Exception as e:
                logger.warning(f"Error processing {patient_id}: {e}")

        # Aggregate mean/std
        for modality in BRATS_MODALITIES:
            if stats['intensity_stats'][modality]['mean']:
                stats['intensity_stats'][modality]['mean'] = np.mean(stats['intensity_stats'][modality]['mean']).item()
                stats['intensity_stats'][modality]['std'] = np.mean(stats['intensity_stats'][modality]['std']).item()

        return stats

    # -------------------------
    # Save metadata to JSON
    # -------------------------
    def save_metadata(self, patient_ids: List[str], output_path: Path = None):
        output_path = output_path or paths.PROCESSED_DATA_DIR / "dataset_metadata.json"
        stats = self.get_dataset_statistics(patient_ids)
        metadata = {
            'dataset': 'BraTS',
            'version': '2020',
            'patient_ids': patient_ids,
            'statistics': stats
        }
        with open(output_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Metadata saved to {output_path}")

# =========================
# Example usage
# =========================

def main():
    # Process training data
    print("Processing Training Data:")
    ingestion_train = BraTSDataIngestion(data_root=paths.DATA_ROOT_TRAIN)
    patient_ids_train = ingestion_train.discover_patients()
    print(f"Found {len(patient_ids_train)} training patients")

    if patient_ids_train:
        sample_id = patient_ids_train[0]
        print(f"\nLoading sample training patient: {sample_id}")
        mri_data = ingestion_train.load_patient_data(sample_id)

        print(f"Patient ID: {mri_data.patient_id}")
        print(f"Modalities: {list(mri_data.modalities.keys())}")
        print(f"Has segmentation: {mri_data.segmentation is not None}")
        if mri_data.modalities:
            shape = mri_data.modalities[BRATS_MODALITIES[0]].shape
            print(f"Image shape: {shape}")

        is_valid = ingestion_train.validate_data(mri_data)
        print(f"Data valid: {is_valid}")

    ingestion_train.save_metadata(patient_ids_train, output_path=paths.PROCESSED_DATA_DIR / "dataset_metadata_train.json")

    # Process validation data
    print("\nProcessing Validation Data:")
    ingestion_val = BraTSDataIngestion(data_root=paths.DATA_ROOT_VAL)
    patient_ids_val = ingestion_val.discover_patients()
    print(f"Found {len(patient_ids_val)} validation patients")

    if patient_ids_val:
        sample_id = patient_ids_val[0]
        print(f"\nLoading sample validation patient: {sample_id}")
        mri_data = ingestion_val.load_patient_data(sample_id)

        print(f"Patient ID: {mri_data.patient_id}")
        print(f"Modalities: {list(mri_data.modalities.keys())}")
        print(f"Has segmentation: {mri_data.segmentation is not None}")
        if mri_data.modalities:
            shape = mri_data.modalities[BRATS_MODALITIES[0]].shape
            print(f"Image shape: {shape}")

        is_valid = ingestion_val.validate_data(mri_data)
        print(f"Data valid: {is_valid}")

    ingestion_val.save_metadata(patient_ids_val, output_path=paths.PROCESSED_DATA_DIR / "dataset_metadata_val.json")

if __name__ == "__main__":
    main()

"""
MedRAG-X Configuration File
VS Code compatible | No API keys required
"""

import nibabel as nib
import numpy as np
from pathlib import Path
from typing import cast
from nibabel.nifti1 import Nifti1Image
from dataclasses import dataclass
from typing import Tuple, Dict, List, Optional
import json
from loguru import logger
from tqdm import tqdm



# =========================
# PATH CONFIGURATION
# =========================

@dataclass
class PathConfig:
    """File paths configuration"""

    # Project root (safe for VS Code)
    ROOT_DIR: Path = Path(__file__).resolve().parent

    # CHANGE THIS PATH TO YOUR BraTS DATASET LOCATION
    DATA_ROOT_TRAIN: Path = Path("BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData")  # <-- MODIFY THIS
    DATA_ROOT_VAL: Path = Path("BraTS2020_ValidationData/MICCAI_BraTS2020_ValidationData")  # <-- MODIFY THIS
    DATA_ROOT: Path = DATA_ROOT_TRAIN  # Default to training for backward compatibility

    RAW_DATA_DIR: Path = DATA_ROOT / "raw"
    PROCESSED_DATA_DIR: Path = DATA_ROOT / "processed"
    EMBEDDINGS_DIR: Path = DATA_ROOT / "embeddings"

    # Models
    MODEL_DIR: Path = ROOT_DIR / "models"
    SEGMENTATION_MODEL_DIR: Path = MODEL_DIR / "segmentation"
    EMBEDDING_MODEL_DIR: Path = MODEL_DIR / "embeddings"

    # Databases
    GRAPH_DB_DIR: Path = ROOT_DIR / "graph_db"
    VECTOR_DB_DIR: Path = ROOT_DIR / "vector_db"

    # Logs & outputs
    LOG_DIR: Path = ROOT_DIR / "logs"
    OUTPUT_DIR: Path = ROOT_DIR / "outputs"
    AUDIT_DIR: Path = ROOT_DIR / "audit_trails"

    def __post_init__(self):
        """Create all directories safely"""
        for attr in self.__dict__.values():
            if isinstance(attr, Path):
                attr.mkdir(parents=True, exist_ok=True)

# =========================
# MODEL CONFIGURATION
# =========================

@dataclass
class ModelConfig:
    """Model configurations"""

    # Segmentation (BraTS)
    SEGMENTATION_MODEL: str = "3dunet"
    SEGMENTATION_IN_CHANNELS: int = 4
    SEGMENTATION_OUT_CHANNELS: int = 4
    SEGMENTATION_FEATURE_SIZE: int = 16

    IMG_SIZE: Tuple[int, int, int] = (128, 128, 128)

    # Embedding models (local / HuggingFace)
    TEXT_EMBEDDING_MODEL: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
    VISION_EMBEDDING_MODEL: str = "microsoft/swin-tiny-patch4-window7-224"
    MULTIMODAL_EMBEDDING_MODEL: str = "flaviagiammarino/pubmed-clip-vit-base-patch32"

    EMBEDDING_DIM: int = 768

    # LLM DISABLED (No API keys)
    ENABLE_LLM: bool = False
    LLM_MODEL: str = "microsoft/biogpt"  # Placeholder (Ollama / LM Studio later)
    LLM_MAX_TOKENS: int = 2048
    LLM_TEMPERATURE: float = 0.0

    # Retrieval
    TOP_K_SEMANTIC: int = 10
    TOP_K_STRUCTURAL: int = 5
    SIMILARITY_THRESHOLD: float = 0.7

# =========================
# TRAINING CONFIGURATION
# =========================

@dataclass
class TrainingConfig:
    """Training configurations"""

    BATCH_SIZE: int = 2
    LEARNING_RATE: float = 1e-4
    NUM_EPOCHS: int = 100
    VAL_INTERVAL: int = 2

    # Early stopping
    PATIENCE: int = 10
    MIN_DELTA: float = 0.001

    # Augmentation
    USE_AUGMENTATION: bool = True
    ROTATION_RANGE: Tuple[float, float] = (-15, 15)
    SCALE_RANGE: Tuple[float, float] = (0.9, 1.1)

# =========================
# GRAPH CONFIGURATION
# =========================

@dataclass
class GraphConfig:
    """Graph database configuration"""

    # Neo4j (optional – can disable if not installed)
    ENABLE_GRAPH_DB: bool = False
    NEO4J_URI: str = "bolt://localhost:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = "neo4j"

    EDGE_SIMILARITY_THRESHOLD: float = 0.8
    MAX_EDGES_PER_NODE: int = 20

# =========================
# API / UI CONFIGURATION
# =========================

@dataclass
class APIConfig:
    """API and UI configuration"""

    ENABLE_API: bool = False

    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000

    STREAMLIT_HOST: str = "localhost"
    STREAMLIT_PORT: int = 8501

# =========================
# GLOBAL INSTANCES
# =========================

paths = PathConfig()
model_config = ModelConfig()
training_config = TrainingConfig()
graph_config = GraphConfig()
api_config = APIConfig()

# =========================
# BRATS CONFIGURATION
# =========================

BRATS_MODALITIES = ["t1", "t2", "flair", "t1ce"]

BRATS_LABELS = {
    0: "background",
    1: "necrotic_core",
    2: "edema",
    4: "enhancing_tumor"
}

# =========================
# VALIDATION CODE
# =========================

@dataclass
class MRIData:
    """Container for MRI patient data"""
    patient_id: str
    modalities: Dict[str, np.ndarray]
    segmentation: Optional[np.ndarray] = None

    @property
    def has_segmentation(self) -> bool:
        return self.segmentation is not None

    def validate_data(self) -> bool:
        """Validate MRI data integrity"""
        try:
            # Check if all modalities have the same shape
            shapes = [mod.shape for mod in self.modalities.values()]
            if not all(shape == shapes[0] for shape in shapes):
                logger.error(f"Shape mismatch in modalities for {self.patient_id}: {shapes}")
                return False

            # Check for NaN/Inf values
            for mod_name, mod_data in self.modalities.items():
                if np.any(np.isnan(mod_data)) or np.any(np.isinf(mod_data)):
                    logger.error(f"NaN/Inf values found in {mod_name} for {self.patient_id}")
                    return False

            # Check segmentation shape if present
            if self.has_segmentation:
                if self.segmentation.shape != shapes[0]:
                    logger.error(f"Segmentation shape mismatch for {self.patient_id}: {self.segmentation.shape} vs {shapes[0]}")
                    return False

            return True
        except Exception as e:
            logger.error(f"Validation error for {self.patient_id}: {e}")
            return False

class BraTSDataIngestion:
    """Handles BraTS data ingestion and validation"""

    def __init__(self, data_root: Path):
        self.data_root = data_root
        logger.info(f"Initializing BraTS data ingestion from: {data_root}")

    def discover_patients(self) -> List[str]:
        """Discover patient directories"""
        if not self.data_root.exists():
            raise FileNotFoundError(f"Data root directory not found: {self.data_root}")

        patient_dirs = [d for d in self.data_root.iterdir() if d.is_dir() and d.name.startswith("BraTS")]
        patient_dirs.sort()
        logger.info(f"Found {len(patient_dirs)} patient directories")
        return [d.name for d in patient_dirs]

    @staticmethod
    def load_nifti(filepath: Path) -> np.ndarray:
        """Load NIfTI file"""
        try:
            img = cast(Nifti1Image, nib.load(filepath))
            data = img.get_fdata(dtype=np.float32)
            return data
        except Exception as e:
            logger.error(f"Failed to load {filepath}: {e}")
            raise

    def load_patient_data(self, patient_id: str) -> MRIData:
        """Load all data for a patient"""
        patient_dir = self.data_root / patient_id
        modalities = {}

        # Load modalities
        for mod in BRATS_MODALITIES:
            mod_file = patient_dir / f"{patient_id}_{mod}.nii.gz"
            if mod_file.exists():
                modalities[mod] = self.load_nifti(mod_file)
                logger.debug(f"Loaded {mod} for {patient_id}: {modalities[mod].shape}")
            else:
                logger.warning(f"Missing {mod} file for {patient_id}")

        # Load segmentation (if exists)
        seg_file = patient_dir / f"{patient_id}_seg.nii.gz"
        segmentation = None
        if seg_file.exists():
            segmentation = self.load_nifti(seg_file)
            logger.debug(f"Loaded segmentation for {patient_id}: {segmentation.shape}")

        return MRIData(
            patient_id=patient_id,
            modalities=modalities,
            segmentation=segmentation
        )

    @staticmethod
    def save_metadata(patients: List[MRIData], output_path: Path):
        """Save dataset metadata"""
        metadata = {
            "dataset_info": {
                "total_patients": len(patients),
                "modalities": BRATS_MODALITIES,
                "has_segmentation": any(p.has_segmentation for p in patients)
            },
            "patients": [
                {
                    "patient_id": p.patient_id,
                    "modalities": list(p.modalities.keys()),
                    "has_segmentation": p.has_segmentation,
                    "image_shape": list(next(iter(p.modalities.values())).shape) if p.modalities else None
                }
                for p in patients
            ]
        }

        with open(output_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Metadata saved to {output_path}")

def main():
    """Main validation function for both training and validation data"""

    # Ensure processed directories exist
    (paths.DATA_ROOT_TRAIN / "processed").mkdir(parents=True, exist_ok=True)
    (paths.DATA_ROOT_VAL / "processed").mkdir(parents=True, exist_ok=True)

    # Process Training Data
    print("Processing Training Data:")
    train_ingestor = BraTSDataIngestion(paths.DATA_ROOT_TRAIN)
    train_patients = train_ingestor.discover_patients()

    # Load sample training patient
    if train_patients:
        sample_patient = train_ingestor.load_patient_data(train_patients[0])
        print(f"Loading sample training patient: {sample_patient.patient_id}")
        print(f"Patient ID: {sample_patient.patient_id}")
        print(f"Modalities: {list(sample_patient.modalities.keys())}")
        print(f"Has segmentation: {sample_patient.has_segmentation}")
        print(f"Image shape: {list(next(iter(sample_patient.modalities.values())).shape) if sample_patient.modalities else 'N/A'}")
        print(f"Data valid: {sample_patient.validate_data()}")

    # Compute statistics for training data
    train_patient_data = []
    for patient_id in tqdm(train_patients[:50], desc="Computing statistics"):  # Limit to 50 for demo
        patient_data = train_ingestor.load_patient_data(patient_id)
        train_patient_data.append(patient_data)

    # Save training metadata
    train_metadata_path = paths.DATA_ROOT_TRAIN / "processed" / "dataset_metadata_train.json"
    train_ingestor.save_metadata(train_patient_data, train_metadata_path)

    # Process Validation Data
    print("\nProcessing Validation Data:")
    val_ingestor = BraTSDataIngestion(paths.DATA_ROOT_VAL)
    val_patients = val_ingestor.discover_patients()

    # Load sample validation patient
    if val_patients:
        sample_patient = val_ingestor.load_patient_data(val_patients[0])
        print(f"Loading sample validation patient: {sample_patient.patient_id}")
        print(f"Patient ID: {sample_patient.patient_id}")
        print(f"Modalities: {list(sample_patient.modalities.keys())}")
        print(f"Has segmentation: {sample_patient.has_segmentation}")
        print(f"Image shape: {list(next(iter(sample_patient.modalities.values())).shape) if sample_patient.modalities else 'N/A'}")
        print(f"Data valid: {sample_patient.validate_data()}")

    # Compute statistics for validation data
    val_patient_data = []
    for patient_id in tqdm(val_patients[:50], desc="Computing statistics"):  # Limit to 50 for demo
        patient_data = val_ingestor.load_patient_data(patient_id)
        val_patient_data.append(patient_data)

    # Save validation metadata
    val_metadata_path = paths.DATA_ROOT_VAL / "processed" / "dataset_metadata_val.json"
    val_ingestor.save_metadata(val_patient_data, val_metadata_path)

if __name__ == "__main__":
    main()

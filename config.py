"""
MedRAG-X Configuration File
VS Code compatible | No API keys required
"""

import os
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple



# =========================
# PATH CONFIGURATION
# =========================

@dataclass
class PathConfig:
    """File paths configuration"""

    # Project root (safe for VS Code)
    ROOT_DIR: Path = Path(__file__).resolve().parent

    # 🔴 CHANGE THIS PATH TO YOUR BraTS DATASET LOCATION
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

    # 🔕 LLM DISABLED (No API keys)
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

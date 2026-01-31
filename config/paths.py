# config/paths.py

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Paths:
    """
    Centralized filesystem paths for the project.
    """

    # project root = folder containing `config/`
    PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

    # -----------------------
    # Data
    # -----------------------
    DATA_ROOT_TRAIN: Path = PROJECT_ROOT / "data" / "train"
    DATA_ROOT_VAL: Path = PROJECT_ROOT / "data" / "val"

    PROCESSED_DATA_DIR: Path = PROJECT_ROOT / "data" / "processed"

    # -----------------------
    # Artifacts
    # -----------------------
    EMBEDDINGS_DIR: Path = PROJECT_ROOT / "data" / "embeddings"
    VECTOR_DB_DIR: Path = PROJECT_ROOT / "vector_db"

    # -----------------------
    # Models
    # -----------------------
    SEGMENTATION_MODEL_DIR: Path = PROJECT_ROOT / "models" / "segmentation"


# instantiate singleton
paths = Paths()
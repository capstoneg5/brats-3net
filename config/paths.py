from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class PathConfig:
    PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
    DATA_ROOT: Path = PROJECT_ROOT / "data"
    DATA_ROOT_TRAIN: Path = DATA_ROOT / "BraTS2020_TrainingData"
    DATA_ROOT_VAL: Path = DATA_ROOT / "BraTS2020_ValidationData"
    PROCESSED_DATA_DIR: Path = DATA_ROOT / "processed"
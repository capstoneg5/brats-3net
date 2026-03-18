# config/brats.py
from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class BratsConfig:
    BRATS_MODALITIES: List[str] = ("t1", "t2", "flair", "t1ce")
    SEG_KEY: str = "seg"

    BRATS_LABELS = {
        0: "background",
        1: "necrotic/core",
        2: "edema",
        4: "enhancing_tumor",
    }


brats_config = BratsConfig()
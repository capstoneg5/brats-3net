from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional


class Modality(str, Enum):
    TEXT = "text"
    MRI = "mri"
    HYBRID = "hybrid"


@dataclass
class ModalityDecision:
    modality: Modality
    text: Optional[str] = None
    mri_path: Optional[Path] = None


def detect_modality(text: Optional[str], mri_path: Optional[str]) -> ModalityDecision:
    p = Path(mri_path) if mri_path else None
    has_text = bool(text and text.strip())
    has_mri = bool(p and p.exists())

    if has_text and has_mri:
        return ModalityDecision(Modality.HYBRID, text=text, mri_path=p)
    if has_mri:
        return ModalityDecision(Modality.MRI, text=None, mri_path=p)
    return ModalityDecision(Modality.TEXT, text=text, mri_path=None)
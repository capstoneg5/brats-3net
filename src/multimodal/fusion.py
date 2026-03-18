from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
import numpy as np


@dataclass
class FusedQuery:
    vector: List[float]
    dim: int
    strategy: str


def l2_normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v) + 1e-8
    return v / n


def fuse_vectors(
    text_vec: Optional[List[float]],
    image_vec: Optional[List[float]],
    w_text: float = 0.5,
    w_image: float = 0.5,
) -> FusedQuery:
    if text_vec is None and image_vec is None:
        raise ValueError("Nothing to fuse")

    if text_vec is None:
        v = l2_normalize(np.array(image_vec, dtype="float32"))
        return FusedQuery(v.tolist(), int(v.shape[0]), "image_only")

    if image_vec is None:
        v = l2_normalize(np.array(text_vec, dtype="float32"))
        return FusedQuery(v.tolist(), int(v.shape[0]), "text_only")

    t = np.array(text_vec, dtype="float32")
    i = np.array(image_vec, dtype="float32")

    # If dims differ, you must project one side. For now require same dim.
    if t.shape[0] != i.shape[0]:
        raise ValueError(f"Dim mismatch text={t.shape[0]} image={i.shape[0]}")

    v = l2_normalize(w_text * t + w_image * i)
    return FusedQuery(v.tolist(), int(v.shape[0]), f"late_fusion_wt={w_text},wi={w_image}")
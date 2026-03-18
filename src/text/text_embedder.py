from __future__ import annotations
from dataclasses import dataclass
from typing import List
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None


@dataclass
class TextEmbeddingResult:
    vector: List[float]
    dim: int
    model: str


class TextEmbedder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        if SentenceTransformer is None:
            raise RuntimeError("sentence-transformers not installed. pip install sentence-transformers")
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed(self, text: str) -> TextEmbeddingResult:
        v = self.model.encode([text], normalize_embeddings=True)[0]
        v = np.asarray(v, dtype="float32")
        return TextEmbeddingResult(vector=v.tolist(), dim=int(v.shape[0]), model=self.model_name)
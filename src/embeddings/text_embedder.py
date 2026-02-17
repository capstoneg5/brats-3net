# src/embeddings/text_embedder.py
"""
Clinical Text Embedder for MedRAG-X
====================================
Encodes clinical / radiology text into dense 768-d embeddings
aligned with the image embedding space.

Uses a biomedical sentence transformer (PubMedBERT-based) by default.
Falls back to a lightweight model if the biomedical one isn't available.

Usage:
    from src.embeddings.text_embedder import ClinicalTextEmbedder

    embedder = ClinicalTextEmbedder()
    vec = embedder.embed("Large enhancing lesion in left temporal lobe")
    # vec.shape == (768,)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Model priority list (first available wins)
_MODEL_CANDIDATES = [
    "pritamdeka/S-PubMedBert-MS-MARCO",          # biomedical, 768-d
    "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract",  # PubMedBERT
    "sentence-transformers/all-MiniLM-L6-v2",     # lightweight fallback, 384-d
]

DESIRED_DIM = 768  # match image embedder output


@dataclass
class TextEmbeddingResult:
    embedding: np.ndarray   # (DESIRED_DIM,) float32
    text: str
    model_name: str


class ClinicalTextEmbedder:
    """
    Wraps a HuggingFace sentence-transformer for clinical text encoding.

    If the loaded model's hidden dim != 768, a learned linear projection
    is replaced by a fixed random projection (deterministic seed) so that
    the output always has 768 dimensions matching the image embedder.
    """

    def __init__(self, model_name: Optional[str] = None, device: str = "cpu"):
        self._model = None
        self._model_name: str = ""
        self._proj: Optional[np.ndarray] = None
        self._native_dim: int = 0
        self._device = device

        if model_name:
            self._load_model(model_name)
        else:
            for candidate in _MODEL_CANDIDATES:
                try:
                    self._load_model(candidate)
                    break
                except Exception as e:
                    logger.info(f"Model {candidate} not available: {e}")

        if self._model is None:
            logger.warning(
                "No sentence-transformer model available. "
                "Using deterministic hash-based fallback embedder."
            )
            self._model_name = "fallback_hash"

    # ───────────── internal ─────────────

    def _load_model(self, name: str) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers is required. "
                "Install: pip install sentence-transformers --break-system-packages"
            )

        self._model = SentenceTransformer(name, device=self._device)
        self._model_name = name
        # determine native dim from a dummy encode
        dummy = self._model.encode(["test"], convert_to_numpy=True)
        self._native_dim = dummy.shape[1]
        logger.info(f"Loaded text model: {name} (dim={self._native_dim})")

        # build projection if dims don't match
        if self._native_dim != DESIRED_DIM:
            rng = np.random.default_rng(42)
            self._proj = rng.normal(
                0, 1.0 / np.sqrt(self._native_dim),
                size=(DESIRED_DIM, self._native_dim),
            ).astype(np.float32)
            logger.info(
                f"  → projection {self._native_dim} → {DESIRED_DIM} enabled"
            )

    def _normalize(self, v: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(v) + 1e-8
        return (v / norm).astype(np.float32)

    def _fallback_embed(self, text: str) -> np.ndarray:
        """Deterministic hash-based embedding (no model needed)."""
        import hashlib
        h = hashlib.sha256(text.encode("utf-8")).digest()
        seed = int.from_bytes(h[:4], "little")
        rng = np.random.default_rng(seed)
        vec = rng.normal(0, 1, size=(DESIRED_DIM,)).astype(np.float32)
        return self._normalize(vec)

    # ───────────── public API ─────────────

    def embed(self, text: str) -> TextEmbeddingResult:
        """Embed a single text string → 768-d normalized vector."""
        if self._model is None:
            vec = self._fallback_embed(text)
        else:
            raw = self._model.encode(
                [text], convert_to_numpy=True, normalize_embeddings=True
            )[0].astype(np.float32)

            if self._proj is not None:
                raw = (self._proj @ raw).astype(np.float32)

            vec = self._normalize(raw)

        return TextEmbeddingResult(
            embedding=vec,
            text=text,
            model_name=self._model_name,
        )

    def embed_batch(self, texts: List[str], batch_size: int = 32) -> List[TextEmbeddingResult]:
        """Embed a batch of texts."""
        if self._model is None:
            return [self.embed(t) for t in texts]

        raw = self._model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            batch_size=batch_size,
        ).astype(np.float32)

        results = []
        for i, text in enumerate(texts):
            vec = raw[i]
            if self._proj is not None:
                vec = (self._proj @ vec).astype(np.float32)
            vec = self._normalize(vec)
            results.append(TextEmbeddingResult(
                embedding=vec, text=text, model_name=self._model_name,
            ))
        return results

    def embed_lesion_description(
        self,
        patient_id: str,
        lesion_id: str,
        wt_vox: int = 0,
        tc_vox: int = 0,
        et_vox: int = 0,
        et_pct: float = 0.0,
        tc_pct: float = 0.0,
        extra: str = "",
    ) -> TextEmbeddingResult:
        """
        Build a clinical text description from structured lesion data
        and embed it. This creates the text-side of the multimodal pair.
        """
        desc = (
            f"Patient {patient_id}, {lesion_id}: "
            f"brain tumor with whole tumor volume {wt_vox} voxels, "
            f"tumor core {tc_vox} voxels ({tc_pct*100:.1f}%), "
            f"enhancing tumor {et_vox} voxels ({et_pct*100:.1f}%). "
        )
        if et_pct > 0.30:
            desc += "Significant enhancing component suggesting active tumor. "
        elif et_pct < 0.05:
            desc += "Minimal enhancement suggesting low-grade or necrotic lesion. "

        if extra:
            desc += extra

        return self.embed(desc.strip())


# ───────────── CLI test ─────────────
def main():
    import argparse

    ap = argparse.ArgumentParser(description="Test clinical text embedder")
    ap.add_argument("--text", default="Large enhancing glioma in left temporal lobe with surrounding edema")
    args = ap.parse_args()

    embedder = ClinicalTextEmbedder()
    result = embedder.embed(args.text)

    print(f"Model:  {result.model_name}")
    print(f"Text:   {result.text}")
    print(f"Shape:  {result.embedding.shape}")
    print(f"Norm:   {np.linalg.norm(result.embedding):.6f}")
    print(f"First5: {result.embedding[:5]}")

    # test lesion description
    result2 = embedder.embed_lesion_description(
        patient_id="BraTS20_Training_001",
        lesion_id="lesion3",
        wt_vox=211979,
        tc_vox=43185,
        et_vox=27742,
        et_pct=0.1309,
        tc_pct=0.2037,
    )
    print(f"\nLesion desc embedding shape: {result2.embedding.shape}")
    print(f"Cosine sim (text vs lesion): {float(result.embedding @ result2.embedding):.4f}")


if __name__ == "__main__":
    main()


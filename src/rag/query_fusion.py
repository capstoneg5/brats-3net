# src/rag/query_fusion.py
"""
Query Fusion Module for MedRAG-X
=================================
Merges text-based and image-based query embeddings into a single
fused query vector for hybrid multimodal retrieval.

Architecture (from the diagram):
  ┌─────────────────┐    ┌──────────────────────┐
  │  Text Embedding  │    │  Image Embedding     │
  │ (Clinical Tuned) │    │ (MRI 3D Slice/Cube)  │
  └────────┬────────┘    └──────────┬───────────┘
           │                        │
           └──────────┬─────────────┘
                      ▼
               ┌──────────────┐
               │ Query Fusion │
               └──────┬───────┘
                      ▼
          ┌────────────────────────┐
          │ Structural + Semantic  │
          │     Retrieval          │
          └────────────────────────┘

Fusion strategies:
  1. weighted_average  – α * text_emb + (1-α) * image_emb  (default)
  2. concat_project    – project(concat(text, image)) → 768-d
  3. text_only         – use text embedding alone
  4. image_only        – use image embedding alone (current behaviour)

Usage:
    from src.rag.query_fusion import QueryFusionEngine, FusionConfig

    engine = QueryFusionEngine(FusionConfig(strategy="weighted_average", alpha=0.3))
    fused = engine.fuse(text_emb=text_vec, image_emb=image_vec)
    # fused.shape == (768,)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class FusionStrategy(str, Enum):
    WEIGHTED_AVERAGE = "weighted_average"
    CONCAT_PROJECT = "concat_project"
    TEXT_ONLY = "text_only"
    IMAGE_ONLY = "image_only"


@dataclass
class FusionConfig:
    strategy: str = "weighted_average"
    alpha: float = 0.3          # weight for text embedding (0=image only, 1=text only)
    output_dim: int = 768       # final embedding dimension
    normalize: bool = True      # L2-normalize the fused vector


@dataclass
class FusedQuery:
    """Result of query fusion."""
    embedding: np.ndarray       # (output_dim,) float32
    strategy: str
    alpha_used: float
    text_present: bool
    image_present: bool
    metadata: Dict[str, Any] = field(default_factory=dict)


class QueryFusionEngine:
    """
    Fuses text and image query embeddings into a single retrieval vector.
    """

    def __init__(self, config: Optional[FusionConfig] = None):
        self.config = config or FusionConfig()
        self._proj: Optional[np.ndarray] = None  # for concat_project

        if self.config.strategy == FusionStrategy.CONCAT_PROJECT:
            # Deterministic random projection: 2*dim → dim
            rng = np.random.default_rng(seed=12345)
            input_dim = 2 * self.config.output_dim
            self._proj = rng.normal(
                0, 1.0 / np.sqrt(input_dim),
                size=(self.config.output_dim, input_dim),
            ).astype(np.float32)

    @staticmethod
    def _normalize(v: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(v) + 1e-8
        return (v / norm).astype(np.float32)

    def fuse(
        self,
        text_emb: Optional[np.ndarray] = None,
        image_emb: Optional[np.ndarray] = None,
        alpha_override: Optional[float] = None,
    ) -> FusedQuery:
        """
        Fuse text and image embeddings.

        If only one modality is available, returns that embedding
        regardless of the configured strategy.

        Args:
            text_emb:  (D,) float32 text embedding or None
            image_emb: (D,) float32 image embedding or None
            alpha_override: override config alpha for this call

        Returns:
            FusedQuery with fused embedding
        """
        alpha = alpha_override if alpha_override is not None else self.config.alpha
        strategy = self.config.strategy

        has_text = text_emb is not None and text_emb.size > 0
        has_image = image_emb is not None and image_emb.size > 0

        # Handle missing modalities gracefully
        if not has_text and not has_image:
            raise ValueError("At least one of text_emb or image_emb must be provided")

        if not has_text:
            vec = image_emb.astype(np.float32)
            strategy_used = "image_only_fallback"
            alpha_used = 0.0
        elif not has_image:
            vec = text_emb.astype(np.float32)
            strategy_used = "text_only_fallback"
            alpha_used = 1.0
        else:
            # Both modalities present — apply fusion strategy
            t = text_emb.astype(np.float32)
            im = image_emb.astype(np.float32)

            if strategy == FusionStrategy.TEXT_ONLY:
                vec = t
                strategy_used = "text_only"
                alpha_used = 1.0

            elif strategy == FusionStrategy.IMAGE_ONLY:
                vec = im
                strategy_used = "image_only"
                alpha_used = 0.0

            elif strategy == FusionStrategy.CONCAT_PROJECT:
                # Pad/truncate to output_dim if needed
                t = self._pad_or_truncate(t, self.config.output_dim)
                im = self._pad_or_truncate(im, self.config.output_dim)
                concat = np.concatenate([t, im])
                vec = (self._proj @ concat).astype(np.float32)
                strategy_used = "concat_project"
                alpha_used = alpha

            else:  # weighted_average (default)
                t = self._pad_or_truncate(t, self.config.output_dim)
                im = self._pad_or_truncate(im, self.config.output_dim)
                vec = (alpha * t + (1.0 - alpha) * im).astype(np.float32)
                strategy_used = "weighted_average"
                alpha_used = alpha

        # Normalize if configured
        if self.config.normalize:
            vec = self._normalize(vec)

        # Ensure correct output dimension
        vec = self._pad_or_truncate(vec, self.config.output_dim)

        return FusedQuery(
            embedding=vec,
            strategy=strategy_used,
            alpha_used=alpha_used,
            text_present=has_text,
            image_present=has_image,
            metadata={
                "config_strategy": self.config.strategy,
                "config_alpha": self.config.alpha,
            },
        )

    def _pad_or_truncate(self, v: np.ndarray, dim: int) -> np.ndarray:
        """Ensure vector is exactly `dim` dimensions."""
        if v.shape[0] == dim:
            return v
        if v.shape[0] > dim:
            return v[:dim]
        pad = np.zeros(dim - v.shape[0], dtype=np.float32)
        return np.concatenate([v, pad])

    def fuse_for_retrieval(
        self,
        query_text: Optional[str] = None,
        lesion_id: Optional[str] = None,
        text_embedder=None,
        kg_or_retriever=None,
    ) -> FusedQuery:
        """
        High-level convenience: takes raw query text + lesion ID,
        produces a fused query embedding ready for vector search.

        Args:
            query_text:     natural language query string
            lesion_id:      lesion ID to look up image embedding from KG
            text_embedder:  ClinicalTextEmbedder instance
            kg_or_retriever: object with a way to get lesion embedding
        """
        text_emb = None
        image_emb = None

        # Get text embedding
        if query_text and text_embedder is not None:
            result = text_embedder.embed(query_text)
            text_emb = result.embedding

        # Get image embedding from KG
        if lesion_id and kg_or_retriever is not None:
            image_emb = self._get_image_emb_from_source(lesion_id, kg_or_retriever)

        return self.fuse(text_emb=text_emb, image_emb=image_emb)

    @staticmethod
    def _get_image_emb_from_source(lesion_id: str, source) -> Optional[np.ndarray]:
        """
        Try to extract image embedding from either a KG or Neo4j retriever.
        """
        # Try NetworkX KG
        if hasattr(source, "graph"):
            node_data = source.graph.nodes.get(lesion_id, {})
            emb = node_data.get("embedding")
            if emb is not None:
                return np.asarray(emb, dtype=np.float32)

        # Try Neo4j retriever
        if hasattr(source, "driver"):
            try:
                with source.driver.session() as s:
                    row = s.run(
                        "MATCH (l:Lesion {id: $id}) RETURN l.embedding AS emb",
                        id=lesion_id,
                    ).single()
                    if row and row["emb"]:
                        return np.asarray(row["emb"], dtype=np.float32)
            except Exception as e:
                logger.warning(f"Could not fetch embedding for {lesion_id}: {e}")

        return None


# ───────────── CLI demo ─────────────
def main():
    """Demonstrate fusion strategies."""
    rng = np.random.default_rng(0)
    text_emb = rng.normal(0, 1, (768,)).astype(np.float32)
    text_emb /= np.linalg.norm(text_emb)
    image_emb = rng.normal(0, 1, (768,)).astype(np.float32)
    image_emb /= np.linalg.norm(image_emb)

    cos = float(text_emb @ image_emb)
    print(f"Cosine(text, image) = {cos:.4f}")

    for strategy in ["weighted_average", "concat_project", "text_only", "image_only"]:
        engine = QueryFusionEngine(FusionConfig(strategy=strategy, alpha=0.3))
        result = engine.fuse(text_emb=text_emb, image_emb=image_emb)
        cos_text = float(result.embedding @ text_emb)
        cos_image = float(result.embedding @ image_emb)
        print(
            f"  {strategy:20s} | cos(fused,text)={cos_text:.4f} "
            f"| cos(fused,image)={cos_image:.4f} "
            f"| alpha={result.alpha_used:.2f}"
        )

    # Test fallback: only one modality
    engine = QueryFusionEngine(FusionConfig(strategy="weighted_average", alpha=0.3))
    result = engine.fuse(text_emb=text_emb, image_emb=None)
    print(f"\n  text_only fallback   | strategy={result.strategy}")

    result = engine.fuse(text_emb=None, image_emb=image_emb)
    print(f"  image_only fallback  | strategy={result.strategy}")


if __name__ == "__main__":
    main()


# retrieval/index_build.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from loguru import logger

from config import paths, model_config
from retrival.vector_store import create_vector_store


def load_embeddings_jsonl(jsonl_path: Path) -> Tuple[List[str], np.ndarray, List[Dict[str, Any]]]:
    ids: List[str] = []
    metas: List[Dict[str, Any]] = []
    vecs: List[np.ndarray] = []

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                _id = obj["id"]
                emb = np.array(obj["embeddings"], dtype=np.float32)
                meta = obj.get("metadata", {})
            except Exception as e:
                raise ValueError(f"Invalid JSONL at line {line_no}: {e}")

            if emb.ndim != 1:
                raise ValueError(f"Embedding must be 1D at line {line_no}, got shape {emb.shape}")

            ids.append(_id)
            vecs.append(emb)
            metas.append(meta)

    if not vecs:
        return [], np.zeros((0, model_config.EMBEDDING_DIM), dtype=np.float32), []

    mat = np.stack(vecs, axis=0).astype(np.float32)
    return ids, mat, metas


def main() -> None:
    parser = argparse.ArgumentParser(description="Build vector index from embeddings JSONL.")
    parser.add_argument("--backend", choices=["faiss", "chroma"], default="faiss")
    parser.add_argument("--metric", choices=["cosine", "l2"], default="cosine")
    parser.add_argument("--collection", default="medragx")
    parser.add_argument("--embeddings", type=str, required=True, help="Path to embeddings JSONL file")
    parser.add_argument("--persist_dir", type=str, default=str(paths.VECTOR_DB_DIR), help="Index output directory")

    args = parser.parse_args()

    jsonl_path = Path(args.embeddings)
    if not jsonl_path.exists():
        raise FileNotFoundError(f"Embeddings JSONL not found: {jsonl_path}")

    ids, vectors, metas = load_embeddings_jsonl(jsonl_path)
    logger.info(f"Loaded {len(ids)} embeddings from {jsonl_path} | shape={vectors.shape}")

    store = create_vector_store(
        backend=args.backend,
        persist_dir=args.persist_dir,
        dim=vectors.shape[1] if vectors.size else model_config.EMBEDDING_DIM,
        metric=args.metric,
        collection_name=args.collection,
    )

    if len(ids) == 0:
        logger.warning("No embeddings found. Nothing to index.")
        return

    # Chunk ingestion for memory safety
    batch = 512
    for i in range(0, len(ids), batch):
        store.add(ids=ids[i:i+batch], vectors=vectors[i:i+batch], metadatas=metas[i:i+batch])
        logger.info(f"Indexed {min(i+batch, len(ids))}/{len(ids)}")

    store.persist()
    logger.info(f"✅ Index built successfully. Total vectors in store: {store.size}")


if __name__ == "__main__":
    main()

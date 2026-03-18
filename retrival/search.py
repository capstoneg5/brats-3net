# retrieval/search.py
from __future__ import annotations

import argparse
from typing import Any, Dict, Optional

from loguru import logger

from config import paths, model_config
from retrival.vector_store import create_vector_store
from models.embeddings.text_embedder import TextEmbedder


def parse_filter(filter_str: Optional[str]) -> Optional[Dict[str, Any]]:
    """
    Accepts a simple JSON dict as string:
      --filter '{"type":"lesion_3d"}'
    """
    if not filter_str:
        return None
    import json
    return json.loads(filter_str)


def main() -> None:
    parser = argparse.ArgumentParser(description="Search vector store using text query.")
    parser.add_argument("--backend", choices=["faiss", "chroma"], default="faiss")
    parser.add_argument("--metric", choices=["cosine", "l2"], default="cosine")
    parser.add_argument("--collection", default="medragx")
    parser.add_argument("--persist_dir", default=str(paths.VECTOR_DB_DIR))
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--filter", type=str, default=None, help='JSON filter, e.g. \'{"type":"lesion_3d"}\'')
    parser.add_argument("--query", type=str, required=True, help="Text query for semantic search")
    args = parser.parse_args()

    # Load store
    store = create_vector_store(
        backend=args.backend,
        persist_dir=args.persist_dir,
        dim=model_config.EMBEDDING_DIM,
        metric=args.metric,
        collection_name=args.collection,
    )

    if store.size == 0:
        logger.warning("Vector store is empty. Build index first using index_build.py.")
        return

    # Embed query text (PubMedBERT)
    embedder = TextEmbedder(model_name=model_config.TEXT_EMBEDDING_MODEL)
    q_emb = embedder.embed(args.query).embeddings[0]  # (768,)

    filt = parse_filter(args.filter)
    results = store.search(query_vector=q_emb, top_k=args.top_k, filter=filt)

    print("\n====================")
    print("Query:", args.query)
    print("TopK:", args.top_k)
    print("Filter:", filt)
    print("====================\n")

    for rank, r in enumerate(results, start=1):
        meta = r.metadata or {}
        print(f"{rank:02d}. id={r.id}  score={r.score:.4f}")
        # Print key metadata fields if present
        keys = ["type", "patient_id", "modality", "tumor_volume_voxels", "bbox", "centroid_zyx"]
        compact = {k: meta.get(k) for k in keys if k in meta}
        if compact:
            print("    meta:", compact)
        else:
            print("    meta:", meta)

    print("\nDone.\n")


if __name__ == "__main__":
    main()

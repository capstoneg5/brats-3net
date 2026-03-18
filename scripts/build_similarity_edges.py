# scripts/build_similarity_edges.py
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np

from src.graph.knowledge_graph import ClinicalKnowledgeGraph


def _collect_lesions_with_embeddings(
    kg: ClinicalKnowledgeGraph,
) -> Tuple[List[str], np.ndarray]:
    """
    Returns:
      lesion_ids: list[str]
      E: float32 array [N, D]
    """
    lesion_ids: List[str] = []
    embs: List[np.ndarray] = []

    for node_id, attrs in kg.graph.nodes(data=True):
        if attrs.get("type") != "Lesion":
            continue
        emb = attrs.get("embedding", None)
        if emb is None:
            continue
        # embedding is stored as list[float] in your KG json
        v = np.asarray(emb, dtype=np.float32).reshape(-1)
        if v.size == 0:
            continue

        lesion_ids.append(node_id)
        embs.append(v)

    if not embs:
        raise RuntimeError("No lesion embeddings found in the KG.")

    # ensure consistent dimension
    dim0 = embs[0].shape[0]
    for i, v in enumerate(embs):
        if v.shape[0] != dim0:
            raise ValueError(
                f"Embedding dim mismatch at {lesion_ids[i]}: got {v.shape[0]}, expected {dim0}"
            )

    E = np.stack(embs, axis=0)  # [N, D]
    return lesion_ids, E


def _l2_normalize(E: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norms = np.linalg.norm(E, axis=1, keepdims=True)
    return E / (norms + eps)


def add_similarity_edges(
    kg: ClinicalKnowledgeGraph,
    lesion_ids: List[str],
    E: np.ndarray,
    top_k: int = 10,
    min_score: float = 0.0,
    bidirectional: bool = True,
) -> int:
    """
    Adds edges:
      lesion_i --(lesion_similar_to, score=...)-> lesion_j

    Returns: number of edges added
    """
    # cosine similarity via dot product after normalization
    En = _l2_normalize(E)
    S = En @ En.T  # [N, N]
    np.fill_diagonal(S, -1.0)  # exclude self

    added = 0
    N = S.shape[0]

    for i in range(N):
        # top-k indices for row i
        # argpartition is faster than full sort
        k = min(top_k, N - 1)
        idx = np.argpartition(-S[i], kth=k - 1)[:k]
        # sort those k by score desc
        idx = idx[np.argsort(-S[i, idx])]

        src = lesion_ids[i]
        for j in idx:
            score = float(S[i, j])
            if score < min_score:
                continue

            dst = lesion_ids[j]

            kg.graph.add_edge(
                src,
                dst,
                relationship="lesion_similar_to",
                score=score,
                metric="cosine",
            )
            added += 1

            if bidirectional:
                kg.graph.add_edge(
                    dst,
                    src,
                    relationship="lesion_similar_to",
                    score=score,
                    metric="cosine",
                )
                added += 1

    return added


def main():
    kg_in = Path("artifacts/kg_3d.json")
    kg_out = Path("artifacts/kg_3d_with_sim.json")

    kg = ClinicalKnowledgeGraph.load(kg_in)
    lesion_ids, E = _collect_lesions_with_embeddings(kg)

    print(f"Loaded KG: {kg_in}")
    print(f"Lesions with embeddings: {len(lesion_ids)}")
    print(f"Embedding dim: {E.shape[1]}")

    added = add_similarity_edges(
        kg,
        lesion_ids,
        E,
        top_k=10,        # change as needed
        min_score=0.0,   # optionally set like 0.95 to reduce edges
        bidirectional=True,
    )

    kg.save(kg_out)
    print(f"\n✅ Saved KG with similarity edges: {kg_out}")
    print(f"Edges added: {added}")


if __name__ == "__main__":
    main()

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List

from src.rag.neo4j_retriever import Neo4jRetriever


# ---------- Evidence item ----------
@dataclass
class RetrievalItem:
    rank: int
    lesion_id: str
    patient_id: str
    score: float
    facts: Dict[str, Any]


# ---------- Vector retrieve ----------
def vector_retrieve(uri: str, user: str, password: str, lesion_id: str, top_k: int) -> List[RetrievalItem]:
    retriever = Neo4jRetriever(uri, user, password)
    rows = retriever.retrieve_similar(lesion_id, k=top_k)
    retriever.close()

    items: List[RetrievalItem] = []
    for i, r in enumerate(rows, 1):
        items.append(
            RetrievalItem(
                rank=i,
                lesion_id=r["lesion_id"],
                patient_id=r["patient_id"],
                score=float(r["score"]),
                facts=r,
            )
        )
    return items


# ---------- KG retrieve (stub for now) ----------
def kg_retrieve(uri: str, user: str, password: str, lesion_id: str) -> List[RetrievalItem]:
    # Future: real clinical KG retrieval
    return []


# ---------- Fusion ----------
def fuse_and_rerank(vec_items: List[RetrievalItem], kg_items: List[RetrievalItem]) -> List[RetrievalItem]:
    # Simple first version: vector only
    return sorted(vec_items + kg_items, key=lambda x: x.score, reverse=True)
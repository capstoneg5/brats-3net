# retrieval/vector_store.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
from loguru import logger

# Optional imports (backend selection)
try:
    import faiss  # type: ignore
except Exception:  # pragma: no cover
    faiss = None

try:
    import chromadb  # type: ignore
    from chromadb.config import Settings  # type: ignore
except Exception:  # pragma: no cover
    chromadb = None
    Settings = None


JsonDict = Dict[str, Any]


@dataclass
class SearchResult:
    id: str
    score: float
    metadata: JsonDict


def _l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(x, axis=-1, keepdims=True) + eps
    return x / norm


def _ensure_2d_float32(vecs: Union[np.ndarray, Sequence[np.ndarray]]) -> np.ndarray:
    if isinstance(vecs, np.ndarray):
        arr = vecs
    else:
        arr = np.stack(vecs, axis=0)
    if arr.ndim == 1:
        arr = arr[None, :]
    return arr.astype(np.float32, copy=False)


class VectorStore:
    """
    Abstract Vector Store Interface.
    """

    def add(self, ids: List[str], vectors: np.ndarray, metadatas: List[JsonDict]) -> None:
        raise NotImplementedError

    def search(self, query_vector: np.ndarray, top_k: int = 10, filter: Optional[JsonDict] = None) -> List[SearchResult]:
        raise NotImplementedError

    def persist(self) -> None:
        raise NotImplementedError

    @property
    def size(self) -> int:
        raise NotImplementedError


# -------------------------------------------------------
# FAISS BACKEND
# -------------------------------------------------------
class FaissVectorStore(VectorStore):
    """
    FAISS-based vector store with metadata persistence.
    - Uses cosine similarity by default (via L2-normalization + inner product).
    - Persists:
        index.faiss
        meta.jsonl (id + metadata)
    """

    def __init__(
        self,
        persist_dir: Union[str, Path],
        dim: int,
        metric: str = "cosine",  # "cosine" or "l2"
        use_gpu: bool = False,   # left for future (faiss-gpu)
    ) -> None:
        if faiss is None:
            raise ImportError("faiss is not installed. Install faiss-cpu or faiss-gpu.")

        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        self.dim = int(dim)
        self.metric = metric.lower().strip()
        self.use_gpu = use_gpu

        self.index_path = self.persist_dir / "index.faiss"
        self.meta_path = self.persist_dir / "meta.jsonl"

        self._id_to_meta: Dict[str, JsonDict] = {}
        self._ids: List[str] = []

        self._index = self._load_or_create_index()

    def _load_or_create_index(self):
        if self.index_path.exists():
            logger.info(f"Loading FAISS index from {self.index_path}")
            idx = faiss.read_index(str(self.index_path))
            if idx.d != self.dim:
                raise ValueError(f"Index dim mismatch: expected {self.dim}, got {idx.d}")
            self._load_metadata()
            return idx

        logger.info(f"Creating new FAISS index | dim={self.dim} metric={self.metric}")
        if self.metric == "cosine":
            # cosine similarity = inner product on normalized vectors
            idx = faiss.IndexFlatIP(self.dim)
        elif self.metric == "l2":
            idx = faiss.IndexFlatL2(self.dim)
        else:
            raise ValueError("metric must be 'cosine' or 'l2'")
        return idx

    def _load_metadata(self) -> None:
        if not self.meta_path.exists():
            return
        logger.info(f"Loading metadata from {self.meta_path}")
        self._id_to_meta.clear()
        self._ids.clear()
        with self.meta_path.open("r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                _id = obj["id"]
                self._ids.append(_id)
                self._id_to_meta[_id] = obj.get("metadata", {})

    @property
    def size(self) -> int:
        return len(self._ids)

    def add(self, ids: List[str], vectors: np.ndarray, metadatas: List[JsonDict]) -> None:
        if len(ids) != len(metadatas):
            raise ValueError("ids and metadatas length mismatch")

        vecs = _ensure_2d_float32(vectors)
        if vecs.shape[1] != self.dim:
            raise ValueError(f"Vector dim mismatch: expected {self.dim}, got {vecs.shape[1]}")

        if self.metric == "cosine":
            vecs = _l2_normalize(vecs)

        # Ensure no duplicate IDs (simple production behavior)
        for _id, meta in zip(ids, metadatas):
            if _id in self._id_to_meta:
                raise ValueError(f"Duplicate id detected: {_id}")

        self._index.add(vecs)

        # Store metadata in memory and append to disk (write-ahead log style)
        with self.meta_path.open("a", encoding="utf-8") as f:
            for _id, meta in zip(ids, metadatas):
                self._ids.append(_id)
                self._id_to_meta[_id] = meta
                f.write(json.dumps({"id": _id, "metadata": meta}, ensure_ascii=False) + "\n")

    def _passes_filter(self, meta: JsonDict, filt: JsonDict) -> bool:
        # simple exact-match filter (extend as needed)
        for k, v in filt.items():
            if meta.get(k) != v:
                return False
        return True

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
        filter: Optional[JsonDict] = None,
    ) -> List[SearchResult]:
        if self.size == 0:
            return []

        q = _ensure_2d_float32(query_vector)
        if q.shape[1] != self.dim:
            raise ValueError(f"Query dim mismatch: expected {self.dim}, got {q.shape[1]}")

        if self.metric == "cosine":
            q = _l2_normalize(q)

        # Retrieve more than top_k when filtering (best-effort)
        fetch_k = min(max(top_k * 5, top_k), self.size)
        scores, idxs = self._index.search(q, fetch_k)  # scores: [1,k], idxs: [1,k]

        results: List[SearchResult] = []
        for score, i in zip(scores[0].tolist(), idxs[0].tolist()):
            if i < 0 or i >= len(self._ids):
                continue
            _id = self._ids[i]
            meta = self._id_to_meta.get(_id, {})

            if filter and not self._passes_filter(meta, filter):
                continue

            results.append(SearchResult(id=_id, score=float(score), metadata=meta))
            if len(results) >= top_k:
                break

        return results

    def persist(self) -> None:
        logger.info(f"Persisting FAISS index to {self.index_path}")
        faiss.write_index(self._index, str(self.index_path))


# -------------------------------------------------------
# CHROMA BACKEND (OPTIONAL)
# -------------------------------------------------------
class ChromaVectorStore(VectorStore):
    """
    ChromaDB vector store.
    Good for richer metadata filtering & easier ops.
    """

    def __init__(
        self,
        persist_dir: Union[str, Path],
        collection_name: str,
        dim: int,
        metric: str = "cosine",
    ) -> None:
        if chromadb is None:
            raise ImportError("chromadb is not installed. Install chromadb.")

        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        self.dim = int(dim)
        self.metric = metric.lower().strip()

        # Local persistent client
        self._client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(allow_reset=False) if Settings is not None else None,
        )

        # Chroma will handle storage; use get_or_create
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine" if self.metric == "cosine" else "l2"},
        )

    @property
    def size(self) -> int:
        return int(self._collection.count())

    def add(self, ids: List[str], vectors: np.ndarray, metadatas: List[JsonDict]) -> None:
        vecs = _ensure_2d_float32(vectors)
        if vecs.shape[1] != self.dim:
            raise ValueError(f"Vector dim mismatch: expected {self.dim}, got {vecs.shape[1]}")

        if self.metric == "cosine":
            vecs = _l2_normalize(vecs)

        self._collection.add(
            ids=ids,
            embeddings=vecs.tolist(),
            metadatas=metadatas,
        )

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
        filter: Optional[JsonDict] = None,
    ) -> List[SearchResult]:
        q = _ensure_2d_float32(query_vector)
        if q.shape[1] != self.dim:
            raise ValueError(f"Query dim mismatch: expected {self.dim}, got {q.shape[1]}")

        if self.metric == "cosine":
            q = _l2_normalize(q)

        res = self._collection.query(
            query_embeddings=q.tolist(),
            n_results=top_k,
            where=filter or None,
            include=["metadatas", "distances"],
        )

        # For cosine in Chroma, "distances" are usually (1 - cosine_sim) depending on config.
        # We'll convert to a similarity-like score: score = -distance
        ids = res.get("ids", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]

        out: List[SearchResult] = []
        for _id, meta, dist in zip(ids, metas, dists):
            out.append(SearchResult(id=_id, score=float(-dist), metadata=meta or {}))
        return out

    def persist(self) -> None:
        # PersistentClient writes automatically
        logger.info("ChromaDB persistence is automatic.")


# -------------------------------------------------------
# Factory
# -------------------------------------------------------
def create_vector_store(
    backend: str,
    persist_dir: Union[str, Path],
    dim: int,
    metric: str = "cosine",
    collection_name: str = "medragx",
) -> VectorStore:
    backend = backend.lower().strip()
    if backend == "faiss":
        return FaissVectorStore(persist_dir=persist_dir, dim=dim, metric=metric)
    if backend == "chroma":
        return ChromaVectorStore(persist_dir=persist_dir, collection_name=collection_name, dim=dim, metric=metric)
    raise ValueError("backend must be 'faiss' or 'chroma'")

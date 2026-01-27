# app/api.py
from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from loguru import logger

from config import paths, model_config
from retrival.vector_store import create_vector_store, SearchResult
from models.embedding.text_embedder import TextEmbedder


JsonDict = Dict[str, Any]


# ----------------------------
# Request / Response Models
# ----------------------------
class HealthResponse(BaseModel):
    status: str
    backend: str
    size: int
    dim: int
    persist_dir: str


class EmbedTextRequest(BaseModel):
    texts: List[str] = Field(..., min_items=1)
    normalize: bool = True
    batch_size: int = 16


class EmbedTextResponse(BaseModel):
    model_name: str
    dim: int
    embeddings: List[List[float]]


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(10, ge=1, le=50)
    filter: Optional[JsonDict] = None
    backend: str = Field("faiss", pattern="^(faiss|chroma)$")
    metric: str = Field("cosine", pattern="^(cosine|l2)$")
    collection: str = "medragx"
    persist_dir: str = str(paths.VECTOR_DB_DIR)


class SearchHit(BaseModel):
    id: str
    score: float
    metadata: JsonDict


class SearchResponse(BaseModel):
    query: str
    top_k: int
    backend: str
    metric: str
    collection: str
    persist_dir: str
    results: List[SearchHit]


# ----------------------------
# App init
# ----------------------------
app = FastAPI(
    title="MedRAG-X API",
    version="1.0.0",
    description="Semantic search API for lesion-centric 3D embeddings (FAISS/Chroma) + PubMedBERT text query embedding",
)

# Cache embedder in-memory (typical production pattern)
_text_embedder: Optional[TextEmbedder] = None


def get_text_embedder() -> TextEmbedder:
    global _text_embedder
    if _text_embedder is None:
        logger.info(f"Loading text embedder: {model_config.TEXT_EMBEDDING_MODEL}")
        _text_embedder = TextEmbedder(model_name=model_config.TEXT_EMBEDDING_MODEL)
    return _text_embedder


def open_store(backend: str, metric: str, collection: str, persist_dir: str):
    return create_vector_store(
        backend=backend,
        persist_dir=persist_dir,
        dim=model_config.EMBEDDING_DIM,
        metric=metric,
        collection_name=collection,
    )


# ----------------------------
# Routes
# ----------------------------
@app.get("/health", response_model=HealthResponse)
def health(
    backend: str = "faiss",
    metric: str = "cosine",
    collection: str = "medragx",
    persist_dir: str = str(paths.VECTOR_DB_DIR),
) -> HealthResponse:
    try:
        store = open_store(backend, metric, collection, persist_dir)
        return HealthResponse(
            status="ok",
            backend=backend,
            size=store.size,
            dim=model_config.EMBEDDING_DIM,
            persist_dir=str(persist_dir),
        )
    except Exception as e:  # noqa: BLE001
        # Health endpoint should not crash; return useful error
        raise HTTPException(status_code=500, detail=f"Health check failed: {e}")


@app.post("/embed/text", response_model=EmbedTextResponse)
def embed_text(req: EmbedTextRequest) -> EmbedTextResponse:
    try:
        embedder = get_text_embedder()
        out = embedder.embed(req.texts, batch_size=req.batch_size)
        return EmbedTextResponse(
            model_name=out.model_name,
            dim=out.embeddings.shape[1] if out.embeddings.size else model_config.EMBEDDING_DIM,
            embeddings=out.embeddings.tolist(),
        )
    except (ValueError, RuntimeError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Embedding failed: {e}")


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest) -> SearchResponse:
    """
    Text query -> PubMedBERT embedding -> vector store search.
    Optionally filter by metadata, e.g. {"type":"lesion_3d"}.
    """
    try:
        store = open_store(req.backend, req.metric, req.collection, req.persist_dir)

        if store.size == 0:
            raise HTTPException(
                status_code=400,
                detail="Vector store is empty. Build index first (index_build.py or pipeline.py --build_index).",
            )

        embedder = get_text_embedder()
        q_vec = embedder.embed(req.query).embeddings[0]  # (768,)

        results: List[SearchResult] = store.search(query_vector=q_vec, top_k=req.top_k, filter=req.filter)

        hits = [SearchHit(id=r.id, score=r.score, metadata=r.metadata or {}) for r in results]

        return SearchResponse(
            query=req.query,
            top_k=req.top_k,
            backend=req.backend,
            metric=req.metric,
            collection=req.collection,
            persist_dir=req.persist_dir,
            results=hits,
        )

    except HTTPException:
        raise
    except (ValueError, FileNotFoundError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Search failed: {e}")

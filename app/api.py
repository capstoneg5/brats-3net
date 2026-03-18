# app/api.py
from __future__ import annotations

import hashlib
import os
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, Field
from loguru import logger
import numpy as np

from config import paths, model_config
from retrival.vector_store import create_vector_store, SearchResult
from models.embeddings.text_embedder import TextEmbedder
from src.guardrails.ethical_guardrails import (
    append_audit_event,
    check_hitl_approval,
    check_purpose_allowed,
    check_role_permission,
    enforce_retention_policy,
    get_ethical_policy,
    minimize_query_text,
)


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
    description="Semantic search API for lesion-centric 3D embeddings (FAISS/Chroma) + PubMedBERT text query embeddings",
)

# Cache embedder in-memory (typical production pattern)
_text_embedder: Optional[Any] = None
ETHICAL_POLICY = get_ethical_policy()
_RETENTION_STATS = enforce_retention_policy(ETHICAL_POLICY)
EMBEDDER_BACKEND = os.getenv("MEDRAGX_EMBEDDER_BACKEND", "hash").strip().lower()


class _HashEmbedder:
    """Deterministic lightweight embedder for local/offline and low-memory execution."""

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.model_name = f"hash-{dim}"

    def _embed_one(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        tokens = (text or "").lower().split()
        if not tokens:
            return vec
        for tok in tokens:
            h = int(hashlib.sha256(tok.encode("utf-8")).hexdigest()[:8], 16)
            idx = h % self.dim
            vec[idx] += 1.0
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    def embed(self, texts, batch_size: int = 16):  # noqa: ARG002
        if isinstance(texts, str):
            texts_list = [texts]
        else:
            texts_list = list(texts)
        arr = np.stack([self._embed_one(t) for t in texts_list], axis=0).astype(np.float32)

        class _Out:
            def __init__(self, embeddings: np.ndarray, model_name: str) -> None:
                self.embeddings = embeddings
                self.model_name = model_name

        return _Out(arr, self.model_name)


def _load_api_keys() -> Dict[str, str]:
    raw = os.getenv("MEDRAGX_API_KEYS", "").strip()
    if not raw:
        return {}

    parsed: Dict[str, str] = {}
    for part in raw.split(","):
        if ":" not in part:
            continue
        key, role = part.split(":", 1)
        key = key.strip()
        role = role.strip().lower()
        if key and role:
            parsed[key] = role
    return parsed


API_KEYS = _load_api_keys()


def _safe_actor_from_key(api_key: str) -> str:
    digest = hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:12]
    return f"key:{digest}"


def _audit(event_type: str, outcome: str, actor: str, role: str, purpose: str, details: Optional[Dict[str, Any]] = None) -> None:
    try:
        append_audit_event(
            event_type=event_type,
            outcome=outcome,
            actor=actor,
            role=role,
            purpose=purpose,
            details=details or {},
            policy=ETHICAL_POLICY,
        )
    except Exception:
        pass


def _parse_bool_header(value: Optional[str]) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _resolve_identity(
    request: Request,
    x_api_key: Optional[str],
    x_role: Optional[str],
    x_purpose: Optional[str],
) -> Dict[str, str]:
    purpose = (x_purpose or "research").strip().lower()

    if API_KEYS:
        if not x_api_key:
            _audit("auth", "deny", actor="unknown", role="unknown", purpose=purpose, details={"reason": "missing_api_key"})
            raise HTTPException(status_code=401, detail="Missing X-API-Key.")
        role = API_KEYS.get(x_api_key)
        if role is None:
            _audit("auth", "deny", actor="unknown", role="unknown", purpose=purpose, details={"reason": "invalid_api_key"})
            raise HTTPException(status_code=401, detail="Invalid API key.")
        actor = _safe_actor_from_key(x_api_key)
        return {"actor": actor, "role": role, "purpose": purpose}

    role = (x_role or os.getenv("MEDRAGX_API_DEFAULT_ROLE", "viewer")).strip().lower()
    actor = request.client.host if request.client else "api_anonymous"
    return {"actor": actor, "role": role, "purpose": purpose}


def _authorize_action(action: str, actor: str, role: str, purpose: str, query_meta: Optional[Dict[str, Any]] = None) -> None:
    purpose_check = check_purpose_allowed(purpose, ETHICAL_POLICY)
    if not purpose_check.allowed:
        _audit(action, "deny", actor=actor, role=role, purpose=purpose, details={"reason": purpose_check.reason, "query_hash": (query_meta or {}).get("sha256")})
        raise HTTPException(status_code=403, detail=purpose_check.reason)

    role_check = check_role_permission(role, action, ETHICAL_POLICY)
    if not role_check.allowed:
        _audit(action, "deny", actor=actor, role=role, purpose=purpose, details={"reason": role_check.reason, "query_hash": (query_meta or {}).get("sha256")})
        raise HTTPException(status_code=403, detail=role_check.reason)


def get_text_embedder():
    global _text_embedder
    if _text_embedder is None:
        if EMBEDDER_BACKEND == "transformer":
            logger.info(f"Loading transformer text embedder: {model_config.TEXT_EMBEDDING_MODEL}")
            _text_embedder = TextEmbedder(model_name=model_config.TEXT_EMBEDDING_MODEL)
        else:
            logger.info("Loading lightweight hash text embedder (set MEDRAGX_EMBEDDER_BACKEND=transformer for PubMedBERT)")
            _text_embedder = _HashEmbedder(dim=model_config.EMBEDDING_DIM)
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
    request: Request,
    backend: str = "faiss",
    metric: str = "cosine",
    collection: str = "medragx",
    persist_dir: str = str(paths.VECTOR_DB_DIR),
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
    x_role: Optional[str] = Header(default=None, alias="X-Role"),
    x_purpose: Optional[str] = Header(default=None, alias="X-Purpose"),
) -> HealthResponse:
    identity = _resolve_identity(request, x_api_key, x_role, x_purpose)
    _authorize_action("health", actor=identity["actor"], role=identity["role"], purpose=identity["purpose"])
    try:
        store = open_store(backend, metric, collection, persist_dir)
        resp = HealthResponse(
            status="ok",
            backend=backend,
            size=store.size,
            dim=model_config.EMBEDDING_DIM,
            persist_dir=str(persist_dir),
        )
        _audit(
            "health",
            "allow",
            actor=identity["actor"],
            role=identity["role"],
            purpose=identity["purpose"],
            details={"backend": backend, "collection": collection, "retention_cleanup": _RETENTION_STATS},
        )
        return resp
    except Exception as e:  # noqa: BLE001
        _audit(
            "health",
            "error",
            actor=identity["actor"],
            role=identity["role"],
            purpose=identity["purpose"],
            details={"reason": str(e)},
        )
        # Health endpoint should not crash; return useful error
        raise HTTPException(status_code=500, detail=f"Health check failed: {e}")


@app.post("/embed/text", response_model=EmbedTextResponse)
def embed_text(
    req: EmbedTextRequest,
    request: Request,
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
    x_role: Optional[str] = Header(default=None, alias="X-Role"),
    x_purpose: Optional[str] = Header(default=None, alias="X-Purpose"),
    x_hitl_approved: Optional[str] = Header(default=None, alias="X-HITL-Approved"),
) -> EmbedTextResponse:
    identity = _resolve_identity(request, x_api_key, x_role, x_purpose)
    _authorize_action("embed_text", actor=identity["actor"], role=identity["role"], purpose=identity["purpose"])

    minimized_texts = []
    query_meta: Optional[Dict[str, Any]] = None
    for text in req.texts:
        safe_text, meta = minimize_query_text(text, ETHICAL_POLICY)
        minimized_texts.append(safe_text)
        if query_meta is None:
            query_meta = meta

    joined_preview = " ".join(minimized_texts[:2])
    hitl = check_hitl_approval(joined_preview, _parse_bool_header(x_hitl_approved))
    if not hitl.allowed:
        _audit(
            "embed_text",
            "deny",
            actor=identity["actor"],
            role=identity["role"],
            purpose=identity["purpose"],
            details={"reason": hitl.reason, "query_hash": (query_meta or {}).get("sha256")},
        )
        raise HTTPException(status_code=403, detail=hitl.reason)

    try:
        embedder = get_text_embedder()
        out = embedder.embed(minimized_texts, batch_size=req.batch_size)
        response = EmbedTextResponse(
            model_name=out.model_name,
            dim=out.embeddings.shape[1] if out.embeddings.size else model_config.EMBEDDING_DIM,
            embeddings=out.embeddings.tolist(),
        )
        _audit(
            "embed_text",
            "allow",
            actor=identity["actor"],
            role=identity["role"],
            purpose=identity["purpose"],
            details={
                "count": len(minimized_texts),
                "query_hash": (query_meta or {}).get("sha256"),
                "query_truncated": (query_meta or {}).get("truncated"),
                "query_redacted": (query_meta or {}).get("redacted"),
            },
        )
        return response
    except (ValueError, RuntimeError) as e:
        _audit(
            "embed_text",
            "error",
            actor=identity["actor"],
            role=identity["role"],
            purpose=identity["purpose"],
            details={"reason": str(e), "query_hash": (query_meta or {}).get("sha256")},
        )
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:  # noqa: BLE001
        _audit(
            "embed_text",
            "error",
            actor=identity["actor"],
            role=identity["role"],
            purpose=identity["purpose"],
            details={"reason": str(e), "query_hash": (query_meta or {}).get("sha256")},
        )
        raise HTTPException(status_code=500, detail=f"Embedding failed: {e}")


@app.post("/search", response_model=SearchResponse)
def search(
    req: SearchRequest,
    request: Request,
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
    x_role: Optional[str] = Header(default=None, alias="X-Role"),
    x_purpose: Optional[str] = Header(default=None, alias="X-Purpose"),
    x_hitl_approved: Optional[str] = Header(default=None, alias="X-HITL-Approved"),
) -> SearchResponse:
    """
    Text query -> PubMedBERT embeddings -> vector store search.
    Optionally filter by metadata, e.g. {"type":"lesion_3d"}.
    """
    identity = _resolve_identity(request, x_api_key, x_role, x_purpose)
    safe_query, query_meta = minimize_query_text(req.query, ETHICAL_POLICY)
    _authorize_action(
        "search",
        actor=identity["actor"],
        role=identity["role"],
        purpose=identity["purpose"],
        query_meta=query_meta,
    )

    hitl = check_hitl_approval(safe_query, _parse_bool_header(x_hitl_approved))
    if not hitl.allowed:
        _audit(
            "search",
            "deny",
            actor=identity["actor"],
            role=identity["role"],
            purpose=identity["purpose"],
            details={"reason": hitl.reason, "query_hash": query_meta.get("sha256")},
        )
        raise HTTPException(status_code=403, detail=hitl.reason)

    try:
        store = open_store(req.backend, req.metric, req.collection, req.persist_dir)

        if store.size == 0:
            raise HTTPException(
                status_code=400,
                detail="Vector store is empty. Build index first (index_build.py or pipeline.py --build_index).",
            )

        embedder = get_text_embedder()
        q_vec = embedder.embed(safe_query).embeddings[0]  # (768,)

        results: List[SearchResult] = store.search(query_vector=q_vec, top_k=req.top_k, filter=req.filter)

        hits = [SearchHit(id=r.id, score=r.score, metadata=r.metadata or {}) for r in results]

        response = SearchResponse(
            query=safe_query,
            top_k=req.top_k,
            backend=req.backend,
            metric=req.metric,
            collection=req.collection,
            persist_dir=req.persist_dir,
            results=hits,
        )
        _audit(
            "search",
            "allow",
            actor=identity["actor"],
            role=identity["role"],
            purpose=identity["purpose"],
            details={
                "top_k": req.top_k,
                "result_count": len(hits),
                "query_hash": query_meta.get("sha256"),
                "query_truncated": query_meta.get("truncated"),
                "query_redacted": query_meta.get("redacted"),
            },
        )
        return response

    except HTTPException:
        raise
    except (ValueError, FileNotFoundError) as e:
        _audit(
            "search",
            "error",
            actor=identity["actor"],
            role=identity["role"],
            purpose=identity["purpose"],
            details={"reason": str(e), "query_hash": query_meta.get("sha256")},
        )
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:  # noqa: BLE001
        _audit(
            "search",
            "error",
            actor=identity["actor"],
            role=identity["role"],
            purpose=identity["purpose"],
            details={"reason": str(e), "query_hash": query_meta.get("sha256")},
        )
        raise HTTPException(status_code=500, detail=f"Search failed: {e}")

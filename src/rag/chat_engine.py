from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.guardrails import precheck_user_query
from src.guardrails import postcheck_llm_output
from src.guardrails import safe_answer_wrapper

from src.rag.context_builder import build_prompt  # your prompt builder
from src.rag.multimodal_router import route_for_chat  # your nifti routing tool

# If you already have vector store retrieval utilities, import them here
from retrival.vector_store import create_vector_store
from models.embeddings.text_embedder import TextEmbedder
from config import paths, model_config


@dataclass
class ChatResponse:
    answer: str
    overlays: List[str]
    debug: Dict[str, Any]


class MedRAGXChatEngine:
    def __init__(self, backend: str = "faiss", collection: str = "medragx") -> None:
        self.store = create_vector_store(
            backend=backend,
            persist_dir=paths.VECTOR_DB_DIR,
            dim=model_config.EMBEDDING_DIM,
            metric="cosine",
            collection_name=collection,
        )
        self.text_embedder = TextEmbedder(model_name=model_config.TEXT_EMBEDDING_MODEL)

    def retrieve_chunks(self, query: str, top_k: int = 5, filter_: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        if self.store.size == 0:
            return []
        q_vec = self.text_embedder.embed(query).embeddings[0]
        results = self.store.search(query_vector=q_vec, top_k=top_k, filter=filter_)
        chunks: List[Dict[str, Any]] = []
        for r in results:
            chunks.append({
                "source": r.id,
                "score": float(r.score),
                "text": str(r.metadata)  # Replace with actual chunk text if you store docs separately
            })
        return chunks

    def call_llm(self, system: str, user: str, context: str) -> str:
        """
        Replace this with your actual Ollama call.
        Keep it as a single function so guardrails wrap it easily.
        """
        # TODO: implement Ollama /api/chat call here
        return f"(stub) Answer based on context: {context[:600]}..."

    def answer(
        self,
        user_query: str,
        uploaded_paths: Optional[List[Path]] = None,
        want_nifti_processing: bool = True,
    ) -> ChatResponse:
        uploaded_paths = uploaded_paths or []

        # 1) Optional NIfTI processing (only if user asked something like tumor size / volume)
        case_stats = None
        overlays: List[str] = []

        if want_nifti_processing and uploaded_paths:
            rr = route_for_chat(
                uploaded_paths=uploaded_paths,
                out_dir=paths.ARTIFACTS_DIR if hasattr(paths, "ARTIFACTS_DIR") else Path("runtime/outputs"),
                case_id="chat_case",
                mask_source="gt_seg",   # or "pred_mask" with checkpoint
                checkpoint=None,
            )
            if rr.kind == "nifti":
                case_stats = {"rag_text": rr.rag_text, "structured": rr.structured}
                overlays = rr.overlay_paths

        # 2) Retrieval (text RAG)
        retrieved_chunks = self.retrieve_chunks(
            query=user_query,
            top_k=5,
            filter_={"type": "lesion_3d"},  # optional filter
        )

        # ✅ 3) Guardrails PRECHECK (THIS is where you add it!)
        decision = precheck_user_query(
            user_query=user_query,
            has_case_stats=(case_stats is not None),
            has_retrieved_sources=bool(retrieved_chunks),
        )
        if not decision.allowed:
            return ChatResponse(
                answer=decision.safe_response or "Request blocked by guardrails.",
                overlays=overlays,
                debug={"reasons": decision.reasons, "risk": decision.risk},
            )

        # 4) Build prompt
        pb = build_prompt(
            question=user_query,
            nifti_rag_text=case_stats.get("rag_text") if case_stats else None,
            nifti_structured=case_stats.get("structured") if case_stats else None,
            retrieved_chunks=retrieved_chunks,
        )

        system = pb.system + "\n\n" + (decision.prompt_rules or "")

        # 5) Call LLM
        raw = self.call_llm(system=system, user=pb.user, context=pb.context)

        # ✅ 6) Guardrails POSTCHECK (THIS is also where you add it!)
        post = postcheck_llm_output(raw)
        if not post.allowed:
            return ChatResponse(
                answer=post.safe_response or "Output blocked by guardrails.",
                overlays=overlays,
                debug={"reasons": post.reasons, "risk": post.risk},
            )

        # 7) Wrap with disclaimer
        final = safe_answer_wrapper(raw)

        return ChatResponse(
            answer=final,
            overlays=overlays,
            debug={"risk": decision.risk, "precheck": decision.reasons},
        )
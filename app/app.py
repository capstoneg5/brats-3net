# app.py
"""
MedRAG-X — Streamlit Chat Interface
=====================================
Full chat-based clinical reasoning UI with:
  • Conversational chat history (multi-turn)
  • MRI image upload (PNG/JPG/NIfTI slices) with inline display
  • Lesion ID selector OR free-text queries
  • Image-aware analysis (displays uploaded MRI alongside response)
  • Evidence explorer, comparison charts, confidence badges
  • Guardrail status indicators

Run:
  PYTHONPATH=. streamlit run app.py

Requires:
  pip install streamlit plotly pillow --break-system-packages
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st
import numpy as np
import hashlib
from src.guardrails.ethical_guardrails import (
    append_audit_event,
    check_hitl_approval,
    check_purpose_allowed,
    check_role_permission,
    check_vendor_governance,
    enforce_retention_policy,
    get_ethical_policy,
    minimize_query_text,
    redact_sensitive_text,
)
from src.rag.prompt_templates import get_tumor_analysis_prompt

# ─────────────────── Page Config ───────────────────
st.set_page_config(
    page_title="MedRAG-X",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _apply_light_theme() -> None:
    """Force a light app shell even when the browser prefers dark mode."""
    st.markdown(
        """
        <style>
        .stApp,
        [data-testid="stAppViewContainer"],
        [data-testid="stHeader"] {
            background: #ffffff;
            color: #111827;
        }

        [data-testid="stSidebar"] {
            background: #f5f7fa;
        }

        [data-testid="stSidebar"] * ,
        [data-testid="stAppViewContainer"] * {
            color: inherit;
        }

        [data-testid="stChatMessage"],
        [data-testid="stExpander"],
        [data-testid="stMetric"],
        [data-testid="stFileUploader"] {
            background: #ffffff;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

# ─────────────────── Constants ───────────────────
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434/api/chat")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
UPLOAD_DIR = Path("artifacts/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
ETHICAL_POLICY = get_ethical_policy()


def _audit(event_type: str, outcome: str, role: str, purpose: str, details: Optional[Dict[str, Any]] = None) -> None:
    """Best-effort audit: never break UI flow."""
    try:
        append_audit_event(
            event_type=event_type,
            outcome=outcome,
            actor="streamlit_user",
            role=role,
            purpose=purpose,
            details=details or {},
            policy=ETHICAL_POLICY,
        )
    except Exception:
        pass


def _default_neo4j_password() -> str:
    """Use env password when set; otherwise allow local dev default."""
    env_pw = (os.getenv("NEO4J_PASSWORD", "") or "").strip()
    if env_pw:
        return env_pw
    uri = (os.getenv("NEO4J_URI", NEO4J_URI) or "").lower()
    is_local = "127.0.0.1" in uri or "localhost" in uri
    return "neo4j123" if is_local else ""

def save_uploaded_file(uploaded_file, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    original = Path(uploaded_file.name).name
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", original)
    if not safe_name:
        safe_name = "upload.bin"
    dst = out_dir / safe_name
    with open(dst, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return dst


def load_nifti(path: Path):
    import nibabel as nib
    nii = nib.load(str(path))
    data = nii.get_fdata()
    zooms = nii.header.get_zooms()
    return nii, data, zooms


# ═══════════════════════════════════════════════════
#  Session state initialization
# ═══════════════════════════════════════════════════

def init_session():
    """Initialize all session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "pending_images" not in st.session_state:
        st.session_state.pending_images = []
    if "pending_image" not in st.session_state:
        st.session_state.pending_image = None
    if "last_rag_result" not in st.session_state:
        st.session_state.last_rag_result = None
    if "live_result" not in st.session_state:
        st.session_state.live_result = None
    if "selected_lesion" not in st.session_state:
        st.session_state.selected_lesion = None
    if "neo4j_password" not in st.session_state:
        st.session_state.neo4j_password = _default_neo4j_password()
    if "user_role" not in st.session_state:
        st.session_state.user_role = os.getenv("MEDRAGX_USER_ROLE", "analyst").strip().lower() or "analyst"
    if "user_purpose" not in st.session_state:
        st.session_state.user_purpose = os.getenv("MEDRAGX_USER_PURPOSE", "clinical_reasoning").strip().lower() or "clinical_reasoning"
    if "hitl_approved" not in st.session_state:
        st.session_state.hitl_approved = False
    if "retention_cleanup_done" not in st.session_state:
        cleanup_stats = enforce_retention_policy(ETHICAL_POLICY)
        st.session_state.retention_cleanup_done = True
        st.session_state.retention_cleanup_stats = cleanup_stats

init_session()


def _current_neo4j_password() -> str:
    """Resolve Neo4j password from session first, then environment."""
    return (st.session_state.get("neo4j_password") or _default_neo4j_password()).strip()


# ═══════════════════════════════════════════════════
#  Neo4j helpers (cached)
# ═══════════════════════════════════════════════════

@st.cache_resource
def get_retriever(uri: str, user: str, password: str):
    if not password:
        raise RuntimeError("Missing Neo4j password. Set sidebar Neo4j Password or NEO4J_PASSWORD environment variable.")
    from src.rag.neo4j_retriever import Neo4jRetriever
    return Neo4jRetriever(uri=uri, user=user, password=password)


@st.cache_data(ttl=300)
def fetch_lesion_ids() -> List[Dict[str, str]]:
    retriever = get_retriever(NEO4J_URI, NEO4J_USER, _current_neo4j_password())
    with retriever.driver.session() as s:
        rows = s.run(
            "MATCH (l:Lesion) WHERE l.embedding IS NOT NULL "
            "RETURN l.id AS id, l.patient_id AS patient_id "
            "ORDER BY l.id"
        ).data()
    return rows


@st.cache_data(ttl=300)
def fetch_lesion_details(lesion_id: str) -> Optional[Dict[str, Any]]:
    retriever = get_retriever(NEO4J_URI, NEO4J_USER, _current_neo4j_password())
    with retriever.driver.session() as s:
        row = s.run(
            "MATCH (l:Lesion {id: $id}) "
            "RETURN l.id AS id, l.patient_id AS patient_id, "
            "l.wt_vox AS wt_vox, l.tc_vox AS tc_vox, l.et_vox AS et_vox, "
            "l.et_pct AS et_pct, l.tc_pct AS tc_pct, "
            "l.tumor_volume_voxels AS tumor_vol, "
            "l.centroid_zyx AS centroid, l.lesion_type AS lesion_type",
            id=lesion_id,
        ).single()
    return dict(row) if row else None


# ═══════════════════════════════════════════════════
#  Image processing helpers
# ═══════════════════════════════════════════════════

def process_uploaded_image(uploaded_file) -> Dict[str, Any]:
    """
    Process an uploaded image file.
    Supports: PNG, JPG, JPEG, TIFF, NIfTI (.nii/.nii.gz), NumPy (.npy)
    Returns dict with 'display_image' (bytes) and 'metadata'.
    """
    filename = uploaded_file.name.lower()
    raw_bytes = uploaded_file.read()
    uploaded_file.seek(0)

    result = {
        "filename": uploaded_file.name,
        "size_bytes": len(raw_bytes),
        "metadata": {},
        "display_image": None,
        "numpy_slice": None,
    }

    # ── Standard image (PNG/JPG) ──
    if filename.endswith((".png", ".jpg", ".jpeg", ".tiff", ".bmp")):
        result["display_image"] = raw_bytes
        try:
            from PIL import Image
            img = Image.open(io.BytesIO(raw_bytes))
            result["metadata"] = {
                "format": img.format,
                "size": f"{img.size[0]}x{img.size[1]}",
                "mode": img.mode,
            }
        except Exception:
            result["metadata"] = {"format": filename.split(".")[-1]}
        return result

    # ── NumPy array (.npy) ──
    if filename.endswith(".npy"):
        try:
            arr = np.load(io.BytesIO(raw_bytes))
            result["numpy_slice"] = arr
            result["metadata"] = {"shape": list(arr.shape), "dtype": str(arr.dtype)}
            result["display_image"] = _numpy_to_png(arr)
        except Exception as e:
            result["metadata"] = {"error": str(e)}
        return result

    # ── NIfTI volume (.nii / .nii.gz) ──
    if filename.endswith((".nii", ".nii.gz")):
        try:
            import nibabel as nib

            # Save file to disk so we can reload later (don't delete temp)
            saved_path = save_uploaded_file(uploaded_file, UPLOAD_DIR)

            nii = nib.load(str(saved_path))
            data = nii.get_fdata()
            zooms = nii.header.get_zooms()

            # detect channel-last (BraTS often [H,W,Z,4])
            channels = 1
            if data.ndim == 4:
                channels = data.shape[-1]

            result["metadata"] = {
                "shape": list(data.shape),
                "dtype": str(data.dtype),
                "voxel_sizes": [round(float(v), 3) for v in zooms[:3]],
                "channels": channels,
                "saved_path": str(saved_path),
            }

            # preview: choose mid-slice of Z, channel 0 by default
            if data.ndim >= 3:
                z_mid = data.shape[2] // 2

                if data.ndim == 4:
                    ch = 0  # default preview channel
                    slice_2d = data[:, :, z_mid, ch]
                    result["metadata"]["preview_channel"] = ch
                else:
                    slice_2d = data[:, :, z_mid]

                result["numpy_slice"] = slice_2d
                result["display_image"] = _numpy_to_png(slice_2d)

            return result

        except ImportError:
            result["metadata"] = {"note": "nibabel not installed; cannot preview NIfTI"}
            return result
        except Exception as e:
            result["metadata"] = {"error": str(e)}
            return result


# -----------------------------
# RAG Metrics + Dashboard
# -----------------------------
def load_offline_rag_metrics(out_dir: str | Path = "artifacts/rag_eval_results") -> dict | None:
    """Load offline/aggregate RAG eval metrics written by run_rag_eval.py.

    Expected default file: artifacts/rag_eval_results/rag_eval_summary.json
    """
    try:
        out_dir = Path(out_dir)
        summary_path = out_dir / "rag_eval_summary.json"
        if not summary_path.exists():
            return None
        return json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def compute_live_rag_metrics(result: dict, top_k: int) -> dict:
    """Compute per-query (live) metrics from a single RAG run result."""
    status = (result.get("status") or "unknown")

    # Try common keys for retrieved evidence
    retrieved = result.get("retrieved_rows") or result.get("retrieved") or result.get("evidence") or result.get("contexts") or []
    if isinstance(retrieved, dict):
        retrieved = list(retrieved.values())
    if not isinstance(retrieved, list):
        retrieved = []

    # Top score
    scores: List[float] = []
    for r in retrieved:
        if isinstance(r, dict):
            s = r.get("score")
            if isinstance(s, (int, float)):
                scores.append(float(s))
    top_score = max(scores) if scores else float(result.get("top_score") or 0.0)

    # Coverage (fallback: retrieved/top_k)
    coverage = result.get("coverage")
    if not isinstance(coverage, (int, float)):
        coverage = (len(retrieved) / float(top_k)) if top_k else 0.0

    # Grounding / hallucination (prefer explicit metrics dict if present)
    metrics = result.get("metrics") if isinstance(result.get("metrics"), dict) else {}
    grounding = metrics.get("grounding", result.get("grounding"))
    halluc = metrics.get("hallucination_rate", result.get("hallucination_rate"))
    metrics_available = grounding is not None or halluc is not None

    if grounding is not None:
        try:
            grounding = float(grounding)
        except Exception:
            grounding = None
    if halluc is not None:
        try:
            halluc = float(halluc)
        except Exception:
            halluc = None

    # Confidence can be a str or dict
    conf = result.get("confidence")
    if isinstance(conf, dict):
        confidence_label = conf.get("label") or conf.get("value") or "N/A"
        confidence_rationale = conf.get("rationale") or result.get("confidence_rationale")
    else:
        confidence_label = conf if conf is not None else "N/A"
        confidence_rationale = result.get("confidence_rationale")

    return {
        "status": status,
        "grounding": grounding,
        "hallucination_rate": halluc,
        "metrics_available": metrics_available,
        "confidence": str(confidence_label),
        "confidence_rationale": confidence_rationale,
        "retrieved_count": len(retrieved),
        "top_score": float(top_score),
        "coverage": float(coverage),
    }


def render_rag_dashboard_ui(live_result: dict | None, top_k: int, min_score: float) -> None:
    """Two-section RAG dashboard: Live metrics + Offline eval metrics."""
    st.markdown("## RAG Dashboard")

    # ---- Live ----
    st.markdown("### Live (this query)")
    if not live_result:
        st.warning("No live RAG metrics yet.")

        # pick a lesion id to run
        default_lid = st.session_state.get("selected_lesion")
        if not default_lid:
            last = st.session_state.get("last_rag_result") or {}
            default_lid = last.get("lesion_id")

        lid = st.text_input("Lesion ID to run", value=default_lid or "", placeholder="e.g. lesion38")

        if st.button("▶ Run RAG Query", key="rag_dash_run"):
            if not lid.strip():
                st.error("Please select a lesion (sidebar) or type a lesion id here.")
            else:
                with st.spinner(f"Running RAG for {lid}..."):
                    result = run_guarded_rag(
                        lesion_id=lid.strip(),
                        top_k=top_k,
                        min_score=min_score,
                        role=st.session_state.get("user_role", "analyst"),
                        purpose=st.session_state.get("user_purpose", "clinical_reasoning"),
                    )
                st.session_state["live_result"] = result
                st.session_state["last_rag_result"] = result
                st.rerun()
    else:
        m = compute_live_rag_metrics(live_result, top_k=top_k)

        c1, c2, c3 = st.columns(3)
        c1.metric("Status", m["status"])
        c2.metric("Grounding", f'{m["grounding"]:.3f}' if m["grounding"] is not None else "N/A")
        c3.metric(
            "Hallucination rate",
            f'{m["hallucination_rate"]:.3f}' if m["hallucination_rate"] is not None else "N/A",
        )

        c4, c5, c6 = st.columns(3)
        c4.metric("Confidence", m["confidence"])
        c5.metric("Retrieved", str(m["retrieved_count"]))
        c6.metric("Top score", f'{m["top_score"]:.4f}')

        st.metric("Coverage", f'{m["coverage"]:.3f}')
        if not m.get("metrics_available"):
            st.caption("Grounding and hallucination are not computed for live queries in this view. Use offline RAG eval for those metrics.")
        if m.get("confidence_rationale"):
            with st.expander("Confidence rationale"):
                st.write(m["confidence_rationale"])

    # ---- Offline ----
    st.markdown("### Offline (overall eval)")
    offline = load_offline_rag_metrics()
    if not offline:
        st.info("Offline eval metrics not found. Run run_rag_eval.py to generate artifacts/rag_eval_results/rag_eval_summary.json")
    else:
        overall = offline.get("overall") if isinstance(offline, dict) else None
        if not isinstance(overall, dict):
            overall = offline if isinstance(offline, dict) else {}

        cols = st.columns(3)
        keys = [k for k in ["n_queries", "avg_grounding", "avg_hallucination_rate", "success_rate", "avg_top_score"] if k in overall]
        for i, k in enumerate(keys[:3]):
            cols[i].metric(k.replace("_", " ").title(), str(overall[k]))
        for k in keys[3:]:
            st.metric(k.replace("_", " ").title(), str(overall[k]))

        with st.expander("Raw offline summary"):
            st.json(offline)
def _numpy_to_png(arr: np.ndarray) -> bytes:
    """Convert a 2D numpy array to PNG bytes for display."""
    from PIL import Image

    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
        arr = arr[0]
    elif arr.ndim == 3 and arr.shape[-1] in (1, 3, 4):
        arr = arr[:, :, 0]
    if arr.ndim != 2:
        arr = arr[arr.shape[0] // 2] if arr.ndim == 3 else arr.reshape(arr.shape[0], -1)

    arr = arr.astype(np.float64)
    amin, amax = arr.min(), arr.max()
    if amax - amin > 0:
        arr = (arr - amin) / (amax - amin) * 255.0
    arr = arr.astype(np.uint8)

    img = Image.fromarray(arr, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ═══════════════════════════════════════════════════
#  LLM helpers
# ═══════════════════════════════════════════════════

def call_ollama(prompt: str, model: str = OLLAMA_MODEL) -> Tuple[str, Optional[str]]:
    import requests
    governance = check_vendor_governance(OLLAMA_URL, model, ETHICAL_POLICY)
    if not governance.allowed:
        return "", governance.reason

    payload = {
        "model": model, "stream": False,
        "messages": [{"role": "user", "content": prompt}],
        "options": {"temperature": 0.2, "top_p": 0.9, "num_ctx": 4096},
    }
    try:
        r = requests.post(OLLAMA_URL, json=payload, timeout=600)
        r.raise_for_status()
        return r.json()["message"]["content"], None
    except Exception as e:
        safe_error, _ = redact_sensitive_text(str(e))
        return "", safe_error


def call_ollama_chat(messages: List[Dict[str, str]], model: str = OLLAMA_MODEL) -> Tuple[str, Optional[str]]:
    """Multi-turn chat with conversation history."""
    import requests
    governance = check_vendor_governance(OLLAMA_URL, model, ETHICAL_POLICY)
    if not governance.allowed:
        return "", governance.reason

    payload = {
        "model": model, "stream": False, "messages": messages,
        "options": {"temperature": 0.3, "top_p": 0.9, "num_ctx": 4096},
    }
    try:
        r = requests.post(OLLAMA_URL, json=payload, timeout=600)
        r.raise_for_status()
        return r.json()["message"]["content"], None
    except Exception as e:
        safe_error, _ = redact_sensitive_text(str(e))
        return "", safe_error


# ═══════════════════════════════════════════════════
#  RAG Pipeline
# ═══════════════════════════════════════════════════

def run_guarded_rag(
    lesion_id: str,
    top_k: int = 5,
    min_score: float = 0.70,
    user_query: str = "",
    image_metas: Optional[List[Dict[str, Any]]] = None,
    role: str = "analyst",
    purpose: str = "clinical_reasoning",
) -> Dict[str, Any]:
    from src.guardrails import (
        input_guardrail, retrieval_guardrail, output_guardrail,
        RetrievalBundle, RetrievalItem,
    )
    from src.guardrails.prompts import SYSTEM_PROMPT, USER_TEMPLATE
    from src.guardrails.clinical_guardrails import enforce_grounding_instructions

    result = {
        "lesion_id": lesion_id, "status": "success", "guardrails": {},
        "context": "", "facts": "", "llm_answer": "",
        "confidence": "", "confidence_rationale": "",
        "retrieved_rows": [], "timing": {}, "ethical": {},
    }

    purpose_check = check_purpose_allowed(purpose, ETHICAL_POLICY)
    if not purpose_check.allowed:
        result["status"] = "blocked_purpose"
        result["guardrails"]["ethical"] = purpose_check.reason
        _audit("rag_query", "deny", role=role, purpose=purpose, details={"lesion_id": lesion_id, "reason": purpose_check.reason})
        return result

    role_check = check_role_permission(role, "rag_query", ETHICAL_POLICY)
    if not role_check.allowed:
        result["status"] = "blocked_access"
        result["guardrails"]["ethical"] = role_check.reason
        _audit("rag_query", "deny", role=role, purpose=purpose, details={"lesion_id": lesion_id, "reason": role_check.reason})
        return result

    base_query = f"Compare {lesion_id} to its most similar lesions and explain ET% differences."
    if user_query:
        base_query = f"{user_query} (Reference lesion: {lesion_id})"
    if image_metas:
        image_fragments = []
        for meta in image_metas[:10]:
            shape = meta.get("shape", "unknown")
            vox = meta.get("voxel_sizes", "unknown")
            channels = meta.get("channels", 1)
            sel = meta.get("selected_channel", None)
            labels = meta.get("channel_labels", None)
            modality_note = ""
            if channels and int(channels) > 1 and sel is not None and labels and int(sel) < len(labels):
                modality_note = f", selected_channel={sel} label={labels[int(sel)]}"
            image_fragments.append(
                f"{meta.get('filename', 'image')} shape={shape} voxel_sizes={vox} channels={channels}{modality_note}"
            )
        base_query += f" [Uploaded MRIs ({len(image_fragments)}): " + " | ".join(image_fragments) + "]"
    query, minimization = minimize_query_text(base_query, ETHICAL_POLICY)
    result["ethical"]["data_minimization"] = minimization

    # (1) INPUT GUARDRAIL
    t0 = time.time()
    d0 = input_guardrail(query)
    result["timing"]["input_guardrail"] = time.time() - t0
    if d0.action != "allow":
        result["status"] = "blocked_input"
        result["guardrails"]["input"] = d0.reason or "Blocked"
        _audit(
            "rag_query",
            "deny",
            role=role,
            purpose=purpose,
            details={"lesion_id": lesion_id, "reason": result["guardrails"]["input"], "query_hash": minimization.get("sha256")},
        )
        return result
    result["guardrails"]["input"] = "allowed"

    # (2) RETRIEVAL
    t1 = time.time()
    try:
        retriever = get_retriever(NEO4J_URI, NEO4J_USER, _current_neo4j_password())
        rows = retriever.retrieve_similar(lesion_id=lesion_id, k=top_k)
    except Exception as e:
        result["status"] = "retrieval_error"
        safe_error, _ = redact_sensitive_text(str(e))
        result["guardrails"]["retrieval"] = safe_error
        _audit(
            "rag_query",
            "error",
            role=role,
            purpose=purpose,
            details={"lesion_id": lesion_id, "reason": safe_error, "query_hash": minimization.get("sha256")},
        )
        return result
    result["timing"]["retrieval"] = time.time() - t1
    result["retrieved_rows"] = rows
    if not rows:
        result["status"] = "no_results"
        _audit(
            "rag_query",
            "deny",
            role=role,
            purpose=purpose,
            details={"lesion_id": lesion_id, "reason": "no_results", "query_hash": minimization.get("sha256")},
        )
        return result

    # (3) RETRIEVAL GUARDRAIL
    items = _rows_to_items(rows)
    bundle = RetrievalBundle(query=query, items=items, k=top_k, min_score=min_score)
    d1 = retrieval_guardrail(bundle)
    if d1.action != "allow":
        result["status"] = "blocked_retrieval"
        result["guardrails"]["retrieval"] = d1.reason or "Blocked"
        _audit(
            "rag_query",
            "deny",
            role=role,
            purpose=purpose,
            details={"lesion_id": lesion_id, "reason": result["guardrails"]["retrieval"], "query_hash": minimization.get("sha256")},
        )
        return result
    result["guardrails"]["retrieval"] = "allowed"

    # (4) Deterministic facts
    diffs, q = _compute_diffs(rows, lesion_id)
    facts_block = _format_facts(diffs, q)
    context_block = retriever.build_context(rows)
    result["context"] = context_block
    result["facts"] = facts_block

    # (5) Prompt
    q_text = (user_query or "").lower()
    is_comparison_query = any(
        kw in q_text for kw in ["compare", "similar", "difference", "delta", "closest", "match"]
    )
    required_lesions = [d["lesion_id"] for d in diffs if d["lesion_id"] != lesion_id][:max(0, top_k - 1)]
    evidence_lines = "\n".join([
        f"• [{i+1}] lesion={d['lesion_id']}"
        for i, d in enumerate(diffs) if d["lesion_id"] != lesion_id
    ][:max(0, top_k - 1)])

    if is_comparison_query:
        system = SYSTEM_PROMPT + "\n" + enforce_grounding_instructions() + f"""
IMPORTANT OUTPUT FORMAT:
- Output EXACTLY {len(required_lesions)} bullet lines.
- NO extra headers.
- Each bullet MUST start EXACTLY with: "- lesion_id=<id>"
- Exclude the query lesion ({lesion_id}).
- Each bullet MUST include similarity score=0.xxxx, ET%, ΔET_pp.
- Use ONLY the FACTS block numbers.

After the bullets, output:
Evidence used:
{evidence_lines}

Safety note: <one sentence only>
"""
    else:
        system = SYSTEM_PROMPT + "\n" + enforce_grounding_instructions() + f"""
IMPORTANT OUTPUT FORMAT:
- Start with: "Direct answer: <Yes/No/Insufficient>"
- Then section "Reasoning:" with 2-4 concise bullets grounded in FACTS.
- Then section:
Evidence used:
{evidence_lines}
- Then:
Safety note: <one sentence only>

Rules:
- Answer the user's question directly from FACTS and RAW CONTEXT only.
- If anatomical location (e.g., left temporal lobe) is asked but not explicitly present in FACTS/RAW CONTEXT, state that location cannot be confirmed.
- Do not invent diagnosis/treatment recommendations.
"""
    user_prompt = USER_TEMPLATE.format(query=query)
    if user_query:
        user_prompt += f"\n\nAdditional user context: {user_query}"
    prompt = f"{system}\n\nFACTS:\n{facts_block}\n\nRAW CONTEXT:\n{context_block}\n\n{user_prompt}"

    t2 = time.time()
    llm_answer, llm_error = call_ollama(prompt)
    result["timing"]["llm"] = time.time() - t2
    if llm_error:
        result["status"] = "llm_error"
        result["llm_answer"] = "LLM is unavailable right now. Please verify Ollama is running and try again."
        result["confidence"] = "N/A"
        result["confidence_rationale"] = "LLM connection failed"
        result["guardrails"]["output"] = "skipped"
        _audit(
            "rag_query",
            "error",
            role=role,
            purpose=purpose,
            details={
                "lesion_id": lesion_id,
                "reason": "llm_error",
                "llm_error": llm_error,
                "query_hash": minimization.get("sha256"),
            },
        )
        return result

    # (6) OUTPUT GUARDRAIL
    d2 = output_guardrail(llm_answer)
    if d2.action != "allow":
        result["status"] = "blocked_output"
        result["guardrails"]["output"] = d2.reason or "Blocked"
        _audit(
            "rag_query",
            "deny",
            role=role,
            purpose=purpose,
            details={"lesion_id": lesion_id, "reason": result["guardrails"]["output"], "query_hash": minimization.get("sha256")},
        )
        return result
    result["guardrails"]["output"] = "allowed"

    safe_answer, output_redacted = redact_sensitive_text(llm_answer)
    result["ethical"]["output_redacted"] = output_redacted

    # (7) Confidence
    scores = [float(d["score"]) for d in diffs if d.get("score")]
    top_score = max(scores) if scores else 0.0
    passed = [s for s in scores if s >= min_score]
    coverage = len(passed) / len(scores) if scores else 0.0
    if top_score >= 0.95 and coverage >= 0.80:
        conf, rationale = "High", f"Top score {top_score:.4f}, coverage {coverage:.2f}"
    elif top_score >= 0.90 and coverage >= 0.60:
        conf, rationale = "Moderate", f"Top score {top_score:.4f}, coverage {coverage:.2f}"
    else:
        conf, rationale = "Low", f"Top score {top_score:.4f}, coverage {coverage:.2f}"
    case_ids_text = _format_case_ids(rows, lesion_id, top_k=top_k)
    result["id_trace"] = case_ids_text
    result["llm_answer"] = f"{case_ids_text}\n\n{safe_answer}"
    result["confidence"] = conf
    result["confidence_rationale"] = rationale
    _audit(
        "rag_query",
        "allow",
        role=role,
        purpose=purpose,
        details={
            "lesion_id": lesion_id,
            "top_k": top_k,
            "min_score": min_score,
            "query_hash": minimization.get("sha256"),
            "query_truncated": minimization.get("truncated"),
            "query_redacted": minimization.get("redacted"),
        },
    )
    return result


def handle_general_chat(
    user_msg: str,
    image_metas: Optional[List[Dict[str, Any]]] = None,
    uploaded_images: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[str, Optional[str]]:
    """Handle on-topic medical chat that doesn't trigger the full RAG pipeline."""
    nifti_analysis = _analyze_uploaded_nifti(uploaded_images=uploaded_images)
    direct_nifti_answer = _maybe_answer_from_uploaded_nifti(
        user_msg,
        uploaded_images=uploaded_images,
        nifti_analysis=nifti_analysis,
    )
    if direct_nifti_answer:
        return direct_nifti_answer, None

    system_context = (
        "You are MedRAG-X, a clinical reasoning assistant ONLY for brain tumor MRI analysis "
        "using the BraTS dataset.\n\n"
        "STRICT RULES YOU MUST FOLLOW:\n"
        "1. ONLY answer questions about: brain MRI, BraTS, tumors, lesions, segmentation, "
        "   medical imaging, tumor metrics (ET%, TC%, WT), knowledge graphs, or the MedRAG-X system.\n"
        "2. If a question is NOT related to these topics, you MUST refuse. Say:\n"
        "   'I can only assist with brain MRI analysis and BraTS-related clinical reasoning.'\n"
        "3. NEVER answer questions about sports, cricket, movies, politics, general knowledge,\n"
        "   cooking, weather, coding tutorials, or ANY non-medical topic.\n"
        "4. Do NOT provide diagnosis, treatment advice, or medication guidance.\n"
        "5. Always note your outputs are for research/decision support, not autonomous diagnosis.\n"
    )
    if uploaded_images and _is_tumor_imaging_query(user_msg):
        system_context += (
            "\nThe user is asking a tumor-analysis question about uploaded imaging. "
            "Follow these tumor-analysis instructions in addition to the MedRAG-X safety rules.\n\n"
            f"{get_tumor_analysis_prompt()}\n"
        )
        if nifti_analysis:
            system_context += (
                "\nImaging-derived analysis from the uploaded NIfTI file(s):\n"
                f"{nifti_analysis['rag_text']}\n\n"
                "Imaging-derived structured metrics:\n"
                f"{json.dumps(nifti_analysis['structured'], indent=2)}\n"
                "Use these computed metrics directly when answering tumor-related questions.\n"
            )
    if image_metas:
        compact_meta = []
        for meta in image_metas[:10]:
            compact_meta.append({
                "filename": meta.get("filename"),
                "shape": meta.get("shape"),
                "voxel_sizes": meta.get("voxel_sizes"),
                "channels": meta.get("channels"),
                "selected_channel": meta.get("selected_channel"),
            })
        system_context += (
            f"\nThe user uploaded {len(compact_meta)} MRI file(s). "
            f"Metadata: {json.dumps(compact_meta, default=str)}. "
            "Acknowledge all uploaded files and provide relevant observations about what can be "
            "analyzed for each file (format, dimensions, modality if identifiable).\n"
        )

    history_messages = [{"role": "system", "content": system_context}]
    recent = st.session_state.messages[-10:]
    for msg in recent:
        role = "user" if msg["role"] == "user" else "assistant"
        content = msg.get("text", "")
        if content:
            history_messages.append({"role": role, "content": content})
    history_messages.append({"role": "user", "content": user_msg})
    answer, error = call_ollama_chat(history_messages)
    if error:
        return "", error
    safe_answer, _ = redact_sensitive_text(answer)
    return safe_answer, None


def _is_tumor_size_query(text: str) -> bool:
    q = text.lower()
    keywords = [
        "tumor size",
        "tumour size",
        "tumor volume",
        "tumour volume",
        "how big",
        "volume",
        "size of the tumor",
        "size of the tumour",
    ]
    return any(kw in q for kw in keywords)


def _is_tumor_imaging_query(text: str) -> bool:
    q = text.lower()
    keywords = [
        "tumor",
        "tumour",
        "lesion",
        "segmentation",
        "mask",
        "enhancing tumor",
        "edema",
        "necrotic",
        "whole tumor",
        "tumor size",
        "tumor volume",
    ]
    return any(kw in q for kw in keywords)


def _format_nifti_metric_answer(structured: Dict[str, Any], filename: str) -> str:
    stats = structured.get("lesion_stats", {})
    labels_present = stats.get("labels_present") or []
    bbox = stats.get("bbox_ijk") or {}
    bbox_min = bbox.get("min")
    bbox_max = bbox.get("max")
    approx_diam_mm = stats.get("approx_max_diameter_mm")

    lines = [
        f"Tumor size for `{filename}`:",
        f"- Total tumor volume: {float(stats.get('lesion_volume_ml', 0.0)):.3f} mL",
        f"- Total tumor volume: {float(stats.get('lesion_volume_mm3', 0.0)):.0f} mm^3",
        f"- Tumor voxel count: {int(stats.get('lesion_voxels', 0))}",
    ]
    if approx_diam_mm is not None:
        lines.append(f"- Approximate max extent: {float(approx_diam_mm):.1f} mm")
    if bbox_min is not None and bbox_max is not None:
        lines.append(f"- Bounding box: min={bbox_min}, max={bbox_max}")
    if labels_present:
        lines.append(f"- Labels present in mask: {labels_present}")
    lines.append("Computed directly from the uploaded segmentation mask and voxel spacing. Research use only.")
    return "\n".join(lines)


def _analyze_uploaded_nifti(
    uploaded_images: Optional[List[Dict[str, Any]]] = None,
) -> Optional[Dict[str, Any]]:
    if not uploaded_images:
        return None

    saved_paths: List[Path] = []
    first_filename: Optional[str] = None
    for img in uploaded_images[:10]:
        meta = img.get("metadata", {}) if isinstance(img, dict) else {}
        saved_path = meta.get("saved_path")
        if saved_path:
            saved_paths.append(Path(saved_path))
            if first_filename is None:
                first_filename = img.get("filename") or Path(saved_path).name

    if not saved_paths:
        return None

    try:
        from src.rag.multimodal_router import route_for_chat

        rr = route_for_chat(
            uploaded_paths=saved_paths,
            out_dir=Path("artifacts/runtime"),
            case_id="streamlit_upload",
            mask_source="gt_seg",
            checkpoint=None,
        )
    except Exception:
        return None

    return {
        "filename": first_filename or saved_paths[0].name,
        "rag_text": rr.rag_text,
        "structured": rr.structured if isinstance(rr.structured, dict) else {},
    }


def _maybe_answer_from_uploaded_nifti(
    user_msg: str,
    uploaded_images: Optional[List[Dict[str, Any]]] = None,
    nifti_analysis: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    if not uploaded_images or not _is_tumor_size_query(user_msg):
        return None

    analysis = nifti_analysis or _analyze_uploaded_nifti(uploaded_images=uploaded_images)
    if not analysis:
        return None

    structured = analysis.get("structured", {})
    if not structured.get("has_lesion"):
        return None

    return _format_nifti_metric_answer(structured, str(analysis.get("filename") or "uploaded_file"))


def _should_prioritize_uploaded_nifti(user_msg: str, uploaded_images: Optional[List[Dict[str, Any]]] = None) -> bool:
    if not uploaded_images or not _is_tumor_size_query(user_msg):
        return False
    for img in uploaded_images[:10]:
        meta = img.get("metadata", {}) if isinstance(img, dict) else {}
        if meta.get("saved_path"):
            return True
    return False


# ─────────────── helper functions ───────────────

def _rows_to_items(rows):
    from src.guardrails import RetrievalItem
    items = []
    for r in rows:
        facts = {k: r.get(k) for k in ["lesion_id", "patient_id", "wt_vox", "tc_vox",
                                         "et_vox", "et_pct", "tc_pct", "centroid", "bbox", "regions"]}
        items.append(RetrievalItem(
            source="neo4j", id=str(r.get("lesion_id")),
            score=float(r.get("score", 0.0)), facts=facts,
            text=f"lesion={facts['lesion_id']} patient={facts['patient_id']} score={float(r.get('score', 0.0)):.4f}",
        ))
    items.sort(key=lambda x: x.score, reverse=True)
    return items


def _compute_diffs(rows, query_lesion_id):
    def _to_float(v, default=0.0) -> float:
        if v is None:
            return default
        try:
            return float(v)
        except Exception:
            return default

    # pick query row if present; else fallback to first row
    q = next((r for r in rows if r.get("lesion_id") == query_lesion_id or r.get("id") == query_lesion_id), None)
    if q is None:
        q = rows[0] if rows else {}

    q_et = _to_float(q.get("et_pct"), 0.0)
    q_tc = _to_float(q.get("tc_pct"), 0.0)

    out = []
    for r in rows:
        et = _to_float(r.get("et_pct"), 0.0)
        tc = _to_float(r.get("tc_pct"), 0.0)

        out.append({
            "lesion_id": r.get("lesion_id") or r.get("id") or "",
            "patient_id": r.get("patient_id", ""),
            "score": _to_float(r.get("score"), 0.0),
            "et_pp": et * 100,
            "tc_pp": tc * 100,
            "et_delta_pp": (et - q_et) * 100,
            "tc_delta_pp": (tc - q_tc) * 100,
            "et_abs_pp": abs(et - q_et) * 100,
            "tc_abs_pp": abs(tc - q_tc) * 100,
            "et_pct": et,
            "tc_pct": tc,
        })

    out.sort(key=lambda x: (-x["score"], str(x["lesion_id"])))
    return out, {
        "lesion_id": (q.get("lesion_id") or q.get("id")),
        "patient_id": q.get("patient_id"),
        "et_pct": q_et,
        "tc_pct": q_tc,
    }


def _format_facts(diffs, q):
    lines = [
        f"QUERY lesion={q['lesion_id']} patient={q['patient_id']}",
        f"QUERY ET%={q['et_pct']*100:.2f}%  TC%={q['tc_pct']*100:.2f}%", "",
    ]
    for i, d in enumerate(diffs, 1):
        lines.append(
            f"[{i}] lesion={d['lesion_id']} score={d['score']:.4f} "
            f"ET%={d['et_pp']:.2f}% ΔET={d['et_delta_pp']:+.2f}pp "
            f"TC%={d['tc_pp']:.2f}% ΔTC={d['tc_delta_pp']:+.2f}pp"
        )
    return "\n".join(lines)


def _format_case_ids(rows, query_lesion_id: str, top_k: int) -> str:
    """Build a compact ID trace shown directly inside the assistant response."""
    diffs, q = _compute_diffs(rows, query_lesion_id)
    query_lid = q.get("lesion_id") or query_lesion_id
    query_pid = q.get("patient_id") or "unknown"

    lines = [
        "Case IDs:",
        f"- Query: lesion_id={query_lid} | patient_id={query_pid}",
    ]

    shown = 0
    for d in diffs:
        lid = d.get("lesion_id")
        if not lid or lid == query_lid:
            continue
        pid = d.get("patient_id") or "unknown"
        lines.append(f"- Retrieved: lesion_id={lid} | patient_id={pid}")
        shown += 1
        if shown >= max(0, top_k - 1):
            break

    return "\n".join(lines)


def _detect_lesion_id(text: str) -> Optional[str]:
    m = re.search(r"\b(lesion\d+)\b", text, re.IGNORECASE)
    return m.group(1) if m else None


def _is_rag_query(text: str) -> bool:
    rag_keywords = [
        "compare", "similar", "retriev", "analyz", "ET%", "TC%", "tumor",
        "enhancing", "lesion", "rag", "knowledge graph", "closest", "match",
        "difference", "delta",
    ]
    return any(kw in text.lower() for kw in rag_keywords)


# ═══════════════════════════════════════════════════
#  UI RENDERING
# ═══════════════════════════════════════════════════

def render_confidence_badge(confidence: str):
    colors = {"High": "#28a745", "Moderate": "#ffc107", "Low": "#dc3545"}
    color = colors.get(confidence, "#6c757d")
    st.markdown(
        f'<span style="background-color:{color}; color:white; padding:4px 12px; '
        f'border-radius:12px; font-weight:bold;">'
        f'Confidence: {confidence}</span>', unsafe_allow_html=True,
    )


def render_guardrail_pills(guardrails: Dict[str, str]):
    pills = []
    for label in ["input", "retrieval", "output"]:
        status = guardrails.get(label, "n/a")
        icon = "✅" if status == "allowed" else "❌"
        pills.append(f"{icon} {label.title()}")
    st.caption("Guardrails: " + "  |  ".join(pills))


def render_uploaded_files_plot(images: List[Dict[str, Any]]) -> None:
    """Visual summary for up to 10 uploaded files."""
    if not images:
        return
    try:
        import pandas as pd
        import plotly.express as px
    except ImportError:
        return

    rows = []
    for img in images[:10]:
        meta = img.get("metadata") or {}
        shape = meta.get("shape")
        shape_z = None
        if isinstance(shape, list) and len(shape) >= 3:
            try:
                shape_z = int(shape[2])
            except Exception:
                shape_z = None

        size_mb = round(float(img.get("size_bytes", 0)) / (1024.0 * 1024.0), 3)
        rows.append({
            "filename": img.get("filename", "unknown"),
            "size_mb": size_mb,
            "channels": int(meta.get("channels", 1)) if str(meta.get("channels", 1)).isdigit() else 1,
            "depth_z": shape_z,
        })

    df = pd.DataFrame(rows)

    fig1 = px.bar(
        df,
        x="filename",
        y="size_mb",
        color="channels",
        title="Uploaded File Size (MB)",
        labels={"size_mb": "Size (MB)", "filename": "File"},
    )
    fig1.update_layout(height=280, margin=dict(t=50, b=40))
    st.plotly_chart(fig1, width="stretch")

    if df["depth_z"].notna().any():
        fig2 = px.scatter(
            df[df["depth_z"].notna()],
            x="filename",
            y="depth_z",
            size="size_mb",
            color="channels",
            title="Volume Depth (Z) by File",
            labels={"depth_z": "Depth Z", "filename": "File"},
        )
        fig2.update_layout(height=280, margin=dict(t=50, b=40))
        st.plotly_chart(fig2, width="stretch")


def render_comparison_chart(diffs, query_id, key_salt=""):
    non_query = [d for d in diffs if d["lesion_id"] != query_id]
    if not non_query:
        return

    labels = [d["lesion_id"] for d in non_query]
    et_vals = [d["et_pp"] for d in non_query]
    tc_vals = [d["tc_pp"] for d in non_query]
    det_vals = [d["et_delta_pp"] for d in non_query]

    try:
        import plotly.graph_objects as go

        fig = go.Figure(data=[
            go.Bar(name="ET %", x=labels, y=et_vals, marker_color="#e74c3c"),
            go.Bar(name="TC %", x=labels, y=tc_vals, marker_color="#3498db"),
            go.Bar(name="ΔET (pp)", x=labels, y=det_vals, marker_color="#f39c12"),
        ])

        fig.update_layout(barmode="group", height=350, margin=dict(t=30, b=30))
        raw = f"{key_salt}|{query_id}|{'_'.join(labels)}"
        chart_key = "cmp_" + hashlib.md5(raw.encode()).hexdigest()
        st.plotly_chart(fig, width="stretch", key=chart_key)
        return
    except Exception:
        pass

    try:
        import matplotlib.pyplot as plt

        x = np.arange(len(labels))
        width = 0.26
        fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.3), 4))
        ax.bar(x - width, et_vals, width=width, label="ET %", color="#e74c3c")
        ax.bar(x, tc_vals, width=width, label="TC %", color="#3498db")
        ax.bar(x + width, det_vals, width=width, label="ΔET (pp)", color="#f39c12")
        ax.axhline(0, color="#777", linewidth=1)
        ax.set_title(f"Lesion Comparison (Query: {query_id})")
        ax.set_ylabel("Percent / Delta (pp)")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.legend()
        fig.tight_layout()
        st.pyplot(fig, clear_figure=True)
        plt.close(fig)
        return
    except Exception:
        pass

    import pandas as pd
    fallback_df = pd.DataFrame({
        "ET %": et_vals,
        "TC %": tc_vals,
        "ΔET (pp)": det_vals,
    }, index=labels)
    st.bar_chart(fallback_df)


def render_clinical_profile_plot(diffs, query_id, key_salt=""):
    """Clinical ET/TC profile across retrieved lesions with query reference lines."""
    non_query = [d for d in diffs if d["lesion_id"] != query_id]
    if not non_query:
        return

    query_row = next((d for d in diffs if d["lesion_id"] == query_id), None)
    if query_row is None:
        query_row = diffs[0]

    labels = [d["lesion_id"] for d in non_query]
    et_vals = [d["et_pp"] for d in non_query]
    tc_vals = [d["tc_pp"] for d in non_query]
    q_et = float(query_row.get("et_pp", 0.0))
    q_tc = float(query_row.get("tc_pp", 0.0))

    try:
        import plotly.graph_objects as go

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=labels, y=et_vals, mode="lines+markers",
                name="ET %", line=dict(color="#e74c3c", width=2),
                marker=dict(size=8),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=labels, y=tc_vals, mode="lines+markers",
                name="TC %", line=dict(color="#3498db", width=2),
                marker=dict(size=8),
            )
        )

        fig.add_hline(
            y=q_et,
            line_dash="dash",
            line_color="#e74c3c",
            annotation_text=f"Query ET% {q_et:.2f}",
            annotation_position="top left",
        )
        fig.add_hline(
            y=q_tc,
            line_dash="dash",
            line_color="#3498db",
            annotation_text=f"Query TC% {q_tc:.2f}",
            annotation_position="bottom left",
        )

        fig.update_layout(
            title=f"Clinical ET/TC Profile (Query: {query_id})",
            yaxis_title="Percent (%)",
            xaxis_title="Retrieved Lesion",
            height=360,
            margin=dict(t=45, b=35),
        )

        raw = f"{key_salt}|clinical|{query_id}|{'_'.join(labels)}"
        chart_key = "clinical_" + hashlib.md5(raw.encode()).hexdigest()
        st.plotly_chart(fig, width="stretch", key=chart_key)
        return
    except Exception:
        pass

    try:
        import matplotlib.pyplot as plt

        x = np.arange(len(labels))
        fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.3), 4))
        ax.plot(x, et_vals, marker="o", color="#e74c3c", label="ET %")
        ax.plot(x, tc_vals, marker="o", color="#3498db", label="TC %")
        ax.axhline(q_et, linestyle="--", color="#e74c3c", linewidth=1.3, label=f"Query ET% {q_et:.2f}")
        ax.axhline(q_tc, linestyle="--", color="#3498db", linewidth=1.3, label=f"Query TC% {q_tc:.2f}")
        ax.set_title(f"Clinical ET/TC Profile (Query: {query_id})")
        ax.set_ylabel("Percent (%)")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.legend()
        fig.tight_layout()
        st.pyplot(fig, clear_figure=True)
        plt.close(fig)
        return
    except Exception:
        pass

    import pandas as pd
    fallback_df = pd.DataFrame({"ET %": et_vals, "TC %": tc_vals}, index=labels)
    st.line_chart(fallback_df)


def render_rag_response(result: Dict[str, Any], msg_index: int | None = None):
    """Render a full RAG response inline in the chat."""
    if result["status"] != "success":
        if result["status"] == "llm_error":
            st.warning(result.get("llm_answer") or "Query failed: llm_error")
        else:
            st.error(f"Query failed: {result['status']}")
        if result.get("guardrails"):
            st.json(result["guardrails"])
        return

    col1, col2 = st.columns([1, 3])
    with col1:
        render_confidence_badge(result.get("confidence", ""))
    with col2:
        render_guardrail_pills(result.get("guardrails", {}))

    st.markdown(result.get("llm_answer", ""))
    if result.get("confidence_rationale"):
        st.caption(f"*{result['confidence_rationale']}*")

    with st.expander("📊 Lesion Comparison Chart"):
        if result.get("retrieved_rows"):
            diffs, q = _compute_diffs(result["retrieved_rows"], result["lesion_id"])
            non_query = [d for d in diffs if d["lesion_id"] != result["lesion_id"]]
            if non_query:
                import pandas as pd
                df = pd.DataFrame([{
                    "Lesion": d["lesion_id"], "Patient": d["patient_id"],
                    "Score": f"{d['score']:.4f}",
                    "ET%": f"{d['et_pp']:.2f}%", "ΔET": f"{d['et_delta_pp']:+.2f}pp",
                    "TC%": f"{d['tc_pp']:.2f}%", "ΔTC": f"{d['tc_delta_pp']:+.2f}pp",
                } for d in non_query])
                st.dataframe(df, width="stretch", hide_index=True)
                salt = f"msg{msg_index}" if msg_index is not None else f"ts{time.time_ns()}"
                render_comparison_chart(diffs, result["lesion_id"], key_salt=salt)
                render_clinical_profile_plot(diffs, result["lesion_id"], key_salt=salt)

    with st.expander("📋 Retrieved Evidence & Facts"):
        role = st.session_state.get("user_role", "viewer")
        can_view_context = check_role_permission(role, "view_raw_context", ETHICAL_POLICY).allowed
        if can_view_context or st.session_state.get("hitl_approved"):
            if result.get("facts"):
                st.code(result["facts"], language="text")
            if result.get("context"):
                st.code(result["context"], language="text")
        else:
            st.info("Raw context is restricted by least-privilege policy. Use analyst/admin role or enable HITL approval.")

    with st.expander("⏱️ Pipeline Timing"):
        timing = result.get("timing", {})
        if timing:
            cols = st.columns(min(len(timing) + 1, 5))
            for col, (step, t) in zip(cols, timing.items()):
                col.metric(step.replace("_", " ").title(), f"{t:.2f}s")
            if len(cols) > len(timing):
                cols[-1].metric("Total", f"{sum(timing.values()):.2f}s")


# ═══════════════════════════════════════════════════
#  MAIN APP
# ═══════════════════════════════════════════════════

def main():
    _apply_light_theme()

    # ─── Header ───
    st.markdown("""
    <div style="text-align:center; padding: 5px 0 15px 0;">
        <h1 style="margin-bottom:0;">🧠 MedRAG-X</h1>
        <p style="color:#888; font-size:15px; margin-top:5px;">
        MRI-Aware Clinical Reasoning Chat · Multimodal Knowledge Graphs
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ─── Sidebar ───
    with st.sidebar:
        st.header("⚙️ Settings")
        top_k = st.slider("Top-K Retrieval", 3, 20, 5)
        min_score = st.slider("Min Similarity Score", 0.50, 0.99, 0.70, 0.05)

        st.divider()
        st.subheader("🛡️ Ethical Guardrails")
        if st.session_state.get("user_role") not in {"viewer", "analyst", "admin"}:
            st.session_state.user_role = "analyst"
        st.selectbox(
            "Role",
            options=["viewer", "analyst", "admin"],
            key="user_role",
            help="Least privilege is enforced by role.",
        )
        purpose_options = sorted(ETHICAL_POLICY.allowed_purposes) or ["clinical_reasoning"]
        if st.session_state.get("user_purpose") not in purpose_options and purpose_options:
            st.session_state.user_purpose = purpose_options[0]
        st.selectbox(
            "Purpose",
            options=purpose_options,
            key="user_purpose",
            help="Requests are allowed only for approved purposes.",
        )
        st.checkbox(
            "HITL approval for risky actions",
            key="hitl_approved",
            help="Required for export/share/raw-context style prompts or prompts containing secrets.",
        )
        vendor_ok = check_vendor_governance(OLLAMA_URL, OLLAMA_MODEL, ETHICAL_POLICY).allowed
        st.caption(f"Vendor governance: {'✅ allowed' if vendor_ok else '❌ blocked'}")
        cleanup_stats = st.session_state.get("retention_cleanup_stats") or {}
        if cleanup_stats:
            st.caption(
                f"Retention cleanup: uploads={cleanup_stats.get('uploads_removed', 0)} | "
                f"audit={cleanup_stats.get('audit_entries_removed', 0)}"
            )

        st.divider()
        st.subheader("🗄️ Neo4j Connection")
        st.text_input(
            "Neo4j Password",
            key="neo4j_password",
            type="password",
            help="Used for lesion retrieval from Neo4j (same as NEO4J_PASSWORD env var).",
        )
        if not _current_neo4j_password():
            st.warning("Set Neo4j password above or export NEO4J_PASSWORD. Local default is neo4j123.")

        st.divider()

        # ── Lesion selector ──
        st.subheader("🔬 Quick Lesion Select")
        try:
            lesion_data = fetch_lesion_ids()
            if lesion_data:
                options = {f"{r['id']} ({r['patient_id']})": r['id'] for r in lesion_data}
                lesion_filter = st.text_input(
                    "Filter lesion list",
                    placeholder="Type part of lesion ID or patient ID...",
                    key="quick_lesion_filter",
                )
                option_keys = list(options.keys())
                if lesion_filter:
                    q = lesion_filter.strip().lower()
                    option_keys = [k for k in option_keys if q in k.lower()]

                selected = st.selectbox(
                    "Pick a lesion",
                    ["(none)"] + option_keys,
                    key="quick_lesion_select",
                )

                if selected != "(none)":
                    st.session_state.selected_lesion = options[selected]
                    details = fetch_lesion_details(st.session_state.selected_lesion)
                    if details:
                        st.caption(f"Patient: {details.get('patient_id')}")
                        et = details.get("et_pct")
                        tc = details.get("tc_pct")
                        if et is not None:
                            st.caption(f"ET%: {float(et)*100:.2f}%  |  TC%: {float(tc or 0)*100:.2f}%")
                elif st.session_state.get("selected_lesion"):
                    st.caption(f"Current selection: {st.session_state['selected_lesion']}")

                if st.button("📎 Send selected lesion to chat"):
                    if st.session_state.selected_lesion:
                        lid = st.session_state.selected_lesion
                        st.session_state.messages.append({
                            "role": "user",
                            "text": f"Compare {lid} to its most similar lesions and explain ET% differences.",
                            "lesion_id": lid,
                            "trigger_rag": True,
                        })
                        st.rerun()
            else:
                st.warning("No lesions in Neo4j")
        except Exception as e:
            st.error(f"Neo4j: {e}")

        st.divider()

        # ── Image upload ──
        st.subheader("🖼️ Upload MRI Image")
        uploaded_files = st.file_uploader(
            "Drag & drop or browse",
            type=["png", "jpg", "jpeg", "npy", "nii", "gz", "tiff", "tif"],
            accept_multiple_files=True,
            help="PNG/JPG slice, NumPy (.npy), or NIfTI (.nii/.nii.gz)",
        )
        if uploaded_files:
            if len(uploaded_files) > 10:
                st.warning(f"You uploaded {len(uploaded_files)} files. Only the first 10 will be analyzed.")
                uploaded_files = uploaded_files[:10]

            processed_images: List[Dict[str, Any]] = []
            for file_idx, uploaded_file in enumerate(uploaded_files):
                img_result = process_uploaded_image(uploaded_file)
                processed_images.append(img_result)

                st.markdown(f"**File {file_idx + 1}: {img_result.get('filename', f'file_{file_idx + 1}')}**")

                # ── Show preview image ──
                if img_result.get("display_image"):
                    st.image(img_result["display_image"], caption=img_result["filename"], width="stretch")

                # ── Show metadata ──
                if img_result.get("metadata"):
                    st.json(img_result["metadata"])

                # ==========================================================
                # 🧩 Channel / modality UI (ONLY for 4D NIfTI)
                # ==========================================================
                meta = img_result.get("metadata", {})
                file_key = "upl_" + hashlib.md5(f"{file_idx}_{img_result.get('filename', '')}".encode()).hexdigest()[:8]

                if meta.get("channels", 1) > 1 and meta.get("saved_path"):
                    n_ch = int(meta["channels"])
                    default_labels = ["FLAIR", "T1", "T1ce", "T2"]
                    labels = []
                    for i in range(n_ch):
                        default = default_labels[i] if i < len(default_labels) else f"CH{i}"
                        labels.append(
                            st.text_input(
                                f"Label for channel {i} ({img_result.get('filename', '')})",
                                value=default,
                                key=f"mod_lbl_{file_key}_{i}",
                            )
                        )

                    selected_ch = st.selectbox(
                        f"Preview channel ({img_result.get('filename', '')})",
                        options=list(range(n_ch)),
                        format_func=lambda i: f"{i} — {labels[i]}",
                        key=f"selected_channel_idx_{file_key}",
                    )

                    try:
                        _, data, _ = load_nifti(Path(meta["saved_path"]))
                        z_mid = data.shape[2] // 2
                        slice_2d = data[:, :, z_mid, selected_ch]

                        img_result["display_image"] = _numpy_to_png(slice_2d)
                        img_result["numpy_slice"] = slice_2d
                        img_result["metadata"]["selected_channel"] = int(selected_ch)
                        img_result["metadata"]["channel_labels"] = labels

                        st.image(
                            img_result["display_image"],
                            caption=f"Preview: {labels[selected_ch]} (ch={selected_ch})",
                            width=320,
                        )
                    except Exception as e:
                        st.warning(f"Could not preview selected channel: {e}")

                st.divider()

            with st.expander("📈 Uploaded Files Plot", expanded=True):
                render_uploaded_files_plot(processed_images)

            st.session_state.pending_images = processed_images
            st.session_state.pending_image = processed_images[0] if processed_images else None
            st.caption(f"✅ {len(processed_images)} file(s) ready — all will be attached to your next message.")
        else:
            st.session_state.pending_images = []
            st.session_state.pending_image = None

        st.divider()
        st.caption(f"Neo4j: `{NEO4J_URI}`  |  LLM: `{OLLAMA_MODEL}`")

        if st.button("🗑️ Clear chat history"):
            st.session_state.messages = []
            st.session_state.last_rag_result = None
            st.session_state.pending_images = []
            st.session_state.pending_image = None
            st.rerun()




    # ─── Chat History ───
    for i, msg in enumerate(st.session_state.messages):
        role = msg["role"]
        with st.chat_message(role, avatar="🧑‍⚕️" if role == "user" else "🧠"):
            # Show image(s) if attached
            if msg.get("images"):
                for img in msg.get("images", [])[:10]:
                    if img.get("display_image"):
                        st.image(img["display_image"], caption=img.get("filename", "MRI"), width=300)
            elif msg.get("image") and msg["image"].get("display_image"):
                st.image(msg["image"]["display_image"], caption=msg["image"].get("filename", "MRI"), width=300)
            # Show text
            text = msg.get("text", "")
            if text:
                st.markdown(text)
            # Show RAG result
            if msg.get("rag_result"):
                render_rag_response(msg["rag_result"])

    # ─── Chat Input ───
    user_input = st.chat_input("Ask about brain MRI, lesions, or BraTS analysis...")

    if user_input:
        role = st.session_state.get("user_role", "viewer")
        purpose = st.session_state.get("user_purpose", "")
        approved = bool(st.session_state.get("hitl_approved", False))
        safe_user_input, query_meta = minimize_query_text(user_input, ETHICAL_POLICY)

        purpose_check = check_purpose_allowed(purpose, ETHICAL_POLICY)
        if not purpose_check.allowed:
            msg = f"Blocked by purpose limitation: {purpose_check.reason}"
            st.error(msg)
            _audit("chat", "deny", role=role, purpose=purpose, details={"reason": purpose_check.reason, "query_hash": query_meta.get("sha256")})
            st.stop()

        role_check = check_role_permission(role, "chat", ETHICAL_POLICY)
        if not role_check.allowed:
            msg = f"Blocked by access control: {role_check.reason}"
            st.error(msg)
            _audit("chat", "deny", role=role, purpose=purpose, details={"reason": role_check.reason, "query_hash": query_meta.get("sha256")})
            st.stop()

        hitl_check = check_hitl_approval(safe_user_input, approved)
        if not hitl_check.allowed:
            with st.chat_message("assistant", avatar="🧠"):
                st.markdown(f"⚠️ {hitl_check.reason}")
            st.session_state.messages.append({"role": "assistant", "text": hitl_check.reason})
            _audit("chat", "deny", role=role, purpose=purpose, details={"reason": hitl_check.reason, "query_hash": query_meta.get("sha256")})
            st.stop()

        # Grab pending images
        attached_images = st.session_state.get("pending_images", [])[:10]
        image_metas = [img.get("metadata", {}) for img in attached_images if isinstance(img, dict)]

        # Add user message
        user_msg = {"role": "user", "text": safe_user_input}
        if attached_images:
            user_msg["images"] = attached_images

        st.session_state.messages.append(user_msg)

        # Display immediately
        with st.chat_message("user", avatar="🧑‍⚕️"):
            for img in attached_images:
                if img.get("display_image"):
                    st.image(img["display_image"], caption=img.get("filename", "MRI"), width=300)
            st.markdown(safe_user_input)

        # ╔══════════════════════════════════════════════════╗
        # ║  TOPIC GUARDRAIL — blocks non-medical queries    ║
        # ╚══════════════════════════════════════════════════╝
        from src.guardrails.clinical_guardrails import topic_guardrail

        topic_check = topic_guardrail(safe_user_input)
        if topic_check.action != "allow":
            with st.chat_message("assistant", avatar="🧠"):
                st.markdown(topic_check.safe_reply or
                            "I can only assist with brain MRI analysis and BraTS-related queries.")
            st.session_state.messages.append({
                "role": "assistant",
                "text": topic_check.safe_reply or "Off-topic query blocked by guardrail.",
            })
            _audit("chat", "deny", role=role, purpose=purpose, details={"reason": topic_check.reason, "query_hash": query_meta.get("sha256")})
            st.session_state.pending_images = []
            st.session_state.pending_image = None
            st.stop()

        # ── Decide: RAG or general chat ──
        lesion_id = _detect_lesion_id(safe_user_input) or st.session_state.selected_lesion
        should_rag = _is_rag_query(safe_user_input) and lesion_id is not None
        if _should_prioritize_uploaded_nifti(safe_user_input, attached_images):
            should_rag = False

        # Also trigger RAG if message was sent from sidebar button
        if st.session_state.messages and st.session_state.messages[-1].get("trigger_rag"):
            should_rag = True
            lesion_id = st.session_state.messages[-1].get("lesion_id", lesion_id)

        with st.chat_message("assistant", avatar="🧠"):
            if should_rag and lesion_id:
                rag_role_check = check_role_permission(role, "rag_query", ETHICAL_POLICY)
                if not rag_role_check.allowed:
                    st.error(f"Blocked by access control: {rag_role_check.reason}")
                    _audit("rag_query", "deny", role=role, purpose=purpose, details={"reason": rag_role_check.reason, "query_hash": query_meta.get("sha256")})
                    st.stop()
                with st.spinner(f"Running clinical RAG for **{lesion_id}**..."):
                    result = run_guarded_rag(
                        lesion_id=lesion_id, top_k=top_k, min_score=min_score,
                        user_query=safe_user_input, image_metas=image_metas, role=role, purpose=purpose,
                    )
                st.session_state.messages.append({
                    "role": "assistant", "text": "", "rag_result": result,
                })
                st.session_state.last_rag_result = result
                st.session_state["live_result"] = result
                render_rag_response(result)
            else:
                with st.spinner("Thinking..."):
                    response, llm_error = handle_general_chat(
                        safe_user_input,
                        image_metas=image_metas,
                        uploaded_images=attached_images,
                    )
                if llm_error:
                    safe_error, _ = redact_sensitive_text(llm_error)
                    msg = f"LLM unavailable: {safe_error}"
                    st.warning(msg)
                    st.session_state.messages.append({"role": "assistant", "text": msg})
                    _audit("chat", "error", role=role, purpose=purpose, details={"reason": "llm_error", "llm_error": safe_error, "query_hash": query_meta.get("sha256")})
                else:
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "text": response})
                    _audit(
                        "chat",
                        "allow",
                        role=role,
                        purpose=purpose,
                        details={
                            "query_hash": query_meta.get("sha256"),
                            "query_truncated": query_meta.get("truncated"),
                            "query_redacted": query_meta.get("redacted"),
                        },
                    )

        # Clear pending images after use
        st.session_state.pending_images = []
        st.session_state.pending_image = None

    # ─── Welcome screen ───

    # ─── RAG dashboard ───
    with st.expander("RAG Dashboard", expanded=False):
        live_result = st.session_state.get("live_result") or st.session_state.get("last_rag_result")
        render_rag_dashboard_ui(live_result, top_k=top_k, min_score=min_score)

    if not st.session_state.messages:
        st.markdown("""
        <div style="text-align:center; padding:40px; color:#888;">
            <h3>Welcome to MedRAG-X 🧠</h3>
            <p style="margin-top:15px;">I'm a clinical reasoning assistant for <b>brain tumor MRI analysis</b>.</p>
            <p>I can help with:</p>
            <p>💬 <b>Lesion comparison</b> — <code>Compare lesion3 to similar lesions</code></p>
            <p>🖼️ <b>MRI image analysis</b> — upload a brain MRI in the sidebar</p>
            <p>📊 <b>Tumor metrics</b> — ET%, TC%, whole tumor volume</p>
            <p>🔬 <b>Knowledge graph queries</b> — retrieval, similarity, evidence</p>
            <hr style="width:50%; margin:20px auto;">
            <p style="font-size:13px; color:#aaa;">
            ⚠️ I <b>only</b> answer questions related to brain MRI, BraTS, and medical imaging.<br>
            Off-topic questions (sports, movies, general knowledge, etc.) will be declined.
            </p>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

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

# ─────────────────── Page Config ───────────────────
st.set_page_config(
    page_title="MedRAG-X",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────── Constants ───────────────────
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434/api/chat")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
UPLOAD_DIR = Path("artifacts/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════
#  Session state initialization
# ═══════════════════════════════════════════════════

def init_session():
    """Initialize all session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "pending_image" not in st.session_state:
        st.session_state.pending_image = None
    if "last_rag_result" not in st.session_state:
        st.session_state.last_rag_result = None
    if "selected_lesion" not in st.session_state:
        st.session_state.selected_lesion = None

init_session()


# ═══════════════════════════════════════════════════
#  Neo4j helpers (cached)
# ═══════════════════════════════════════════════════

@st.cache_resource
def get_retriever():
    if not NEO4J_PASSWORD:
        raise RuntimeError("Missing NEO4J_PASSWORD environment variable.")
    from src.rag.neo4j_retriever import Neo4jRetriever
    return Neo4jRetriever(uri=NEO4J_URI, user=NEO4J_USER, password=NEO4J_PASSWORD)


@st.cache_data(ttl=300)
def fetch_lesion_ids() -> List[Dict[str, str]]:
    retriever = get_retriever()
    with retriever.driver.session() as s:
        rows = s.run(
            "MATCH (l:Lesion) WHERE l.embedding IS NOT NULL "
            "RETURN l.id AS id, l.patient_id AS patient_id "
            "ORDER BY l.id"
        ).data()
    return rows


@st.cache_data(ttl=300)
def fetch_lesion_details(lesion_id: str) -> Optional[Dict[str, Any]]:
    retriever = get_retriever()
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
            import tempfile
            suffix = ".nii.gz" if filename.endswith(".nii.gz") else ".nii"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(raw_bytes)
                tmp_path = tmp.name
            nii = nib.load(tmp_path)
            data = nii.get_fdata()
            result["metadata"] = {
                "shape": list(data.shape),
                "dtype": str(data.dtype),
                "voxel_sizes": [round(float(v), 3) for v in nii.header.get_zooms()],
            }
            if data.ndim >= 3:
                mid = data.shape[2] // 2
                slice_2d = data[:, :, mid] if data.ndim == 3 else data[:, :, mid, 0]
                result["numpy_slice"] = slice_2d
                result["display_image"] = _numpy_to_png(slice_2d)
            os.unlink(tmp_path)
        except ImportError:
            result["metadata"] = {"note": "nibabel not installed; cannot preview NIfTI"}
        except Exception as e:
            result["metadata"] = {"error": str(e)}
        return result

    result["metadata"] = {"note": f"Unsupported format: {filename}"}
    return result


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

def call_ollama(prompt: str, model: str = OLLAMA_MODEL) -> str:
    import requests
    payload = {
        "model": model, "stream": False,
        "messages": [{"role": "user", "content": prompt}],
        "options": {"temperature": 0.2, "top_p": 0.9, "num_ctx": 4096},
    }
    try:
        r = requests.post(OLLAMA_URL, json=payload, timeout=600)
        r.raise_for_status()
        return r.json()["message"]["content"]
    except Exception as e:
        return f"⚠️ LLM Error: {e}"


def call_ollama_chat(messages: List[Dict[str, str]], model: str = OLLAMA_MODEL) -> str:
    """Multi-turn chat with conversation history."""
    import requests
    payload = {
        "model": model, "stream": False, "messages": messages,
        "options": {"temperature": 0.3, "top_p": 0.9, "num_ctx": 4096},
    }
    try:
        r = requests.post(OLLAMA_URL, json=payload, timeout=600)
        r.raise_for_status()
        return r.json()["message"]["content"]
    except Exception as e:
        return f"⚠️ LLM Error: {e}"

def compute_live_rag_metrics(result: dict, top_k: int | None = None) -> dict:
    """
    Build a stable metrics dict for UI.
    Works even if result has missing fields.
    """
    retrieved = len(result.get("retrieved", [])) if isinstance(result, dict) else 0

    if retrieved == 0:
        return {
            "status": "no_result",
            "grounding": 0.0,
            "hallucination_rate": 1.0,
            "retrieved": 0,
            "top_score": 0.0,
            "confidence": "Low",
            "coverage": 0.0,
        }

    status = result.get("status", "unknown")
    retrieved = result.get("retrieved") or []
    diffs = result.get("diffs") or []
    conf = result.get("confidence", None)

    if isinstance(conf, dict):
        confidence_label = conf.get("label") or conf.get("confidence") or "N/A"
    elif isinstance(conf, str):
        confidence_label = conf
    elif conf is None:
        confidence_label = "N/A"
    else:
        confidence_label = str(conf)
    top_score = 0.0
    if retrieved:
        try:
            top_score = float(retrieved[0].get("score", 0.0))
        except Exception:
            top_score = 0.0

    # Simple/robust "grounding" heuristic:
    # grounded if we have retrieved evidence AND the run succeeded.
    grounded = 1.0 if (status == "ok" and len(retrieved) > 0) else 0.0

    # Simple "hallucination" heuristic:
    # if not ok or no evidence, treat as higher hallucination risk.
    hallucination_rate = 0.0 if grounded == 1.0 else 1.0

    # Coverage: how many comparisons we actually produced vs requested top_k
    if top_k is None:
        coverage = 1.0 if retrieved else 0.0
    else:
        coverage = min(1.0, (len(diffs) / float(max(1, top_k))))

    return {
        "status": status,
        "grounding": grounded,
        "hallucination_rate": hallucination_rate,
        "retrieved": len(retrieved),
        "top_score": top_score,
        "confidence": confidence_label,
        "coverage": coverage,
    }


def load_offline_rag_eval_summary(path: str | Path) -> dict | None:
    """
    Load offline summary produced by run_rag_eval.py (rag_eval_summary.json).
    """
    p = Path(path)
    if not p.exists() or not p.is_file() or p.stat().st_size == 0:
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def render_rag_dashboard_ui(
    *,
    offline_summary_path: str | Path = "artifacts/rag_eval_results/rag_eval_summary.json",
) -> None:
    """
    Shows two sections:
      - Live (this query)
      - Offline (overall eval)
    """
    st.subheader("RAG Dashboard")

    live_result = st.session_state.get("last_rag_result", None)

    tab_live, tab_offline = st.tabs(["Live (this query)", "Offline (overall eval)"])

    with tab_live:
        if not live_result:
            st.info("Run a query first to see live RAG metrics.")
        else:
            # If you store top_k in session, use it; else None is fine.
            top_k = st.session_state.get("last_top_k", None)
            m = compute_live_rag_metrics(live_result, top_k=top_k)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Status", str(m.get("status", "unknown")))
            c2.metric("Grounding", f"{float(m.get('grounding', 0.0)):.3f}")
            c3.metric("Hallucination rate", f"{float(m.get('hallucination_rate', 1.0)):.3f}")
            c4.metric("Confidence", str(m.get("confidence", "N/A")))

            c5, c6, c7 = st.columns(3)
            c5.metric("Retrieved", int(m.get("retrieved", 0)))
            c6.metric("Top score", f"{float(m.get('top_score', 0.0)):.4f}")
            c7.metric("Coverage", f"{float(m.get('coverage', 0.0)):.3f}")

            # Optional: show evidence table
            retrieved = live_result.get("retrieved") or []
            if retrieved:
                st.markdown("**Top retrieved evidence**")
                rows = []
                for r in retrieved[:10]:
                    rows.append({
                        "lesion_id": r.get("lesion_id"),
                        "patient_id": r.get("patient_id"),
                        "score": r.get("score"),
                        "ET%": r.get("et_percent"),
                        "TC%": r.get("tc_percent"),
                    })
                st.dataframe(rows, use_container_width=True)

    with tab_offline:
        summary = load_offline_rag_eval_summary(offline_summary_path)
        if not summary:
            st.warning(
                "Offline eval summary not found.\n\n"
                "Run `python run_rag_eval.py ...` to generate:\n"
                f"`{offline_summary_path}`"
            )
        else:
            # Your run_rag_eval.py summary structure may differ; display safely.
            overall = summary.get("overall") or summary

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Queries evaluated", int(overall.get("num_queries", overall.get("n", 0)) or 0))
            c2.metric("Avg grounding", f"{float(overall.get('avg_grounding', overall.get('grounding', 0.0)) or 0.0):.3f}")
            c3.metric("Avg hallucination", f"{float(overall.get('avg_hallucination_rate', overall.get('hallucination_rate', 0.0)) or 0.0):.3f}")
            c4.metric("Pass rate", f"{float(overall.get('pass_rate', 0.0) or 0.0):.3f}")

            with st.expander("Show offline summary JSON", expanded=False):
                st.json(summary)

# ═══════════════════════════════════════════════════
#  RAG Pipeline
# ═══════════════════════════════════════════════════

def run_guarded_rag(lesion_id: str, top_k: int = 5, min_score: float = 0.70,
                    user_query: str = "", image_meta: Optional[Dict] = None) -> Dict[str, Any]:
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
        "retrieved_rows": [], "timing": {},
    }

    base_query = f"Compare {lesion_id} to its most similar lesions and explain ET% differences."
    if user_query:
        base_query = f"{user_query} (Reference lesion: {lesion_id})"
    if image_meta:
        base_query += f" [Uploaded MRI: {image_meta.get('filename', 'image')}, shape={image_meta.get('shape', 'unknown')}]"
    query = base_query

    # (1) INPUT GUARDRAIL
    t0 = time.time()
    d0 = input_guardrail(query)
    result["timing"]["input_guardrail"] = time.time() - t0
    if d0.action != "allow":
        result["status"] = "blocked_input"
        result["guardrails"]["input"] = d0.reason or "Blocked"
        return result
    result["guardrails"]["input"] = "allowed"

    # (2) RETRIEVAL
    t1 = time.time()
    retriever = get_retriever()
    rows = retriever.retrieve_similar(lesion_id=lesion_id, k=top_k)
    result["timing"]["retrieval"] = time.time() - t1
    result["retrieved_rows"] = rows
    if not rows:
        result["status"] = "no_results"
        return result

    # (3) RETRIEVAL GUARDRAIL
    items = _rows_to_items(rows)
    bundle = RetrievalBundle(query=query, items=items, k=top_k, min_score=min_score)
    d1 = retrieval_guardrail(bundle)
    if d1.action != "allow":
        result["status"] = "blocked_retrieval"
        result["guardrails"]["retrieval"] = d1.reason or "Blocked"
        return result
    result["guardrails"]["retrieval"] = "allowed"

    # (4) Deterministic facts
    diffs, q = _compute_diffs(rows, lesion_id)
    facts_block = _format_facts(diffs, q)
    context_block = retriever.build_context(rows)
    result["context"] = context_block
    result["facts"] = facts_block

    # (5) Prompt
    required_lesions = [d["lesion_id"] for d in diffs if d["lesion_id"] != lesion_id][:max(0, top_k - 1)]
    evidence_lines = "\n".join([
        f"• [{i+1}] lesion={d['lesion_id']}"
        for i, d in enumerate(diffs) if d["lesion_id"] != lesion_id
    ][:max(0, top_k - 1)])

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
    user_prompt = USER_TEMPLATE.format(query=query)
    if user_query:
        user_prompt += f"\n\nAdditional user context: {user_query}"
    prompt = f"{system}\n\nFACTS:\n{facts_block}\n\nRAW CONTEXT:\n{context_block}\n\n{user_prompt}"

    t2 = time.time()
    llm_answer = call_ollama(prompt)
    result["timing"]["llm"] = time.time() - t2

    # (6) OUTPUT GUARDRAIL
    d2 = output_guardrail(llm_answer)
    if d2.action != "allow":
        result["status"] = "blocked_output"
        result["guardrails"]["output"] = d2.reason or "Blocked"
        return result
    result["guardrails"]["output"] = "allowed"

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
    result["llm_answer"] = llm_answer
    result["confidence"] = conf
    result["confidence_rationale"] = rationale
    return result

# ─────────────────── RAG Metrics Loader ───────────────────
def load_rag_metrics():
    path = Path("artifacts/rag_eval_results/rag_eval_summary.json")
    if not path.exists():
        return None
    return json.loads(path.read_text())


def handle_general_chat(user_msg: str, image_meta: Optional[Dict] = None) -> str:
    """Handle on-topic medical chat that doesn't trigger the full RAG pipeline."""
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
    if image_meta:
        system_context += (
            f"\nThe user uploaded an MRI image: {image_meta.get('filename', 'image')}. "
            f"Metadata: {json.dumps({k: v for k, v in image_meta.items() if k not in ('display_image', 'numpy_slice')}, default=str)}. "
            "Acknowledge the image and provide relevant observations about what can be "
            "analyzed (format, dimensions, modality if identifiable).\n"
        )

    history_messages = [{"role": "system", "content": system_context}]
    recent = st.session_state.messages[-10:]
    for msg in recent:
        role = "user" if msg["role"] == "user" else "assistant"
        content = msg.get("text", "")
        if content:
            history_messages.append({"role": role, "content": content})
    history_messages.append({"role": "user", "content": user_msg})
    return call_ollama_chat(history_messages)


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
    q = next((r for r in rows if r.get("lesion_id") == query_lesion_id), rows[0] if rows else {})
    q_et, q_tc = float(q.get("et_pct", 0)), float(q.get("tc_pct", 0))
    out = []
    for r in rows:
        et, tc = float(r.get("et_pct", 0)), float(r.get("tc_pct", 0))
        out.append({
            "lesion_id": r.get("lesion_id", ""), "patient_id": r.get("patient_id", ""),
            "score": float(r.get("score", 0)),
            "et_pp": et * 100, "tc_pp": tc * 100,
            "et_delta_pp": (et - q_et) * 100, "tc_delta_pp": (tc - q_tc) * 100,
            "et_abs_pp": abs(et - q_et) * 100, "tc_abs_pp": abs(tc - q_tc) * 100,
            "et_pct": et, "tc_pct": tc,
        })
    out.sort(key=lambda x: (-x["score"], str(x["lesion_id"])))
    return out, {"lesion_id": q.get("lesion_id"), "patient_id": q.get("patient_id"), "et_pct": q_et, "tc_pct": q_tc}


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


def render_comparison_chart(diffs, query_id):
    try:
        import plotly.graph_objects as go
        import hashlib   # ← add import here (or at top of file)
    except ImportError:
        return

    non_query = [d for d in diffs if d["lesion_id"] != query_id]
    if not non_query:
        return

    labels = [d["lesion_id"] for d in non_query]

    fig = go.Figure(data=[
        go.Bar(name="ET %", x=labels, y=[d["et_pp"] for d in non_query], marker_color="#e74c3c"),
        go.Bar(name="TC %", x=labels, y=[d["tc_pp"] for d in non_query], marker_color="#3498db"),
        go.Bar(name="ΔET (pp)", x=labels, y=[d["et_delta_pp"] for d in non_query], marker_color="#f39c12"),
    ])

    fig.update_layout(barmode="group", height=350, margin=dict(t=30, b=30))

    # ✅ ADD THIS BLOCK RIGHT BEFORE st.plotly_chart
    raw = f"{query_id}_{'_'.join(labels)}"
    chart_key = "cmp_" + hashlib.md5(raw.encode()).hexdigest()

    st.plotly_chart(fig, use_container_width=True, key=chart_key)


def render_rag_response(result: Dict[str, Any]):
    """Render a full RAG response inline in the chat."""
    if result["status"] != "success":
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
                st.dataframe(df, use_container_width=True, hide_index=True)
                render_comparison_chart(diffs, result["lesion_id"])

    with st.expander("📋 Retrieved Evidence & Facts"):
        if result.get("facts"):
            st.code(result["facts"], language="text")
        if result.get("context"):
            st.code(result["context"], language="text")

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

        # ── Lesion selector ──
        st.subheader("🔬 Quick Lesion Select")
        try:
            lesion_data = fetch_lesion_ids()
            if lesion_data:
                options = {f"{r['id']} ({r['patient_id']})": r['id'] for r in lesion_data}
                selected = st.selectbox("Pick a lesion", ["(none)"] + list(options.keys()))
                if selected != "(none)":
                    st.session_state.selected_lesion = options[selected]
                    details = fetch_lesion_details(st.session_state.selected_lesion)
                    if details:
                        st.caption(f"Patient: {details.get('patient_id')}")
                        et = details.get("et_pct")
                        tc = details.get("tc_pct")
                        if et is not None:
                            st.caption(f"ET%: {float(et)*100:.2f}%  |  TC%: {float(tc or 0)*100:.2f}%")
                else:
                    st.session_state.selected_lesion = None

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
        uploaded_file = st.file_uploader(
            "Drag & drop or browse",
            type=["png", "jpg", "jpeg", "npy", "nii", "tiff"],
            help="PNG/JPG slice, NumPy (.npy), or NIfTI (.nii/.nii.gz)",
        )
        if uploaded_file is not None:
            img_result = process_uploaded_image(uploaded_file)
            if img_result.get("display_image"):
                st.image(img_result["display_image"], caption=img_result["filename"],
                         use_container_width=True)
            if img_result.get("metadata"):
                st.json(img_result["metadata"])
            # Store as pending image for next chat message
            st.session_state.pending_image = img_result
            st.caption("✅ Image ready — it will be attached to your next message.")
        else:
            st.session_state.pending_image = None

        st.divider()
        st.caption(f"Neo4j: `{NEO4J_URI}`  |  LLM: `{OLLAMA_MODEL}`")

        if st.button("🗑️ Clear chat history"):
            st.session_state.messages = []
            st.session_state.last_rag_result = None
            st.session_state.pending_image = None
            st.rerun()

    # ─── Chat History ───
    for i, msg in enumerate(st.session_state.messages):
        role = msg["role"]
        with st.chat_message(role, avatar="🧑‍⚕️" if role == "user" else "🧠"):
            # Show image if attached
            if msg.get("image") and msg["image"].get("display_image"):
                st.image(msg["image"]["display_image"],
                         caption=msg["image"].get("filename", "MRI"), width=300)
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
        # Grab pending image
        attached_image = st.session_state.pending_image
        image_meta = attached_image.get("metadata") if attached_image else None

        # Add user message
        user_msg = {"role": "user", "text": user_input}
        if attached_image:
            user_msg["image"] = attached_image

        st.session_state.messages.append(user_msg)

        # Display immediately
        with st.chat_message("user", avatar="🧑‍⚕️"):
            if attached_image and attached_image.get("display_image"):
                st.image(attached_image["display_image"], width=300)
            st.markdown(user_input)

        # ╔══════════════════════════════════════════════════╗
        # ║  TOPIC GUARDRAIL — blocks non-medical queries    ║
        # ╚══════════════════════════════════════════════════╝
        from src.guardrails.clinical_guardrails import topic_guardrail

        topic_check = topic_guardrail(user_input)
        if topic_check.action != "allow":
            with st.chat_message("assistant", avatar="🧠"):
                st.markdown(topic_check.safe_reply or
                            "I can only assist with brain MRI analysis and BraTS-related queries.")
            st.session_state.messages.append({
                "role": "assistant",
                "text": topic_check.safe_reply or "Off-topic query blocked by guardrail.",
            })
            st.session_state.pending_image = None
            st.stop()

        # ── Decide: RAG or general chat ──
        lesion_id = _detect_lesion_id(user_input) or st.session_state.selected_lesion
        should_rag = _is_rag_query(user_input) and lesion_id is not None

        # Also trigger RAG if message was sent from sidebar button
        if st.session_state.messages and st.session_state.messages[-1].get("trigger_rag"):
            should_rag = True
            lesion_id = st.session_state.messages[-1].get("lesion_id", lesion_id)

        with st.chat_message("assistant", avatar="🧠"):
            if should_rag and lesion_id:
                with st.spinner(f"Running clinical RAG for **{lesion_id}**..."):
                    result = run_guarded_rag(
                        lesion_id=lesion_id, top_k=top_k, min_score=min_score,
                        user_query=user_input, image_meta=image_meta,
                    )
                st.session_state.messages.append({
                    "role": "assistant", "text": "", "rag_result": result,
                })
                st.session_state.last_rag_result = result
                render_rag_response(result)
            else:
                with st.spinner("Thinking..."):
                    response = handle_general_chat(user_input, image_meta=image_meta)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "text": response})

        # Clear pending image after use
        st.session_state.pending_image = None

    render_rag_dashboard_ui(
        offline_summary_path="artifacts/rag_eval_results/rag_eval_summary.json"
    )

    # ─── Welcome screen ───
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

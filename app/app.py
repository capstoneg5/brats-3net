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
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "neo4j123")
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


def handle_general_chat(user_msg: str, image_meta: Optional[Dict] = None) -> str:
    """Handle general chat messages that don't trigger RAG."""
    system_context = (
        "You are MedRAG-X, a clinical reasoning assistant for brain tumor MRI analysis. "
        "You help clinicians understand lesion comparisons, tumor metrics (ET%, TC%, WT), "
        "and MRI findings. Be concise, clinically precise, and always note that your "
        "outputs are for decision support — not autonomous diagnosis.\n"
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
    st.plotly_chart(fig, use_container_width=True)


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
    user_input = st.chat_input("Ask about a lesion, upload an image, or type anything...")

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

    # ─── Welcome screen ───
    if not st.session_state.messages:
        st.markdown("""
        <div style="text-align:center; padding:40px; color:#888;">
            <h3>Welcome to MedRAG-X 🧠</h3>
            <p style="margin-top:15px;">Start by:</p>
            <p>💬 <b>Typing a question</b> — e.g. <code>Compare lesion3 to similar lesions</code></p>
            <p>🖼️ <b>Uploading an MRI image</b> in the sidebar — PNG, NumPy, or NIfTI</p>
            <p>🔬 <b>Selecting a lesion</b> from the sidebar dropdown</p>
            <hr style="width:50%; margin:20px auto;">
            <p style="font-size:13px;">
            Mention a lesion ID + keywords like <i>compare, analyze, tumor, similar</i>
            to trigger full RAG analysis with guardrails.<br>
            Otherwise I'll chat normally about MRI and clinical topics.
            </p>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()


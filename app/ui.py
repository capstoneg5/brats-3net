# app/ui.py
from __future__ import annotations

import json
import requests
import streamlit as st

from config import paths


DEFAULT_API = "http://localhost:8000"


def call_health(api_base: str, backend: str, metric: str, collection: str, persist_dir: str):
    r = requests.get(
        f"{api_base}/health",
        params={"backend": backend, "metric": metric, "collection": collection, "persist_dir": persist_dir},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


def call_search(api_base: str, payload: dict):
    r = requests.post(f"{api_base}/search", json=payload, timeout=120)
    r.raise_for_status()
    return r.json()


st.set_page_config(page_title="MedRAG-X Search", layout="wide")

st.title("🧠 MedRAG-X — Semantic Search (Lesion 3D Embeddings)")
st.caption("Text query → PubMedBERT → FAISS/Chroma search over lesion-centric 3D embeddings")

with st.sidebar:
    st.header("Connection")
    api_base = st.text_input("API Base URL", value=DEFAULT_API)

    st.header("Vector Store")
    backend = st.selectbox("Backend", ["faiss", "chroma"], index=0)
    metric = st.selectbox("Metric", ["cosine", "l2"], index=0)
    collection = st.text_input("Collection (Chroma)", value="medragx")
    persist_dir = st.text_input("Persist Dir", value=str(paths.VECTOR_DB_DIR))

    st.header("Search Settings")
    top_k = st.slider("Top K", min_value=1, max_value=50, value=10)

    st.header("Metadata Filter (optional)")
    filter_enabled = st.checkbox("Enable filter", value=True)
    filter_json = st.text_area('Filter JSON (e.g. {"type":"lesion_3d"})', value='{"type":"lesion_3d"}')

    st.divider()
    if st.button("Check Health"):
        try:
            health = call_health(api_base, backend, metric, collection, persist_dir)
            st.success(f"OK | size={health['size']} | dim={health['dim']} | backend={health['backend']}")
        except Exception as e:
            st.error(f"Health failed: {e}")


query = st.text_input("Enter your clinical query", value="enhancing tumor with edema in frontal lobe")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Search")
    if st.button("🔎 Search", type="primary"):
        try:
            filt = None
            if filter_enabled and filter_json.strip():
                filt = json.loads(filter_json)

            payload = {
                "query": query,
                "top_k": top_k,
                "filter": filt,
                "backend": backend,
                "metric": metric,
                "collection": collection,
                "persist_dir": persist_dir,
            }

            data = call_search(api_base, payload)
            st.session_state["last_results"] = data
            st.success(f"Found {len(data['results'])} results")

        except json.JSONDecodeError as e:
            st.error(f"Invalid filter JSON: {e}")
        except requests.HTTPError as e:
            st.error(f"API error: {e.response.text if e.response is not None else e}")
        except Exception as e:
            st.error(f"Search failed: {e}")

with col2:
    st.subheader("Status")
    try:
        health = call_health(api_base, backend, metric, collection, persist_dir)
        st.info(f"Store size: {health['size']} | dim: {health['dim']} | backend: {health['backend']}")
    except Exception:
        st.warning("API not reachable yet. Start FastAPI server first.")


st.divider()
st.subheader("Results")

data = st.session_state.get("last_results")
if not data:
    st.write("Run a search to see results.")
else:
    for i, hit in enumerate(data["results"], start=1):
        meta = hit.get("metadata", {}) or {}
        patient_id = meta.get("patient_id", "unknown")
        hit_type = meta.get("type", "unknown")
        score = hit.get("score", 0.0)

        with st.expander(f"{i:02d}. {hit_type} | patient={patient_id} | score={score:.4f}", expanded=(i <= 3)):
            st.write("**ID**:", hit["id"])
            st.write("**Score**:", score)

            # Show common lesion metadata if present
            keys = ["tumor_volume_voxels", "bbox", "centroid_zyx", "cube_size", "modality"]
            compact = {k: meta.get(k) for k in keys if k in meta}
            if compact:
                st.json(compact)
            else:
                st.json(meta)

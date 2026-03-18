from pathlib import Path
import numpy as np
import faiss

from src.graph.knowledge_graph import ClinicalKnowledgeGraph

KG_PATH = Path("artifacts/kg_3d.json")
OUT_INDEX = Path("artifacts/lesion_faiss.index")
OUT_META = Path("artifacts/lesion_ids.npy")

def main():
    kg = ClinicalKnowledgeGraph.load(KG_PATH)

    lesion_ids = []
    embs = []

    for node_id, attrs in kg.graph.nodes(data=True):
        if attrs.get("type") == "Lesion" and attrs.get("embedding") is not None:
            lesion_ids.append(node_id)
            embs.append(np.asarray(attrs["embedding"], dtype=np.float32))

    X = np.vstack(embs).astype(np.float32)

    # embeddings already L2-normalized; if not, normalize here.
    # faiss.normalize_L2(X)

    d = X.shape[1]
    index = faiss.IndexFlatIP(d)   # cosine similarity when vectors are normalized
    index.add(X)

    OUT_INDEX.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(OUT_INDEX))
    np.save(str(OUT_META), np.array(lesion_ids))

    print("✅ FAISS index saved:", OUT_INDEX)
    print("✅ Lesion IDs saved:", OUT_META)
    print("Total lesions indexed:", len(lesion_ids))

if __name__ == "__main__":
    main()

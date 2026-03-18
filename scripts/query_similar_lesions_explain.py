from pathlib import Path
import numpy as np
import faiss

from src.graph.knowledge_graph import ClinicalKnowledgeGraph


# ---------------------------
# Explanation function
# ---------------------------
def explain(a: dict, b: dict):
    def pct_diff(x, y):
        return round(abs(x - y) / (abs(x) + 1e-8) * 100, 2)

    return {
        "tumor_volume_diff": abs(a["tumor_volume_voxels"] - b["tumor_volume_voxels"]),
        "tumor_volume_pct_diff": pct_diff(a["tumor_volume_voxels"], b["tumor_volume_voxels"]),
        "wt_pct_diff": pct_diff(a["wt_vox"], b["wt_vox"]),
        "tc_pct_diff": pct_diff(a["tc_vox"], b["tc_vox"]),
        "et_pct_diff": round(abs(a["et_pct"] - b["et_pct"]), 4),
        "tc_over_wt_a": round(a["tc_vox"] / (a["wt_vox"] + 1e-8), 4),
        "tc_over_wt_b": round(b["tc_vox"] / (b["wt_vox"] + 1e-8), 4),
    }

# ---------------------------
# Main query logic
# ---------------------------
def main():
    kg_path = Path("artifacts/kg_3d.json")
    index_path = Path("artifacts/lesion_faiss.index")
    ids_path = Path("artifacts/lesion_ids.npy")

    # load KG
    kg = ClinicalKnowledgeGraph.load(kg_path)

    # load FAISS
    index = faiss.read_index(str(index_path))
    lesion_ids = np.load(ids_path, allow_pickle=True).tolist()

    # ---------------------------
    # pick query lesion
    # ---------------------------
    query_id = "lesion3"

    if query_id not in kg.graph:
        raise ValueError(f"{query_id} not found in KG")

    query_props = kg.graph.nodes[query_id]
    query_embedding = np.asarray(
        query_props["embedding"],
        dtype=np.float32
    )[None, :]

    # ---------------------------
    # search
    # ---------------------------
    top_k = 10
    scores, idxs = index.search(query_embedding, top_k)

    print(f"\n🔎 Query lesion: {query_id}")
    print(f"Patient: {query_props['patient_id']}")
    print("=" * 80)

    # ---------------------------
    # display results
    # ---------------------------
    for rank, (score, idx) in enumerate(zip(scores[0], idxs[0]), start=1):

        lesion_id = lesion_ids[idx]
        props = kg.graph.nodes[lesion_id]

        explanation = explain(query_props, props)

        print(f"\n#{rank}")
        print(f"lesion_id : {lesion_id}")
        print(f"patient   : {props['patient_id']}")
        print(f"similarity: {score:.4f}")

        print("explanation:")
        for k, v in explanation.items():
            print(f"  {k:20s}: {v}")


if __name__ == "__main__":
    main()

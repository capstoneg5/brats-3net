from pathlib import Path
import numpy as np
import faiss

from src.graph.knowledge_graph import ClinicalKnowledgeGraph

KG_PATH = Path("artifacts/kg_3d.json")
INDEX_PATH = Path("artifacts/lesion_faiss.index")
IDS_PATH = Path("artifacts/lesion_ids.npy")

def main():
    kg = ClinicalKnowledgeGraph.load(KG_PATH)
    index = faiss.read_index(str(INDEX_PATH))
    lesion_ids = np.load(str(IDS_PATH), allow_pickle=True).tolist()

    query_id = "lesion3"
    q = np.asarray(kg.graph.nodes[query_id]["embedding"], dtype=np.float32)[None, :]

    k = 30
    scores, idxs = index.search(q, k)

    print(f"\nQuery lesion: {query_id}")
    print("\nTop matches with filtering (enhancing_vox > 10000):")

    shown = 0
    for score, idx in zip(scores[0], idxs[0]):
        lid = lesion_ids[idx]
        props = kg.graph.nodes[lid]

        # filter example:
        if props.get("enhancing_vox", 0) <= 10000:
            continue

        print(
            f"score={float(score):.4f} | {lid} | patient={props.get('patient_id')} | "
            f"wt={props.get('wt_vox')} tc={props.get('tc_vox')} et={props.get('enhancing_vox')}"
        )
        shown += 1
        if shown == 5:
            break

if __name__ == "__main__":
    main()

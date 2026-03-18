# scripts/check_embedding_separation.py
from pathlib import Path
import numpy as np
from src.graph.knowledge_graph import ClinicalKnowledgeGraph

def main():
    kg = ClinicalKnowledgeGraph.load(Path("artifacts/kg_3d.json"))

    embs = []
    for _, attrs in kg.graph.nodes(data=True):
        if attrs.get("type") == "Lesion" and attrs.get("embedding") is not None:
            embs.append(np.asarray(attrs["embedding"], dtype=np.float32))

    E = np.stack(embs, 0)
    E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-8)
    S = E @ E.T
    np.fill_diagonal(S, np.nan)

    vals = S[~np.isnan(S)]
    print("pairs:", vals.size)
    print("min :", float(np.min(vals)))
    print("mean:", float(np.mean(vals)))
    print("p50 :", float(np.percentile(vals, 50)))
    print("p90 :", float(np.percentile(vals, 90)))
    print("p99 :", float(np.percentile(vals, 99)))
    print("max :", float(np.max(vals)))

if __name__ == "__main__":
    main()

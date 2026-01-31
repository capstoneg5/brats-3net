import json
import numpy as np
from pathlib import Path

KG_PATH = Path("artifacts/kg_3d.json")

def explain(a: dict, b: dict):
    """
    Returns human-interpretable reasons why two lesions are similar.
    """
    return {
        "tumor_volume_diff": abs(a["tumor_volume_voxels"] - b["tumor_volume_voxels"]),
        "wt_diff": abs(a["wt_vox"] - b["wt_vox"]),
        "tc_diff": abs(a["tc_vox"] - b["tc_vox"]),
        "et_diff": abs(a["enhancing_vox"] - b["enhancing_vox"]),
        "et_pct_diff": round(abs(a["et_pct"] - b["et_pct"]), 4),
    }


def cosine(a, b, eps=1e-12):
    a = a / (np.linalg.norm(a) + eps)
    b = b / (np.linalg.norm(b) + eps)
    return float(np.dot(a, b))

def main():
    kg = json.loads(KG_PATH.read_text())
    lesions = [n for n in kg["nodes"] if n.get("type") == "Lesion" and n.get("embedding") is not None]

    # pick a query lesion (first one)
    q = lesions[0]
    qemb = np.array(q["embedding"], dtype=np.float32)

    scored = []
    for n in lesions[1:]:
        emb = np.array(n["embedding"], dtype=np.float32)
        scored.append((cosine(qemb, emb), n))

    scored.sort(key=lambda x: x[0], reverse=True)

    print("Query lesion:", q.get("id"), q.get("patient_id"))
    print("\nTop-5 similar lesions:")
    for s, n in scored[:5]:
        print(f"  score={s:.4f} | lesion_id={n.get('id')} | patient_id={n.get('patient_id')} | voxels={n.get('tumor_volume_voxels')}")

if __name__ == "__main__":
    main()

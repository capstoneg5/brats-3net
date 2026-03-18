#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def _infer_patient_id(lesion_id: str, attrs: Dict[str, Any]) -> str:
    pid = attrs.get("patient_id")
    if pid:
        return str(pid)
    if "_lesion" in lesion_id:
        return lesion_id.rsplit("_lesion", 1)[0]
    return "unknown_patient"


def _load_nodes(kg_path: Path) -> List[Dict[str, Any]]:
    data = json.loads(kg_path.read_text(encoding="utf-8"))
    nodes = data.get("nodes")
    if not isinstance(nodes, list):
        raise ValueError(f"Invalid KG format in {kg_path}: missing 'nodes' list")
    return nodes


def main() -> None:
    ap = argparse.ArgumentParser(description="Build embeddings JSONL from KG lesion nodes.")
    ap.add_argument("--kg", type=Path, default=Path("artifacts/kg_3d.json"))
    ap.add_argument("--out", type=Path, default=Path("artifacts/embeddings/lesion_embeddings.jsonl"))
    args = ap.parse_args()

    if not args.kg.exists():
        raise FileNotFoundError(f"KG file not found: {args.kg}")

    nodes = _load_nodes(args.kg)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    kept = 0
    with args.out.open("w", encoding="utf-8") as f:
        for node in nodes:
            if node.get("type") != "Lesion":
                continue
            emb = node.get("embedding")
            if not isinstance(emb, list) or not emb:
                continue

            lesion_id = str(node.get("id", ""))
            if not lesion_id:
                continue

            patient_id = _infer_patient_id(lesion_id, node)
            rec = {
                "lesion_id": lesion_id,
                "patient_id": patient_id,
                "embedding": emb,
                "dim": len(emb),
                "metadata": {
                    "source": "kg_3d",
                    "lesion_type": node.get("lesion_type"),
                },
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            kept += 1

    print(f"✅ Wrote {kept} embeddings to {args.out}")


if __name__ == "__main__":
    main()

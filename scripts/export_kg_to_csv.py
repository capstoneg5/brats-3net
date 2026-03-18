import json
import csv
from pathlib import Path

KG_PATH = Path("artifacts/kg_3d.json")
OUT_DIR = Path("artifacts/neo4j")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _get_edges(payload: dict) -> list[dict]:
    # NetworkX node-link format may use "links" or "edges"
    if "links" in payload:
        return payload["links"]
    if "edges" in payload:
        return payload["edges"]
    raise KeyError(f"KG JSON has no 'links' or 'edges'. Keys: {list(payload.keys())}")


def main():
    kg = json.loads(KG_PATH.read_text(encoding="utf-8"))

    nodes = kg.get("nodes", [])
    edges = _get_edges(kg)

    if not nodes:
        raise RuntimeError("No 'nodes' found in KG JSON.")
    if not edges:
        raise RuntimeError("No edges found in KG JSON.")

    # ------------------------
    # Nodes CSV
    # ------------------------
    node_fields = {"id"}  # node_link_data uses "id"
    for n in nodes:
        node_fields.update(n.keys())
    node_fields = sorted(node_fields)

    with open(OUT_DIR / "nodes.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=node_fields)
        writer.writeheader()
        for n in nodes:
            row = {k: n.get(k, "") for k in node_fields}
            writer.writerow(row)

    # ------------------------
    # Edges CSV
    # ------------------------
    # networkx uses: source, target (and possibly key)
    edge_fields = {"source", "target", "relationship", "score"}
    for e in edges:
        edge_fields.update(e.keys())
    edge_fields = sorted(edge_fields)

    with open(OUT_DIR / "edges.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=edge_fields)
        writer.writeheader()
        for e in edges:
            row = {k: e.get(k, "") for k in edge_fields}
            if not row.get("relationship"):
                row["relationship"] = "RELATED_TO"
            writer.writerow(row)

    print("✅ Exported:")
    print(" -", (OUT_DIR / "nodes.csv").resolve())
    print(" -", (OUT_DIR / "edges.csv").resolve())


if __name__ == "__main__":
    main()

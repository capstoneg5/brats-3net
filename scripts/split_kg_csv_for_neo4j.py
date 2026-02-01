from pathlib import Path
import csv
from collections import defaultdict

BASE = Path("artifacts/neo4j")
OUT = Path("artifacts/neo4j_split")

NODES_OUT = OUT / "nodes"
EDGES_OUT = OUT / "edges"

NODES_OUT.mkdir(parents=True, exist_ok=True)
EDGES_OUT.mkdir(parents=True, exist_ok=True)


def split_nodes():
    path = BASE / "nodes.csv"

    with open(path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    by_type = defaultdict(list)
    for r in rows:
        by_type[r["type"]].append(r)

    for label, items in by_type.items():
        out = NODES_OUT / f"{label}.csv"

        with open(out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=items[0].keys())
            writer.writeheader()
            writer.writerows(items)

        print(f"✅ nodes/{label}.csv  ({len(items)})")


def split_edges():
    path = BASE / "edges.csv"

    with open(path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    by_rel = defaultdict(list)
    for r in rows:
        rel = r["relationship"].upper()
        by_rel[rel].append(r)

    for rel, items in by_rel.items():
        out = EDGES_OUT / f"{rel}.csv"

        with open(out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=items[0].keys())
            writer.writeheader()
            writer.writerows(items)

        print(f"✅ edges/{rel}.csv  ({len(items)})")


def main():
    print("🔹 Splitting Neo4j CSV files...\n")

    split_nodes()
    split_edges()

    print("\n✅ Done.")
    print("📁 Output:", OUT.resolve())


if __name__ == "__main__":
    main()

from pathlib import Path
from src.graph.knowledge_graph import ClinicalKnowledgeGraph


def main():
    kg_path = Path("artifacts/kg_3d.json")
    kg = ClinicalKnowledgeGraph.load(kg_path)

    print("Nodes:", kg.graph.number_of_nodes())
    print("Edges:", kg.graph.number_of_edges())

    lesion_nodes = [
        n for n, d in kg.graph.nodes(data=True)
        if d.get("type") == "Lesion"
    ]

    print("Lesions:", len(lesion_nodes))

    # -------------------------------
    # Inspect one lesion
    # -------------------------------
    if not lesion_nodes:
        return

    lesion_id = lesion_nodes[0]

    print("\nSample lesion node id:", lesion_id)
    print("Lesion properties:")
    for k, v in kg.graph.nodes[lesion_id].items():
        print(f"  {k}: {v}")

    # -------------------------------
    # Inspect lesion regions
    # -------------------------------
    neighbors = kg.get_neighbors(
        lesion_id,
        relationship_type="has_region",
        max_hops=1,
    )

    print("\nRegion neighbors:", neighbors)

    for rid in neighbors:
        print(f"\nRegion node: {rid}")
        for k, v in kg.graph.nodes[rid].items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()

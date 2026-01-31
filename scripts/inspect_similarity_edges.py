# scripts/inspect_similarity_edges.py
from pathlib import Path
from src.graph.knowledge_graph import ClinicalKnowledgeGraph

def main():
    kg = ClinicalKnowledgeGraph.load(Path("artifacts/kg_3d_with_sim.json"))

    rel = "lesion_similar_to"
    count = 0
    for u, v, data in kg.graph.edges(data=True):
        if data.get("relationship") == rel:
            count += 1

    print("Nodes:", kg.graph.number_of_nodes())
    print("Edges:", kg.graph.number_of_edges())
    print(f"Similarity edges ({rel}):", count)

    # show sample
    for u, v, data in kg.graph.edges(data=True):
        if data.get("relationship") == rel:
            print("sample:", u, "->", v, "| score=", data.get("score"))
            break

if __name__ == "__main__":
    main()

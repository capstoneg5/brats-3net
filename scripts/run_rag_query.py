from src.rag.neo4j_retriever import Neo4jRetriever
import os


def main():
    password = os.getenv("NEO4J_PASSWORD")
    if not password:
        raise RuntimeError("Missing NEO4J_PASSWORD environment variable.")

    retriever = Neo4jRetriever(
        uri="neo4j://localhost:7687",
        user="neo4j",
        password=password
    )

    rows = retriever.retrieve_similar(
        lesion_id="lesion3",
        k=5
    )

    context = retriever.build_context(rows)

    print("\n===== RETRIEVED CONTEXT =====\n")
    print(context)


if __name__ == "__main__":
    main()

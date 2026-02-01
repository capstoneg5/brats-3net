from src.rag.neo4j_retriever import Neo4jRetriever


def main():
    retriever = Neo4jRetriever(
        uri="neo4j://localhost:7687",
        user="neo4j",
        password="neo4j123"
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

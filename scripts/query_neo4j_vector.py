from neo4j import GraphDatabase

URI = "neo4j://localhost:7687"
USER = "neo4j"
PASSWORD = "neo4j123"   # <-- your DB password

INDEX_NAME = "lesion_embedding_idx"
QUERY_LESION_ID = "lesion3"
TOP_K = 10

CYPHER = """
MATCH (q:Lesion {id: $lesionId})
WITH q.embedding AS emb
CALL db.index.vector.queryNodes($indexName, $k, emb)
YIELD node, score
RETURN node.id AS lesion_id,
       node.patient_id AS patient_id,
       score
ORDER BY score DESC
LIMIT $k
"""

def main():
    driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))

    try:
        with driver.session() as session:
            # quick sanity check
            ping = session.run("RETURN 1 AS ok").single()["ok"]
            print("✅ Connected to Neo4j:", ping)

            # verify lesion exists
            row = session.run(
                "MATCH (l:Lesion {id:$id}) RETURN l.id AS id, size(l.embedding) AS dims LIMIT 1",
                id=QUERY_LESION_ID
            ).single()

            if row is None:
                print(f"❌ Lesion '{QUERY_LESION_ID}' not found.")
                print("Try this to see a sample lesion id:")
                print("MATCH (l:Lesion) RETURN l.id LIMIT 5;")
                return

            print(f"🔎 Query lesion: {row['id']} | embedding dims: {row['dims']}")

            # run vector search
            res = session.run(
                CYPHER,
                lesionId=QUERY_LESION_ID,
                indexName=INDEX_NAME,
                k=TOP_K
            )

            print(f"\nTop-{TOP_K} similar lesions:")
            for r in res:
                print(f"  score={r['score']:.4f} | {r['lesion_id']} | patient={r['patient_id']}")

    finally:
        driver.close()

if __name__ == "__main__":
    main()

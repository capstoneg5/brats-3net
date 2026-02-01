import json
from neo4j import GraphDatabase

URI = "neo4j://localhost:7687"
USER = "neo4j"
PASSWORD = "neo4j123"

KG_PATH = "artifacts/kg_3d.json"
BATCH = 50

def main():
    kg = json.load(open(KG_PATH, "r"))
    nodes = kg.get("nodes", [])
    lesion_rows = []
    for n in nodes:
        if n.get("type") == "Lesion":
            emb = n.get("embedding")
            if emb is not None:
                lesion_rows.append({"id": n["id"], "embedding": emb})

    print("Lesions with embeddings in JSON:", len(lesion_rows))
    if not lesion_rows:
        raise RuntimeError("No lesion embeddings found in kg_3d.json")

    driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))

    cypher = """
    UNWIND $rows AS row
    MATCH (l:Lesion {id: row.id})
    SET l.embedding = row.embedding
    """

    try:
        with driver.session() as s:
            for i in range(0, len(lesion_rows), BATCH):
                chunk = lesion_rows[i:i+BATCH]
                s.run(cypher, rows=chunk)
                print(f"Updated {i+len(chunk)}/{len(lesion_rows)}")

            dims = s.run(
                "MATCH (l:Lesion {id:$id}) RETURN size(l.embedding) AS dims",
                id="lesion3"
            ).single()["dims"]
            print("lesion3 dims in Neo4j:", dims)
    finally:
        driver.close()

if __name__ == "__main__":
    main()

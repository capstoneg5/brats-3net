from __future__ import annotations
from neo4j import GraphDatabase


def attach_demo_kg(uri: str, user: str, password: str) -> None:
    """
    Minimal KG:
      (Lesion)-[:ASSOCIATED_WITH]->(TumorType)
      (TumorType)-[:HAS_GRADE]->(Grade)
      (TumorType)-[:HAS_OUTCOME]->(Outcome)

    Replace demo mapping later with real labels.
    """
    driver = GraphDatabase.driver(uri, auth=(user, password))
    with driver.session() as s:
        s.run("""
        MERGE (t:TumorType {name:'Glioblastoma'})
        MERGE (g:Grade {name:'IV'})
        MERGE (o:Outcome {name:'Poor'})
        MERGE (t)-[:HAS_GRADE]->(g)
        MERGE (t)-[:HAS_OUTCOME]->(o)
        """)

        # attach all lesions to this demo tumor type
        s.run("""
        MATCH (l:Lesion)
        MATCH (t:TumorType {name:'Glioblastoma'})
        MERGE (l)-[:ASSOCIATED_WITH]->(t)
        """)
    driver.close()
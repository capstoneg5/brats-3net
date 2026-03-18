from __future__ import annotations
from neo4j import GraphDatabase


def ensure_kg_schema(uri: str, user: str, password: str) -> None:
    driver = GraphDatabase.driver(uri, auth=(user, password))
    with driver.session() as s:
        s.run("CREATE CONSTRAINT tumor_type_unique IF NOT EXISTS FOR (t:TumorType) REQUIRE t.name IS UNIQUE")
        s.run("CREATE CONSTRAINT grade_unique IF NOT EXISTS FOR (g:Grade) REQUIRE g.name IS UNIQUE")
        s.run("CREATE CONSTRAINT outcome_unique IF NOT EXISTS FOR (o:Outcome) REQUIRE o.name IS UNIQUE")
    driver.close()
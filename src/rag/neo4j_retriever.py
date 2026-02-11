# src/rag/neo4j_retriever.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from neo4j import GraphDatabase


CYPHER = """
MATCH (q:Lesion {id:$lesionId})
WITH q.embedding AS qEmb
CALL db.index.vector.queryNodes("lesion_embedding_idx", $k, qEmb)
YIELD node, score

// attach region info
OPTIONAL MATCH (node)-[:HAS_REGION]->(r:Region)
WITH node, score,
     collect({
       region: r.region,
       voxels: r.voxels,
       pct: r.pct
     }) AS regions

RETURN
  node.id                AS lesion_id,
  node.patient_id        AS patient_id,
  score                  AS score,
  node.tumor_volume_voxels AS wt_vox,
  node.tc_vox            AS tc_vox,
  node.et_vox            AS et_vox,
  node.et_pct            AS et_pct,
  node.tc_pct            AS tc_pct,
  // these may not exist in Neo4j depending on your import:
  node.centroid_zyx      AS centroid,
  node.bbox              AS bbox,
  regions                AS regions
ORDER BY score DESC
"""


class Neo4jRetriever:
    """
    Neo4j vector retriever for Lesion similarity search.

    IMPORTANT:
    - Use bolt://127.0.0.1:7687 to avoid routing errors on Desktop single instance.
    - centroid_zyx / bbox will be None unless you imported them into Neo4j.
    """

    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        if self.driver is not None:
            self.driver.close()

    def retrieve_similar(self, lesion_id: str, k: int = 5) -> List[Dict[str, Any]]:
        with self.driver.session() as session:
            rows = session.run(CYPHER, lesionId=lesion_id, k=int(k)).data()
        return rows

    @staticmethod
    def build_context(rows: List[Dict[str, Any]]) -> str:
        lines: List[str] = []

        for i, r in enumerate(rows, 1):
            regions_list = r.get("regions") or []
            regions_str = ", ".join(
                f"{x.get('region')}:vox={x.get('voxels')},pct={x.get('pct')}"
                for x in regions_list
                if x.get("region") is not None
            )

            # NOTE: centroid/bbox may be None if not imported
            centroid = r.get("centroid")
            bbox = r.get("bbox")

            et_pct = r.get("et_pct")
            et_pct_str = f"{float(et_pct):.3f}" if et_pct is not None else "NA"

            lines.append(
                f"[{i}] lesion={r.get('lesion_id')} patient={r.get('patient_id')} "
                f"score={float(r.get('score', 0.0)):.4f} "
                f"WT={r.get('wt_vox')} TC={r.get('tc_vox')} ET={r.get('et_vox')} "
                f"ET%={et_pct_str} "
                f"centroid={centroid} bbox={bbox} "
                f"regions=({regions_str})"
            )

        return "\n".join(lines)


def main():
    retriever = Neo4jRetriever(
        uri="bolt://127.0.0.1:7687",
        user="neo4j",
        password="neo4j123"
    )

    rows = retriever.retrieve_similar("lesion3", k=5)
    print(retriever.build_context(rows))

    retriever.close()

if __name__ == "__main__":
    main()
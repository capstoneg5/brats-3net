from neo4j import GraphDatabase


CYPHER = """
MATCH (q:Lesion {id:$lesionId})
WITH q.embedding AS qEmb
CALL db.index.vector.queryNodes("lesion_embedding_idx", $k, qEmb)
YIELD node, score
WITH node, score
OPTIONAL MATCH (node)-[:HAS_REGION]->(r:Region)
WITH node, score, collect({
    region: r.region,
    voxels: r.voxels,
    pct: r.pct
}) AS regions
RETURN
  node.id AS lesion_id,
  node.patient_id AS patient_id,
  score,
  node.tumor_volume_voxels AS wt_vox,
  node.tc_vox AS tc_vox,
  node.et_vox AS et_vox,
  node.et_pct AS et_pct,
  node.tc_pct AS tc_pct,
  node.centroid_zyx AS centroid,
  node.bbox AS bbox,
  regions
ORDER BY score DESC
"""


class Neo4jRetriever:

    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(
            uri,
            auth=(user, password)
        )

    def retrieve_similar(self, lesion_id: str, k: int = 5):
        with self.driver.session() as session:
            rows = session.run(
                CYPHER,
                lesionId=lesion_id,
                k=k
            ).data()
        return rows

    @staticmethod
    def build_context(rows):
        lines = []

        for i, r in enumerate(rows, 1):
            regions = ", ".join(
                f"{x['region']}:vox={x['voxels']},pct={x['pct']}"
                for x in r["regions"]
                if x.get("region") is not None
            )

            lines.append(
                f"[{i}] lesion={r['lesion_id']} "
                f"patient={r['patient_id']} "
                f"score={r['score']:.4f} "
                f"WT={r['wt_vox']} "
                f"TC={r['tc_vox']} "
                f"ET={r['et_vox']} "
                f"ET%={r['et_pct']:.3f} "
                f"centroid={r['centroid']} "
                f"bbox={r['bbox']} "
                f"regions=({regions})"
            )

        return "\n".join(lines)

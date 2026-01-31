import numpy as np

from models.embeddings.lesion_embedder_3d import LesionEmbeddingPipeline3D
from src.graph.knowledge_graph import ClinicalKnowledgeGraph


def make_fake_volume_and_mask(D=96, H=96, W=96):
    rng = np.random.default_rng(0)
    vol = rng.normal(0, 1, size=(4, D, H, W)).astype(np.float32)

    zz, yy, xx = np.ogrid[:D, :H, :W]
    cz, cy, cx = D // 2, H // 2, W // 2
    r = min(D, H, W) // 8
    mask = ((zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2 <= r ** 2).astype(np.uint8)

    vol[:, mask > 0] += 3.0
    return vol, mask


def main():
    # 1) Fake data (same as your test)
    volume_4ch, mask = make_fake_volume_and_mask()

    # 2) Get lesion embedding
    pipeline = LesionEmbeddingPipeline3D(cube_size=(64, 64, 64))
    out = pipeline.embed_from_volume_and_mask(
        volume_4ch=volume_4ch,
        mask=mask,
        patient_id="demo_patient",
        margin=5,
    )

    if out is None:
        print("No lesion found")
        return

    # 3) Build KG
    kg = ClinicalKnowledgeGraph()

    patient_node = kg.add_patient("demo_patient", metadata={"source": "demo"})
    mri_node = kg.add_mri_volume(
        patient_node=patient_node,
        modality="MULTI_4CH",
        embedding=None,  # optional: you can store volume embedding here later
        metadata={"shape": list(volume_4ch.shape)},
    )

    props = dict(out.meta)
    props["embedding_type"] = props.pop("type", None)  # avoid conflict with node attribute 'type'

    kg.add_lesion(
        mri_node=mri_node,
        lesion_type="tumor",
        embedding=out.embedding,
        properties=props,
    )

    # 4) Save
    kg.save("artifacts/kg.json")
    print("✅ Saved KG to artifacts/kg.json")


if __name__ == "__main__":
    main()

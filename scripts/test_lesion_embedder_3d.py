import numpy as np

from models.embeddings.lesion_embedder_3d import LesionEmbeddingPipeline3D


def make_fake_volume_and_mask(D=96, H=96, W=96):
    """
    Fake MRI volume: [4, D, H, W]
    Fake mask: [D, H, W] with a spherical "tumor"
    """
    rng = np.random.default_rng(0)

    # 4 modalities
    vol = rng.normal(0, 1, size=(4, D, H, W)).astype(np.float32)

    # create a spherical tumor in the mask
    zz, yy, xx = np.ogrid[:D, :H, :W]
    cz, cy, cx = D // 2, H // 2, W // 2
    r = min(D, H, W) // 8
    mask = ((zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2 <= r ** 2).astype(np.uint8)

    # make tumor region brighter in all modalities (so it’s not random noise)
    vol[:, mask > 0] += 3.0

    return vol, mask


def main():
    volume_4ch, mask = make_fake_volume_and_mask()

    pipeline = LesionEmbeddingPipeline3D(cube_size=(64, 64, 64))

    out = pipeline.embed_from_volume_and_mask(
        volume_4ch=volume_4ch,
        mask=mask,
        patient_id="demo_patient",
        margin=5,
    )

    if out is None:
        print("No lesion found in mask.")
        return

    print("✅ Embedding shape:", out.embedding.shape)  # (768,)
    print("✅ Embedding dtype:", out.embedding.dtype)
    print("✅ Meta:", out.meta)


if __name__ == "__main__":
    main()

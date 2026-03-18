# models/embeddings/image_embedder.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoImageProcessor, AutoModel

from config import model_config  # expects VISION_EMBEDDING_MODEL, EMBEDDING_DIM


@dataclass
class ImageEmbeddingOutput:
    embeddings: np.ndarray  # shape: [N, 768]
    model_name: str


class ImageEmbedder2D:
    """
    2D image embedder using a HuggingFace vision backbone (e.g., Swin).

    Input: list of images as numpy arrays:
      - (H, W) grayscale
      - (H, W, 1) or (H, W, 3)
    Output: [N, 768] float32 embeddings
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        normalize: bool = True,
        force_proj_to_768: bool = True,
    ) -> None:
        self.model_name = model_name or model_config.VISION_EMBEDDING_MODEL
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.normalize = normalize
        self.force_proj_to_768 = force_proj_to_768

        self.processor = AutoImageProcessor.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
        self.model.eval()

        hidden = (
            getattr(self.model.config, "hidden_size", None)
            or getattr(self.model.config, "hidden_sizes", [None])[-1]
        )
        if hidden is None:
            hidden = model_config.EMBEDDING_DIM

        self.hidden_size = int(hidden)
        self.out_dim = int(model_config.EMBEDDING_DIM)

        self.proj: Optional[nn.Module]
        if self.force_proj_to_768 and self.hidden_size != self.out_dim:
            self.proj = nn.Linear(self.hidden_size, self.out_dim).to(self.device)
        else:
            self.proj = None

    @staticmethod
    def _ensure_uint8_image(img: np.ndarray) -> np.ndarray:
        """
        Convert MRI slice (float32) to uint8 RGB image for vision models.
        Robust percentile scaling.
        """
        if img.ndim == 2:
            x = img.astype(np.float32)
            lo, hi = np.percentile(x, 1), np.percentile(x, 99)
            x = np.clip((x - lo) / (hi - lo + 1e-8), 0.0, 1.0)
            x = (x * 255.0).astype(np.uint8)
            x = np.stack([x, x, x], axis=-1)  # RGB
            return x

        if img.ndim == 3 and img.shape[-1] in (1, 3):
            x = img
            if x.dtype != np.uint8:
                x = x.astype(np.float32)
                lo, hi = np.percentile(x, 1), np.percentile(x, 99)
                x = np.clip((x - lo) / (hi - lo + 1e-8), 0.0, 1.0)
                x = (x * 255.0).astype(np.uint8)
            if x.shape[-1] == 1:
                x = np.repeat(x, 3, axis=-1)
            return x

        raise ValueError(f"Unsupported image shape: {img.shape}")

    @staticmethod
    def _l2_normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        return x / (torch.norm(x, p=2, dim=-1, keepdim=True) + eps)

    @torch.no_grad()
    def embed(self, images: Union[np.ndarray, List[np.ndarray]], batch_size: int = 8) -> ImageEmbeddingOutput:
        if isinstance(images, np.ndarray):
            img_list = [images]
        else:
            img_list = list(images)

        all_vecs: List[np.ndarray] = []

        for i in range(0, len(img_list), batch_size):
            batch_imgs = [self._ensure_uint8_image(x) for x in img_list[i : i + batch_size]]

            inputs = self.processor(images=batch_imgs, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            out = self.model(**inputs)

            # Prefer last_hidden_state: [B, tokens, hidden]
            if hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
                vec = out.last_hidden_state.mean(dim=1)
            elif hasattr(out, "pooler_output") and out.pooler_output is not None:
                vec = out.pooler_output
            else:
                raise RuntimeError("Model output does not contain usable hidden states")

            if self.proj is not None:
                vec = self.proj(vec)

            if self.normalize:
                vec = self._l2_normalize(vec)

            all_vecs.append(vec.detach().cpu().numpy().astype(np.float32))

        embeddings = np.concatenate(all_vecs, axis=0) if all_vecs else np.zeros((0, self.out_dim), np.float32)
        return ImageEmbeddingOutput(embeddings=embeddings, model_name=self.model_name)


# ---------------------------------------------------------
# Demo / CLI runner
# ---------------------------------------------------------
def _make_fake_mri_slices(n: int = 8, h: int = 224, w: int = 224) -> List[np.ndarray]:
    """
    Create MRI-like grayscale slices for testing the embedder without dataset.
    """
    rng = np.random.default_rng(42)
    imgs: List[np.ndarray] = []
    for _ in range(n):
        base = rng.normal(0.0, 1.0, size=(h, w)).astype(np.float32)
        # add a bright "lesion blob"
        cy, cx = rng.integers(h // 4, 3 * h // 4), rng.integers(w // 4, 3 * w // 4)
        yy, xx = np.ogrid[:h, :w]
        mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= (min(h, w) // 10) ** 2
        base[mask] += 4.0
        # normalize-ish
        base = (base - base.min()) / (base.max() - base.min() + 1e-8)
        imgs.append(base)
    return imgs


def main() -> None:
    # Create output folder
    out_dir = Path("artifacts")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Create demo images
    demo_imgs = _make_fake_mri_slices(n=8, h=224, w=224)

    # Run embedder
    embedder = ImageEmbedder2D()
    out = embedder.embed(demo_imgs, batch_size=4)

    print(f"Model: {out.model_name}")
    print(f"Embeddings shape: {out.embeddings.shape}")  # expected (8, 768)
    print(f"Embeddings dtype:  {out.embeddings.dtype}")

    # Save embeddings
    save_path = out_dir / "demo_image_embeddings.npy"
    np.save(save_path, out.embeddings)
    print(f"✅ Saved embeddings to: {save_path}")


if __name__ == "__main__":
    main()

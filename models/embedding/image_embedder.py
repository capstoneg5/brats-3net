# models/embeddings/image_embedder.py
from __future__ import annotations

from dataclasses import dataclass
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
    Useful for slice-based embeddings or quick baselines.

    Input: list of images as numpy arrays:
      - ( H, W ) grayscale
      - (H, W, 3) RGB
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

        hidden = getattr(self.model.config, "hidden_size", None) or getattr(self.model.config, "hidden_sizes", [None])[-1]
        if hidden is None:
            # fallback if model doesn't expose hidden_size
            hidden = model_config.EMBEDDING_DIM

        self.hidden_size = int(hidden)
        self.out_dim = model_config.EMBEDDING_DIM

        self.proj: Optional[nn.Module]
        if self.force_proj_to_768 and self.hidden_size != self.out_dim:
            self.proj = nn.Linear(self.hidden_size, self.out_dim).to(self.device)
        else:
            self.proj = None

    @staticmethod
    def _ensure_uint8_image(img: np.ndarray) -> np.ndarray:
        """
        Convert MRI slice (float32) to uint8 image for vision models.
        """
        if img.ndim == 2:
            x = img
            # robust min-max
            lo, hi = np.percentile(x, 1), np.percentile(x, 99)
            x = np.clip((x - lo) / (hi - lo + 1e-8), 0, 1)
            x = (x * 255.0).astype(np.uint8)
            x = np.stack([x, x, x], axis=-1)  # to RGB
            return x

        if img.ndim == 3 and img.shape[-1] in (1, 3):
            x = img
            if x.dtype != np.uint8:
                lo, hi = np.percentile(x, 1), np.percentile(x, 99)
                x = np.clip((x - lo) / (hi - lo + 1e-8), 0, 1)
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

            # Most HF vision backbones expose last_hidden_state: [B, tokens, hidden]
            if hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
                cls_or_mean = out.last_hidden_state.mean(dim=1)  # mean pool over tokens
            elif hasattr(out, "pooler_output") and out.pooler_output is not None:
                cls_or_mean = out.pooler_output
            else:
                raise RuntimeError("Model output does not contain usable hidden states")

            if self.proj is not None:
                cls_or_mean = self.proj(cls_or_mean)

            if self.normalize:
                cls_or_mean = self._l2_normalize(cls_or_mean)

            all_vecs.append(cls_or_mean.detach().cpu().numpy().astype(np.float32))

        embeddings = np.concatenate(all_vecs, axis=0) if all_vecs else np.zeros((0, self.out_dim), np.float32)
        return ImageEmbeddingOutput(embeddings=embeddings, model_name=self.model_name)

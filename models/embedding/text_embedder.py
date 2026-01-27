# models/embeddings/text_embedder.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Union, Optional

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

from config import model_config  # expects TEXT_EMBEDDING_MODEL, EMBEDDING_DIM


@dataclass
class TextEmbeddingOutput:
    embeddings: np.ndarray  # shape: [N, D]
    model_name: str


class TextEmbedder:
    """
    Text embedding using a clinically tuned LM (e.g., PubMedBERT).
    Produces sentence embeddings using mean pooling over token embeddings.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        max_length: int = 256,
        normalize: bool = True,
    ) -> None:
        self.model_name = model_name or model_config.TEXT_EMBEDDING_MODEL
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.normalize = normalize

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
        self.model.eval()

        # Many BERT-style models output hidden_size=768 (matches your config).
        self.embedding_dim = getattr(self.model.config, "hidden_size", model_config.EMBEDDING_DIM)

    @staticmethod
    def _mean_pooling(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # last_hidden_state: [B, T, H], attention_mask: [B, T]
        mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)  # [B, T, 1]
        summed = torch.sum(last_hidden_state * mask, dim=1)            # [B, H]
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)                # [B, 1]
        return summed / counts

    @staticmethod
    def _l2_normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        return x / (torch.norm(x, p=2, dim=-1, keepdim=True) + eps)

    @torch.no_grad()
    def embed(self, texts: Union[str, List[str]], batch_size: int = 16) -> TextEmbeddingOutput:
        if isinstance(texts, str):
            texts_list = [texts]
        else:
            texts_list = list(texts)

        all_vecs: List[np.ndarray] = []

        for i in range(0, len(texts_list), batch_size):
            batch = texts_list[i : i + batch_size]
            enc = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}

            out = self.model(**enc)
            pooled = self._mean_pooling(out.last_hidden_state, enc["attention_mask"])

            if self.normalize:
                pooled = self._l2_normalize(pooled)

            all_vecs.append(pooled.detach().cpu().numpy().astype(np.float32))

        embeddings = np.concatenate(all_vecs, axis=0) if all_vecs else np.zeros((0, self.embedding_dim), np.float32)
        return TextEmbeddingOutput(embeddings=embeddings, model_name=self.model_name)

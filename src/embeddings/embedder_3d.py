# src/embeddings/embedder_3d.py
from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

from .train_embedder_3d import Encoder3D  # reuse same architecture


class TrainedCubeEmbedder:
    def __init__(self, ckpt_path: str | Path, device: str | None = None):
        self.device = device or ("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
        ckpt = torch.load(str(ckpt_path), map_location="cpu")
        embed_dim = int(ckpt.get("embed_dim", 768))

        self.model = Encoder3D(in_ch=4, embed_dim=embed_dim).to(self.device)
        self.model.load_state_dict(ckpt["encoder"], strict=True)
        self.model.eval()
        self.embed_dim = embed_dim

    @torch.no_grad()
    def embed_cube(self, cube: np.ndarray) -> np.ndarray:
        # cube: [C, D, H, W]
        x = torch.from_numpy(cube).float().unsqueeze(0).to(self.device)  # [1,C,D,H,W]
        z = self.model(x).squeeze(0)  # [D]
        z = F.normalize(z, dim=0)
        return z.detach().cpu().numpy().astype(np.float32)
# models/embeddings/lesion_embedder_3d.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import model_config  # expects EMBEDDING_DIM


@dataclass
class LesionEmbeddingOutput:
    embedding: np.ndarray        # shape: [D] (768,)
    meta: Dict


class _ConvBlock3D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LesionEmbedder3D(nn.Module):
    """
    True 3D lesion embedder:
      Input : [B, 4, D, H, W] (lesion cube)
      Output: [B, 768] embeddings (L2 normalized)
    """

    def __init__(self, in_channels: int = 4, out_dim: int = None) -> None:
        super().__init__()
        self.out_dim = out_dim or model_config.EMBEDDING_DIM

        self.stem = _ConvBlock3D(in_channels, 32)
        self.down1 = nn.Sequential(nn.MaxPool3d(2), _ConvBlock3D(32, 64))
        self.down2 = nn.Sequential(nn.MaxPool3d(2), _ConvBlock3D(64, 128))
        self.down3 = nn.Sequential(nn.MaxPool3d(2), _ConvBlock3D(128, 256))

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),  # [B, 256, 1,1,1]
            nn.Flatten(),             # [B, 256]
            nn.Linear(256, self.out_dim),
        )

    def forward(self, x: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        x = self.stem(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.head(x)
        if normalize:
            x = F.normalize(x, p=2, dim=-1)
        return x


def compute_bbox_3d(mask: np.ndarray, margin: int = 5) -> Optional[Tuple[int, int, int, int, int, int]]:
    """
    mask: [D,H,W] or [H,W,D] depending on your pipeline; here we assume [D,H,W].
    Returns: (zmin, zmax, ymin, ymax, xmin, xmax) inclusive bounds.
    """
    if mask.ndim != 3:
        raise ValueError(f"mask must be 3D, got shape {mask.shape}")

    coords = np.argwhere(mask > 0)
    if coords.size == 0:
        return None

    zmin, ymin, xmin = coords.min(axis=0)
    zmax, ymax, xmax = coords.max(axis=0)

    zmin = max(int(zmin) - margin, 0)
    ymin = max(int(ymin) - margin, 0)
    xmin = max(int(xmin) - margin, 0)

    zmax = int(zmax) + margin
    ymax = int(ymax) + margin
    xmax = int(xmax) + margin

    return zmin, zmax, ymin, ymax, xmin, xmax


def crop_cube_3d(volume: np.ndarray, bbox: Tuple[int, int, int, int, int, int]) -> np.ndarray:
    """
    volume: [C, D, H, W]
    bbox: (zmin, zmax, ymin, ymax, xmin, xmax) in D,H,W index space
    """
    zmin, zmax, ymin, ymax, xmin, xmax = bbox
    C, D, H, W = volume.shape

    zmax = min(zmax, D - 1)
    ymax = min(ymax, H - 1)
    xmax = min(xmax, W - 1)

    return volume[:, zmin : zmax + 1, ymin : ymax + 1, xmin : xmax + 1]


def resize_cube_3d(cube: torch.Tensor, out_size: Tuple[int, int, int]) -> torch.Tensor:
    """
    cube: [B, C, D, H, W]
    out_size: (D, H, W)
    """
    return F.interpolate(cube, size=out_size, mode="trilinear", align_corners=False)


class LesionEmbeddingPipeline3D:
    """
    Convenience wrapper:
      - takes preprocessed volume + mask
      - crops lesion cube
      - resizes to fixed cube
      - runs 3D embedder
    """

    def __init__(
        self,
        embedder: Optional[LesionEmbedder3D] = None,
        device: Optional[str] = None,
        cube_size: Tuple[int, int, int] = (64, 64, 64),
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.cube_size = cube_size
        self.embedder = embedder or LesionEmbedder3D(in_channels=4, out_dim=model_config.EMBEDDING_DIM)
        self.embedder.to(self.device)
        self.embedder.eval()

    @torch.no_grad()
    def embed_from_volume_and_mask(
        self,
        volume_4ch: np.ndarray,
        mask: np.ndarray,
        patient_id: str = "unknown",
        margin: int = 5,
    ) -> Optional[LesionEmbeddingOutput]:
        """
        volume_4ch: [4, D, H, W] (after preprocessing)
        mask: [D, H, W] predicted or GT tumor mask (non-zero = lesion)
        """
        if volume_4ch.ndim != 4 or volume_4ch.shape[0] != 4:
            raise ValueError(f"Expected volume shape [4,D,H,W], got {volume_4ch.shape}")

        bbox = compute_bbox_3d(mask, margin=margin)
        if bbox is None:
            return None  # no lesion found

        cube_np = crop_cube_3d(volume_4ch, bbox)  # [4, d, h, w]

        cube = torch.tensor(cube_np, dtype=torch.float32).unsqueeze(0).to(self.device)  # [1,4,d,h,w]
        cube = resize_cube_3d(cube, self.cube_size)  # [1,4,64,64,64]

        emb = self.embedder(cube, normalize=True)  # [1,768]
        emb_np = emb.squeeze(0).detach().cpu().numpy().astype(np.float32)

        # simple stats
        coords = np.argwhere(mask > 0)
        centroid = coords.mean(axis=0).tolist() if coords.size else [None, None, None]
        tumor_volume = int(coords.shape[0])

        meta = {
            "patient_id": patient_id,
            "type": "lesion_3d",
            "bbox": list(bbox),
            "centroid_zyx": centroid,
            "tumor_volume_voxels": tumor_volume,
            "cube_size": list(self.cube_size),
        }

        return LesionEmbeddingOutput(embedding=emb_np, meta=meta)

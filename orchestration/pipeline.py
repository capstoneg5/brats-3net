# orchestration/pipeline.py
"""
MedRAG-X Orchestration Pipeline (Production-standard)

What it does (end-to-end):
1) Discover patients (BraTS)
2) Load + preprocess MRI modalities (+ seg if available)
3) (Optional) Run segmentation inference using trained 3D UNet (or use GT seg if present)
4) Build lesion-centric 3D embeddings (768-d) from lesion cube
5) Write embeddings to JSONL (id, embeddings, metadata)
6) (Optional) Build/persist vector index (FAISS/Chroma)
7) (Optional) Run a smoke-search query

Usage examples:
  # Build embeddings JSONL + build FAISS index
  python -m orchestration.pipeline run \
    --split train \
    --max_patients 200 \
    --use_pred_mask \
    --checkpoint models/segmentation/unet3d_best.pt \
    --write_jsonl \
    --build_index \
    --backend faiss

  # Only export embeddings (no index build)
  python -m orchestration.pipeline run --split val --write_jsonl

  # Smoke-search
  python -m orchestration.pipeline search \
    --backend faiss \
    --query "enhancing tumor with edema in frontal lobe" \
    --filter '{"type":"lesion_3d"}'
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from loguru import logger

# Project imports (your modules)
from config import paths, model_config, training_config
from ingestion.data_ingestion import BraTSDataIngestion
from preprocessing.mri_preprocessing import MRIPreprocessor, MODALITY_KEYS
from models.segmentation.unet3d import UNet3DSegmenter
from models.embeddings.lesion_embedder_3d import LesionEmbeddingPipeline3D
from models.embeddings.text_embedder import TextEmbedder
from retrival.vector_store import create_vector_store


JsonDict = Dict[str, Any]


# ---------------------------
# Utilities
# ---------------------------
def _now_ms() -> int:
    return int(time.time() * 1000)


def _safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _read_jsonl_ids(jsonl_path: Path) -> set[str]:
    """
    Used for resume support: if JSONL already has entries, skip those IDs.
    """
    ids: set[str] = set()
    if not jsonl_path.exists():
        return ids
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                ids.add(obj["id"])
            except Exception:
                continue
    return ids


def _atomic_append_jsonl(path: Path, obj: JsonDict) -> None:
    """
    Append one JSON object per line. Minimal and safe enough for single-process runs.
    """
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + eps)


def _set_repro(seed: int = 42) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # ok if no cuda
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _remap_brats_labels_np(seg: np.ndarray) -> np.ndarray:
    """
    BraTS labels are {0,1,2,4}. Many pipelines map 4->3 for contiguous classes.
    """
    seg = seg.astype(np.int32, copy=False)
    seg = np.where(seg == 4, 3, seg)
    seg = np.clip(seg, 0, 3)
    return seg.astype(np.uint8)


def _select_split_root(split: str) -> Path:
    split = split.lower().strip()
    if split == "train":
        return paths.DATA_ROOT_TRAIN
    if split == "val":
        return paths.DATA_ROOT_VAL
    raise ValueError("split must be 'train' or 'val'")


# ---------------------------
# Config dataclass
# ---------------------------
@dataclass
class PipelineConfig:
    split: str = "train"
    max_patients: int = 0  # 0 = all
    seed: int = 42

    # Segmentation source
    use_pred_mask: bool = False
    checkpoint: Optional[Path] = None

    # Embedding output
    cube_size: Tuple[int, int, int] = (64, 64, 64)
    margin: int = 5

    # Output controls
    write_jsonl: bool = True
    resume: bool = True

    # Index controls
    build_index: bool = False
    backend: str = "faiss"  # faiss|chroma
    metric: str = "cosine"
    collection: str = "medragx"

    # Storage locations
    embeddings_dir: Path = paths.EMBEDDINGS_DIR
    vector_db_dir: Path = paths.VECTOR_DB_DIR

    # Filtering
    only_with_seg: bool = False  # skip patients without GT seg (for train)


# ---------------------------
# Core Pipeline
# ---------------------------
class MedRAGXPipeline:
    def __init__(self, cfg: PipelineConfig) -> None:
        self.cfg = cfg
        _set_repro(cfg.seed)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Device: {self.device}")

        self.data_root = _select_split_root(cfg.split)
        self.ingestor = BraTSDataIngestion(data_root=self.data_root)
        self.preprocessor = MRIPreprocessor(target_size=model_config.IMG_SIZE)

        self.lesion_embedder = LesionEmbeddingPipeline3D(
            device=self.device,
            cube_size=cfg.cube_size,
        )

        self.seg_model: Optional[UNet3DSegmenter] = None
        if cfg.use_pred_mask:
            if not cfg.checkpoint:
                raise ValueError("use_pred_mask=True requires --checkpoint path")
            self.seg_model = self._load_segmentation_model(cfg.checkpoint)

        # Outputs
        _safe_mkdir(cfg.embeddings_dir)
        _safe_mkdir(cfg.vector_db_dir)

        self.jsonl_path = cfg.embeddings_dir / f"lesion_embeddings_{cfg.split}.jsonl"

    def _load_segmentation_model(self, checkpoint: Path) -> UNet3DSegmenter:
        if not checkpoint.exists():
            raise FileNotFoundError(f"Segmentation checkpoint not found: {checkpoint}")

        model = UNet3DSegmenter(
            in_channels=model_config.SEGMENTATION_IN_CHANNELS,
            out_channels=model_config.SEGMENTATION_OUT_CHANNELS,
            device=self.device,
        )
        model.setup_training(lr=training_config.LEARNING_RATE)

        ckpt = torch.load(checkpoint, map_location=self.device)
        state = ckpt.get("model_state_dict", ckpt)
        model.model.load_state_dict(state)
        model.model.eval()

        logger.info(f"Loaded segmentation checkpoint: {checkpoint}")
        return model

    def _predict_mask(self, image_4ch: torch.Tensor) -> np.ndarray:
        """
        image_4ch: torch.Tensor [4, D, H, W]
        Returns numpy mask [D,H,W] in {0,1,2,3}
        """
        assert self.seg_model is not None
        img = image_4ch.unsqueeze(0).to(self.device)  # [1,4,D,H,W]

        with torch.no_grad():
            logits = self.seg_model.model(img)  # [1,4,D,H,W]
            pred = torch.argmax(logits, dim=1)  # [1,D,H,W]
            pred_np = pred.squeeze(0).detach().cpu().numpy().astype(np.uint8)

        # already contiguous 0..3 (because model outputs 4 channels)
        return pred_np

    @staticmethod
    def _build_image_tensor(processed: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        processed contains t1,t2,flair,t1ce each [1,D,H,W] after EnsureChannelFirstd.
        Returns: [4,D,H,W]
        """
        return torch.cat([processed[k] for k in MODALITY_KEYS], dim=0).float()

    def run(self) -> Path:
        logger.info(f"Starting pipeline | split={self.cfg.split} root={self.data_root}")
        patient_ids = self.ingestor.discover_patients()

        if self.cfg.max_patients and self.cfg.max_patients > 0:
            patient_ids = patient_ids[: self.cfg.max_patients]

        logger.info(f"Patients to process: {len(patient_ids)}")

        already_done: set[str] = set()
        if self.cfg.resume and self.cfg.write_jsonl:
            already_done = _read_jsonl_ids(self.jsonl_path)
            if already_done:
                logger.info(f"Resume enabled: {len(already_done)} embeddings already in JSONL")

        processed_count = 0
        embedded_count = 0
        skipped_no_lesion = 0
        skipped_no_seg = 0
        skipped_resume = 0
        errors = 0

        t0 = _now_ms()

        for pid in patient_ids:
            try:
                sample = self.ingestor.load_patient_data(pid)

                if self.cfg.only_with_seg and sample.segmentation is None:
                    skipped_no_seg += 1
                    continue

                # Preprocess
                processed = self.preprocessor.preprocess_patient(
                    modalities=sample.modalities,
                    segmentation=sample.segmentation,
                    augment=False,
                )

                img_4ch = self._build_image_tensor(processed)  # [4,D,H,W]

                # Choose mask source
                if self.cfg.use_pred_mask:
                    mask = self._predict_mask(img_4ch)
                else:
                    if "seg" not in processed:
                        skipped_no_seg += 1
                        continue
                    # processed["seg"] is torch tensor [1,D,H,W] (or similar)
                    seg_np = processed["seg"].squeeze(0).detach().cpu().numpy()
                    mask = _remap_brats_labels_np(seg_np)

                # Embed lesion (3D)
                vol_np = img_4ch.detach().cpu().numpy().astype(np.float32)  # [4,D,H,W]

                # Build deterministic ID for the lesion embeddings (one per patient)
                emb_id = f"{pid}|lesion_3d|0"

                if emb_id in already_done:
                    skipped_resume += 1
                    continue

                out = self.lesion_embedder.embed_from_volume_and_mask(
                    volume_4ch=vol_np,
                    mask=mask,
                    patient_id=pid,
                    margin=self.cfg.margin,
                )

                processed_count += 1

                if out is None:
                    skipped_no_lesion += 1
                    continue

                embedded_count += 1

                # Prepare JSONL record
                record = {
                    "id": emb_id,
                    "embeddings": out.embedding.tolist(),  # 768 floats
                    "metadata": out.meta,
                }

                if self.cfg.write_jsonl:
                    _atomic_append_jsonl(self.jsonl_path, record)

                if embedded_count % 25 == 0:
                    logger.info(
                        f"Progress: embedded={embedded_count} processed={processed_count} "
                        f"skipped(no_lesion)={skipped_no_lesion} skipped(no_seg)={skipped_no_seg} errors={errors}"
                    )

            except Exception as e:
                errors += 1
                logger.exception(f"Error processing patient {pid}: {e}")

        dt = (_now_ms() - t0) / 1000.0
        logger.info(
            "Pipeline finished | "
            f"processed={processed_count}, embedded={embedded_count}, "
            f"skipped_resume={skipped_resume}, skipped_no_lesion={skipped_no_lesion}, "
            f"skipped_no_seg={skipped_no_seg}, errors={errors}, time_sec={dt:.1f}"
        )

        # Optional index build
        if self.cfg.build_index:
            self.build_index(self.jsonl_path)

        return self.jsonl_path

    def build_index(self, jsonl_path: Path) -> None:
        if not jsonl_path.exists():
            raise FileNotFoundError(f"Embeddings JSONL not found: {jsonl_path}")

        store = create_vector_store(
            backend=self.cfg.backend,
            persist_dir=self.cfg.vector_db_dir,
            dim=model_config.EMBEDDING_DIM,
            metric=self.cfg.metric,
            collection_name=self.cfg.collection,
        )

        ids: List[str] = []
        metas: List[JsonDict] = []
        vecs: List[np.ndarray] = []

        logger.info(f"Building index | backend={self.cfg.backend} metric={self.cfg.metric} from {jsonl_path}")

        with jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                ids.append(obj["id"])
                metas.append(obj.get("metadata", {}))
                vecs.append(np.array(obj["embeddings"], dtype=np.float32))

        if not ids:
            logger.warning("No embeddings found in JSONL; skipping index build.")
            return

        mat = np.stack(vecs, axis=0).astype(np.float32)
        batch = 512
        for i in range(0, len(ids), batch):
            store.add(ids=ids[i:i + batch], vectors=mat[i:i + batch], metadatas=metas[i:i + batch])
            logger.info(f"Indexed {min(i + batch, len(ids))}/{len(ids)}")

        store.persist()
        logger.info(f"✅ Index persisted | total_vectors={store.size} | dir={self.cfg.vector_db_dir}")


# ---------------------------
# Search Command
# ---------------------------
def run_search(
    backend: str,
    metric: str,
    collection: str,
    persist_dir: Path,
    query: str,
    top_k: int,
    filter_json: Optional[str],
) -> None:
    store = create_vector_store(
        backend=backend,
        persist_dir=persist_dir,
        dim=model_config.EMBEDDING_DIM,
        metric=metric,
        collection_name=collection,
    )

    if store.size == 0:
        logger.warning("Vector store is empty. Run pipeline with --build_index first.")
        return

    embedder = TextEmbedder(model_name=model_config.TEXT_EMBEDDING_MODEL)
    q_vec = embedder.embed(query).embeddings[0]  # (768,)

    filt = json.loads(filter_json) if filter_json else None
    results = store.search(query_vector=q_vec, top_k=top_k, filter=filt)

    print("\n====================")
    print("Query:", query)
    print("TopK:", top_k)
    print("Filter:", filt)
    print("====================\n")

    for i, r in enumerate(results, start=1):
        meta = r.metadata or {}
        print(f"{i:02d}. id={r.id}  score={r.score:.4f}")
        # compact meta
        keys = ["type", "patient_id", "tumor_volume_voxels", "bbox", "centroid_zyx", "cube_size"]
        compact = {k: meta.get(k) for k in keys if k in meta}
        print("    meta:", compact if compact else meta)

    print()


# ---------------------------
# CLI
# ---------------------------
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("medragx-pipeline")
    sub = p.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="Run ingest→preprocess→(optional seg)→3D lesion embed→(optional index)")
    run.add_argument("--split", choices=["train", "val"], default="train")
    run.add_argument("--max_patients", type=int, default=0)
    run.add_argument("--seed", type=int, default=42)

    run.add_argument("--use_pred_mask", action="store_true", help="Use predicted mask from UNet instead of GT seg")
    run.add_argument("--checkpoint", type=str, default=None, help="Segmentation checkpoint path (.pt)")

    run.add_argument("--cube_size", type=str, default="64,64,64", help="Lesion cube size D,H,W")
    run.add_argument("--margin", type=int, default=5)

    run.add_argument("--write_jsonl", action="store_true", help="Write embeddings JSONL")
    run.add_argument("--no_resume", action="store_true", help="Disable resume; re-embed even if JSONL already has IDs")
    run.add_argument("--only_with_seg", action="store_true", help="Skip patients without GT seg (useful for train GT)")

    run.add_argument("--build_index", action="store_true", help="Build vector index after exporting embeddings")
    run.add_argument("--backend", choices=["faiss", "chroma"], default="faiss")
    run.add_argument("--metric", choices=["cosine", "l2"], default="cosine")
    run.add_argument("--collection", type=str, default="medragx")

    run.add_argument("--embeddings_dir", type=str, default=str(paths.EMBEDDINGS_DIR))
    run.add_argument("--vector_db_dir", type=str, default=str(paths.VECTOR_DB_DIR))

    search = sub.add_parser("search", help="Search vector store using a text query")
    search.add_argument("--backend", choices=["faiss", "chroma"], default="faiss")
    search.add_argument("--metric", choices=["cosine", "l2"], default="cosine")
    search.add_argument("--collection", type=str, default="medragx")
    search.add_argument("--persist_dir", type=str, default=str(paths.VECTOR_DB_DIR))
    search.add_argument("--query", type=str, required=True)
    search.add_argument("--top_k", type=int, default=10)
    search.add_argument("--filter", type=str, default=None, help='JSON filter e.g. \'{"type":"lesion_3d"}\'')

    return p.parse_args()


def main() -> None:
    args = _parse_args()

    if args.cmd == "run":
        cube = tuple(int(x) for x in args.cube_size.split(","))
        if len(cube) != 3:
            raise ValueError("--cube_size must be like '64,64,64'")

        cfg = PipelineConfig(
            split=args.split,
            max_patients=args.max_patients,
            seed=args.seed,
            use_pred_mask=args.use_pred_mask,
            checkpoint=Path(args.checkpoint) if args.checkpoint else None,
            cube_size=(cube[0], cube[1], cube[2]),
            margin=args.margin,
            write_jsonl=args.write_jsonl,
            resume=(not args.no_resume),
            build_index=args.build_index,
            backend=args.backend,
            metric=args.metric,
            collection=args.collection,
            embeddings_dir=Path(args.embeddings_dir),
            vector_db_dir=Path(args.vector_db_dir),
            only_with_seg=args.only_with_seg,
        )

        pipe = MedRAGXPipeline(cfg)
        jsonl_path = pipe.run()

        if cfg.write_jsonl:
            logger.info(f"Embeddings JSONL: {jsonl_path}")

    elif args.cmd == "search":
        run_search(
            backend=args.backend,
            metric=args.metric,
            collection=args.collection,
            persist_dir=Path(args.persist_dir),
            query=args.query,
            top_k=args.top_k,
            filter_json=args.filter,
        )
    else:
        raise ValueError(f"Unknown command: {args.cmd}")


if __name__ == "__main__":
    main()

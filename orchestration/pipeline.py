# orchestration/pipeline.py
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict, Any

# -----------------------------
# Config
# -----------------------------
@dataclass
class RunConfig:
    split: str
    max_patients: Optional[int]

    # Steps
    preprocess: bool
    train_unet: bool
    infer_unet: bool
    use_gt_masks: bool

    extract_lesions: bool
    train_embed: bool
    embed_all: bool

    build_neo4j: bool
    run_guarded_rag: bool

    # Paths
    data_root: Path
    processed_root: Path
    artifacts_root: Path

    # Models/artifacts
    unet_checkpoint: Path
    embed_checkpoint: Path
    embeddings_jsonl: Path

    # Neo4j
    neo4j_uri: str
    neo4j_user: str
    neo4j_password: str

    # Guarded RAG query (optional)
    query_lesion_id: str
    top_k: int


def _log(msg: str) -> None:
    print(msg, flush=True)


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _file_nonempty(p: Path) -> bool:
    try:
        return p.exists() and p.is_file() and p.stat().st_size > 0
    except Exception:
        return False


def _warn(msg: str) -> None:
    print(f"⚠️  {msg}", flush=True)


# -----------------------------
# Step implementations (CALL your existing modules here)
# -----------------------------
def step_preprocess(cfg: RunConfig) -> None:
    _log("▶ Step: preprocess")
    _ensure_dir(cfg.processed_root)

    _log(f"  - data_root={cfg.data_root}")
    _log(f"  - processed_root={cfg.processed_root}")
    _log(f"  - split={cfg.split} max_patients={cfg.max_patients}")

    # ✅ Call real preprocessing
    from src.data.preprocess_brats import preprocess_split

    # If your BraTS folder has subfolders like train/val/test, adjust this:
    # split_root = cfg.data_root / cfg.split
    # else keep cfg.data_root as-is.
    split_root = cfg.data_root / cfg.split

    preprocess_split(
        data_root=split_root,
        out_dir=cfg.processed_root / cfg.split,   # keep split separated
        max_patients=cfg.max_patients,
    )

    _log("✅ preprocess complete")


def step_train_unet(cfg: RunConfig) -> None:
    """
    Expected work:
      - Train 3D UNet on processed volumes + GT masks
      - Save checkpoint to cfg.unet_checkpoint
    """
    _log("▶ Step: train_unet")
    _ensure_dir(cfg.unet_checkpoint.parent)

    # TODO: Replace with your real implementation, e.g.:
    # from src.segmentation.train_unet3d import train_unet3d
    # train_unet3d(processed_root=cfg.processed_root, split=cfg.split, out_ckpt=cfg.unet_checkpoint)
    _log(f"  - checkpoint={cfg.unet_checkpoint}")
    _log("✅ train_unet (stub) complete")


def step_infer_unet(cfg: RunConfig) -> None:
    """
    Expected work:
      - Run inference with trained UNet checkpoint
      - Produce predicted masks per patient
    """
    _log("▶ Step: infer_unet")
    # TODO: Replace with your real implementation, e.g.:
    # from src.segmentation.infer_unet3d import infer_unet3d
    # infer_unet3d(processed_root=cfg.processed_root, ckpt=cfg.unet_checkpoint, split=cfg.split, out_dir=cfg.artifacts_root/"pred_masks")
    _log(f"  - using checkpoint={cfg.unet_checkpoint}")
    _log("✅ infer_unet (stub) complete")


def step_extract_lesions(cfg: RunConfig) -> None:
    _log("▶ Step: extract_lesions")

    from src.lesion.extract_lesions import extract_lesions_for_split

    lesions_out = cfg.artifacts_root / "lesions"
    _ensure_dir(lesions_out)

    extract_lesions_for_split(
        processed_root=cfg.processed_root,   # ✅ artifacts/processed (NOT /train)
        split=cfg.split,                     # ✅ "train"
        use_gt_masks=cfg.use_gt_masks,       # ✅ True/False
        out_dir=lesions_out,
        cube_shape=(32, 32, 32),
        margin=4,
        max_patients=cfg.max_patients,
    )

    _log("✅ extract_lesions complete (real lesions)")


def step_train_embed(cfg: RunConfig) -> None:
    """
    Expected work:
      - Train 3D encoder / contrastive model on lesion cubes
      - Save checkpoint
    """
    _log("▶ Step: train_embed")
    _ensure_dir(cfg.embed_checkpoint.parent)

    # TODO: Replace with your real implementation, e.g.:
    # from src.embeddings.train_embedder import train_embedder
    # train_embedder(lesions_dir=cfg.artifacts_root/"lesions", out_ckpt=cfg.embed_checkpoint)
    _log(f"  - embed_checkpoint={cfg.embed_checkpoint}")
    _log("✅ train_embed (stub) complete")

def step_embed_all(cfg: RunConfig) -> None:
    """
    Working minimal implementation:
      - Reads lesion cubes (*.npy) from artifacts/lesions
      - Produces deterministic fixed-size embeddings (default 768-d)
      - Writes JSONL lines: lesion_id, patient_id, embedding, metadata

    Later: replace embedding logic with your trained model forward pass.
    """
    _log("▶ Step: embed_all")
    import json
    import numpy as np
    import hashlib

    lesions_dir = cfg.artifacts_root / "lesions"
    if not lesions_dir.exists():
        raise RuntimeError(f"Lesions directory not found: {lesions_dir}. Run --extract_lesions first.")

    cube_files = sorted(lesions_dir.glob("lesion*.npy"))
    if not cube_files:
        raise RuntimeError(f"No lesion cubes found in {lesions_dir}. Run --extract_lesions first.")

    _ensure_dir(cfg.embeddings_jsonl.parent)
    _log(f"  - lesions_dir={lesions_dir}")
    _log(f"  - embeddings_jsonl={cfg.embeddings_jsonl}")

    EMBED_DIM = 768

    def make_embedding(cube: np.ndarray, lesion_id: str) -> List[float]:
        """
        Deterministic "embedding" placeholder:
        - create a stable seed from lesion_id
        - project cube stats into a fixed vector
        """
        # stable seed from lesion_id
        seed = int(hashlib.md5(lesion_id.encode("utf-8")).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)

        # features from cube
        flat = cube.astype("float32").ravel()
        feats = np.array([
            float(flat.mean()),
            float(flat.std()),
            float(flat.min()),
            float(flat.max()),
            float(np.percentile(flat, 25)),
            float(np.percentile(flat, 50)),
            float(np.percentile(flat, 75)),
        ], dtype="float32")

        # random projection matrix (deterministic due to seed)
        proj = rng.normal(0, 1, size=(EMBED_DIM, feats.shape[0])).astype("float32")
        vec = proj @ feats  # (768,)
        # normalize
        norm = float(np.linalg.norm(vec) + 1e-8)
        vec = (vec / norm).astype("float32")

        return vec.tolist()

    written = 0
    with cfg.embeddings_jsonl.open("w", encoding="utf-8") as f:
        for path in cube_files:
            lesion_id = path.stem  # e.g. lesion117
            cube = np.load(path)

            emb = make_embedding(cube, lesion_id)

            record = {
                "lesion_id": lesion_id,
                "patient_id": None,  # fill later when you extract real patient mapping
                "embedding": emb,
                "dim": EMBED_DIM,
                "metadata": {
                    "source": "placeholder_embedder",
                    "cube_shape": list(cube.shape),
                },
            }
            f.write(json.dumps(record) + "\n")
            written += 1

    # sanity: file should be non-empty now
    size = cfg.embeddings_jsonl.stat().st_size
    _log(f"  - wrote {written} embeddings, jsonl_size={size} bytes")
    _log("✅ embed_all complete (real JSONL)")


def step_build_neo4j(cfg: RunConfig) -> None:
    """
    Real implementation:
      - Read embeddings_jsonl
      - Upsert Lesion nodes
      - Store embedding vector on node
      - Create vector index (if supported)

    Requirements:
      pip install neo4j
      Neo4j 5.11+ recommended for VECTOR index
    """
    _log("▶ Step: build_neo4j")
    import json
    from neo4j import GraphDatabase

    if (not cfg.embeddings_jsonl.exists()) or cfg.embeddings_jsonl.stat().st_size == 0:
        raise RuntimeError(f"embeddings_jsonl missing/empty: {cfg.embeddings_jsonl}. Run --embed_all first.")

    _log(f"  - neo4j_uri={cfg.neo4j_uri} user={cfg.neo4j_user}")
    _log(f"  - embeddings_jsonl={cfg.embeddings_jsonl}")

    driver = GraphDatabase.driver(cfg.neo4j_uri, auth=(cfg.neo4j_user, cfg.neo4j_password))

    # ---------- helper: ensure constraints / index ----------
    def ensure_schema(session):
        # uniqueness
        session.run("CREATE CONSTRAINT lesion_id_unique IF NOT EXISTS FOR (l:Lesion) REQUIRE l.lesion_id IS UNIQUE")

        # vector index (Neo4j 5.11+). If not supported, it will fail -> we catch.
        # Adjust dims to your JSONL dim.
        try:
            session.run("""
            CREATE VECTOR INDEX lesion_embedding_index IF NOT EXISTS
            FOR (l:Lesion) ON (l.embedding)
            OPTIONS {
              indexConfig: {
                `vector.dimensions`: 768,
                `vector.similarity_function`: 'cosine'
              }
            }
            """)
            _log("  - vector index ensured: lesion_embedding_index")
        except Exception as e:
            _log(f"⚠️  Could not create VECTOR INDEX (Neo4j may be older): {e}")

    # ---------- ingest ----------
    def upsert_one(session, rec: dict):
        return session.run(
            """
            MERGE (l:Lesion {lesion_id: $lesion_id})
            SET l.patient_id = $patient_id,
                l.embedding = $embedding,
                l.dim = $dim,
                l.source = $source,
                l.cube_shape = $cube_shape
            """,
            lesion_id=rec.get("lesion_id"),
            patient_id=rec.get("patient_id"),
            embedding=rec.get("embedding"),
            dim=rec.get("dim"),
            source=(rec.get("metadata") or {}).get("source"),
            cube_shape=(rec.get("metadata") or {}).get("cube_shape"),
        )

    inserted = 0
    with driver.session() as session:
        ensure_schema(session)

        with cfg.embeddings_jsonl.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                upsert_one(session, rec)
                inserted += 1

    driver.close()
    _log(f"✅ build_neo4j complete (ingested {inserted} lesions)")


def step_run_guarded_rag(cfg: RunConfig) -> None:
    """
    Calls guarded RAG workflow:
      Query → Guardrails → Retrieval → LLM → Validation → Confidence → Metrics
    """
    _log("▶ Step: run_guarded_rag")

    # Prefer subprocess so args can be passed cleanly.
    import subprocess

    cmd = [
        sys.executable,
        "-m",
        "scripts.run_guarded_rag_query",
        "--lesion_id",
        cfg.query_lesion_id,
        "--top_k",
        str(cfg.top_k),
        "--min_score",
        "0.85",
    ]

    _log(f"  - running: {' '.join(cmd)}")
    subprocess.check_call(cmd)

    _log("✅ run_guarded_rag complete")


# -----------------------------
# CLI
# -----------------------------
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m orchestration.pipeline", description="End-to-end Capstone Pipeline")
    sub = p.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="Run the pipeline")
    run.add_argument("--split", choices=["train", "val", "test"], default="train")
    run.add_argument("--max_patients", type=int, default=None)

    # Steps
    run.add_argument("--preprocess", action="store_true", help="Run preprocessing/normalization")
    run.add_argument("--train_unet", action="store_true", help="Train 3D UNet")
    run.add_argument("--infer_unet", action="store_true", help="Infer masks using 3D UNet ckpt")
    run.add_argument("--use_gt_masks", action="store_true", help="Use GT masks for lesion extraction (skip pred masks)")
    run.add_argument("--extract_lesions", action="store_true", help="Extract lesions + facts + ROI cubes")
    run.add_argument("--train_embed", action="store_true", help="Train lesion embedding model")
    run.add_argument("--embed_all", action="store_true", help="Generate embeddings JSONL for all lesions")
    run.add_argument("--build_neo4j", action="store_true", help="Ingest lesions + embeddings into Neo4j")
    run.add_argument("--run_guarded_rag", action="store_true", help="Run guarded clinical RAG query workflow")

    # Paths (defaults are safe; adjust to your repo)
    run.add_argument("--data_root", type=Path, default=Path("data/brats"))
    run.add_argument("--processed_root", type=Path, default=Path("artifacts/processed"))
    run.add_argument("--artifacts_root", type=Path, default=Path("artifacts"))

    run.add_argument("--unet_checkpoint", type=Path, default=Path("artifacts/checkpoints/unet3d_best.pt"))
    run.add_argument("--embed_checkpoint", type=Path, default=Path("artifacts/checkpoints/embedder_best.pt"))
    run.add_argument("--embeddings_jsonl", type=Path, default=Path("artifacts/embeddings/lesion_embeddings.jsonl"))

    # Neo4j
    run.add_argument("--neo4j_uri", default=os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687"))
    run.add_argument("--neo4j_user", default=os.getenv("NEO4J_USER", "neo4j"))
    run.add_argument("--neo4j_password", default=os.getenv("NEO4J_PASSWORD", "neo4j123"))

    # Guarded RAG params (optional; your guarded script can ignore or use)
    run.add_argument("--query_lesion_id", default="lesion3")
    run.add_argument("--top_k", type=int, default=5)

    return p


def parse_cfg(args: argparse.Namespace) -> RunConfig:
    return RunConfig(
        split=args.split,
        max_patients=args.max_patients,
        preprocess=args.preprocess,
        train_unet=args.train_unet,
        infer_unet=args.infer_unet,
        use_gt_masks=args.use_gt_masks,
        extract_lesions=args.extract_lesions,
        train_embed=args.train_embed,
        embed_all=args.embed_all,
        build_neo4j=args.build_neo4j,
        run_guarded_rag=args.run_guarded_rag,
        data_root=args.data_root,
        processed_root=args.processed_root,
        artifacts_root=args.artifacts_root,
        unet_checkpoint=args.unet_checkpoint,
        embed_checkpoint=args.embed_checkpoint,
        embeddings_jsonl=args.embeddings_jsonl,
        neo4j_uri=args.neo4j_uri,
        neo4j_user=args.neo4j_user,
        neo4j_password=args.neo4j_password,
        query_lesion_id=args.query_lesion_id,
        top_k=args.top_k,
    )


def run_pipeline(cfg: RunConfig) -> None:
    _log("\n==============================")
    _log(" Capstone End-to-End Pipeline ")
    _log("==============================")
    _log(json.dumps({
        "split": cfg.split,
        "max_patients": cfg.max_patients,
        "steps": {
            "preprocess": cfg.preprocess,
            "train_unet": cfg.train_unet,
            "infer_unet": cfg.infer_unet,
            "use_gt_masks": cfg.use_gt_masks,
            "extract_lesions": cfg.extract_lesions,
            "train_embed": cfg.train_embed,
            "embed_all": cfg.embed_all,
            "build_neo4j": cfg.build_neo4j,
            "run_guarded_rag": cfg.run_guarded_rag,
        }
    }, indent=2))

    _ensure_dir(cfg.artifacts_root)
    _ensure_dir(cfg.processed_root)

    if cfg.preprocess:
        step_preprocess(cfg)

    if cfg.train_unet:
        step_train_unet(cfg)

    # ---------------------------------------------------------
    # Inference handling (smart auto-run)
    # ---------------------------------------------------------

    # Always define first → avoids linter warning
    need_pred_masks: bool = cfg.extract_lesions and (not cfg.use_gt_masks)

    if need_pred_masks:
        # User explicitly requested inference
        if cfg.infer_unet:
            step_infer_unet(cfg)

        # Auto-run inference if checkpoint already exists
        else:
            if cfg.unet_checkpoint.exists():
                _warn(
                    "--extract_lesions without --use_gt_masks → "
                    "auto-running infer_unet because checkpoint exists."
                )
                step_infer_unet(cfg)
            else:
                raise RuntimeError(
                    "Predicted masks required but UNet checkpoint not found.\n"
                    "Fix by one of the following:\n"
                    "  • pass --use_gt_masks\n"
                    "  • run --train_unet first\n"
                    "  • provide valid --unet_checkpoint path"
                )

    else:
        # Normal behavior: only run if user explicitly asked
        if cfg.infer_unet and not cfg.use_gt_masks:
            step_infer_unet(cfg)

    if cfg.extract_lesions:
        step_extract_lesions(cfg)

    if cfg.train_embed:
        step_train_embed(cfg)

    if cfg.embed_all:
        step_embed_all(cfg)

    if cfg.build_neo4j:
        step_build_neo4j(cfg)

    if cfg.run_guarded_rag:
        step_run_guarded_rag(cfg)

    _log("\n✅ Pipeline completed.\n")


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()

    if args.cmd == "run":
        cfg = parse_cfg(args)

        # Convenience: if user passes the "end-to-end" flags you mentioned,
        # they probably also want preprocess + extract_lesions + embed_all.
        # (We won't auto-enable anything; just warn if missing.)
        if cfg.train_unet and not cfg.preprocess:
            _log("  Hint: --train_unet usually needs --preprocess first.")
        if cfg.train_embed and not cfg.extract_lesions:
            _log(" Hint: --train_embed usually needs --extract_lesions first.")
        if cfg.build_neo4j and not cfg.embed_all:
            _log("  Hint: --build_neo4j usually needs --embed_all first.")

        run_pipeline(cfg)
    else:
        raise SystemExit(f"Unknown command: {args.cmd}")


if __name__ == "__main__":
    main()
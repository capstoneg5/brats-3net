# Datasets And Local Artifacts

This repository keeps code and lightweight metadata in Git, but excludes raw datasets, preprocessed tensors, model checkpoints, vector indexes, and generated artifacts.

## What Is In Git

- Source code under `app/`, `src/`, `scripts/`, `models/`, `config/`
- Lightweight processed metadata:
  - `data/processed/dataset_metadata_train.json`
  - `data/processed/dataset_metadata_val.json`

## What Is Not In Git

These stay local and must be regenerated:

- Raw BraTS data under `data/brats/train/`
- Preprocessed `.pt` tensors under `data/processed/train/`
- Knowledge graph artifacts under `artifacts/`
- FAISS index under `artifacts/lesion_faiss.index`
- Neo4j CSV exports under `artifacts/neo4j/` and `artifacts/neo4j_split/`
- Checkpoints, embeddings, uploads, eval outputs

## Expected Raw Data Layout

Place BraTS training cases under:

```text
data/brats/train/
  BraTS20_Training_001/
  BraTS20_Training_002/
  ...
```

Each case folder should contain the usual BraTS NIfTI files for:

- `flair`
- `t1`
- `t1ce`
- `t2`
- `seg`

The preprocessing script reads from `data/brats/train/` in [scripts/build_train_pt_from_nii.py](/Users/esai/Documents/capstoneg5/brats-3net/scripts/build_train_pt_from_nii.py).

## Regenerate Preprocessed Tensors

From the repo root:

```bash
cd /Users/esai/Documents/capstoneg5/brats-3net
source .venv/bin/activate
python scripts/build_train_pt_from_nii.py
```

This writes one `.pt` file per patient to:

```text
data/processed/train/
```

Each saved tensor bundle contains:

- `image`: 4-channel MRI volume
- `seg`: multiclass BraTS segmentation
- `mask`: binary tumor mask
- `patient_id`

## Regenerate Downstream Artifacts

After preprocessing, regenerate the local retrieval artifacts in this order:

### 1. Build lesion knowledge graph

```bash
python scripts/build_kg_3d_from_pt.py
```

Output:

- `artifacts/kg_3d.json`

### 2. Add similarity edges

```bash
python scripts/build_similarity_edges.py
```

Output:

- `artifacts/kg_3d_with_sim.json`

### 3. Build FAISS lesion index

```bash
python scripts/build_lesion_index.py
```

Outputs:

- `artifacts/lesion_faiss.index`
- `artifacts/lesion_ids.npy`

### 4. Export Neo4j CSV files

```bash
python scripts/export_kg_to_csv.py
python scripts/split_kg_csv_for_neo4j.py
```

Outputs:

- `artifacts/neo4j/nodes.csv`
- `artifacts/neo4j/edges.csv`
- `artifacts/neo4j_split/...`

### 5. Push embeddings into Neo4j

Start Neo4j separately, then:

```bash
export NEO4J_PASSWORD='your_password'
python scripts/push_embeddings_to_neo4j.py
```

This reads from:

- `artifacts/kg_3d.json`

## One-Pass Local Regeneration

If you already have raw BraTS data and dependencies installed:

```bash
cd /Users/esai/Documents/capstoneg5/brats-3net
source .venv/bin/activate
python scripts/build_train_pt_from_nii.py
python scripts/build_kg_3d_from_pt.py
python scripts/build_similarity_edges.py
python scripts/build_lesion_index.py
python scripts/export_kg_to_csv.py
python scripts/split_kg_csv_for_neo4j.py
```

## Verification

Useful checks after regeneration:

```bash
find data/processed/train -name '*.pt' | wc -l
ls -lh artifacts/kg_3d.json
ls -lh artifacts/kg_3d_with_sim.json
ls -lh artifacts/lesion_faiss.index
ls -lh artifacts/neo4j/nodes.csv
```

## Notes

- `artifacts/` and `test_data/` are intentionally excluded from Git.
- The repo currently uses local files for preprocessing and retrieval artifacts; it does not download datasets automatically.
- For broader pipeline details, see [PIPELINE_EXECUTION_GUIDE.md](/Users/esai/Documents/capstoneg5/brats-3net/PIPELINE_EXECUTION_GUIDE.md).

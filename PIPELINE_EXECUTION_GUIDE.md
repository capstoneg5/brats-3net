# MedRAG-X Pipeline Execution Guide
## Three Ways to Run the Complete Pipeline

---

## 🚀 Option 1: Bash Script (Interactive, Recommended)

**Features:**
- ✅ Interactive prompts
- ✅ Skip confirmation for completed steps
- ✅ Colored output
- ✅ Error handling
- ✅ Manual/automated Neo4j import choice

**Usage:**
```bash
cd ~/Documents/capstoneg5/brats-3net
chmod +x run_complete_pipeline.sh
./run_complete_pipeline.sh
```

**What it does:**
- Checks for existing outputs
- Asks before overwriting
- Provides progress feedback
- Allows manual Neo4j import if preferred

---

## 🐍 Option 2: Python Script (Interactive)

**Features:**
- ✅ Cross-platform (Windows/Mac/Linux)
- ✅ Interactive prompts
- ✅ Clean progress output
- ✅ Simplified version

**Usage:**
```bash
cd ~/Documents/capstoneg5/brats-3net
python run_complete_pipeline.py
```

**Same functionality as bash script, but in Python**

---

## ⚡ Option 3: Direct Commands (Non-Interactive)

**For automation / scripting:**

```bash
cd ~/Documents/capstoneg5/brats-3net

# 1. Preprocessing
python scripts/build_train_pt_from_nii.py

# 2. Build KG with embeddings
python scripts/build_kg_3d_from_pt.py

# 3. Add similarity edges
python scripts/build_similarity_edges.py

# 4. Build FAISS index
python scripts/build_lesion_index.py

# 5. Quality checks
python scripts/check_embedding_separation.py
python scripts/inspect_kg_3d.py
python scripts/inspect_similarity_edges.py

# 6. Export to CSV
python scripts/export_kg_to_csv.py

# 7. Start Neo4j
docker compose -f docker/docker-compose.yml up -d neo4j
sleep 30

# 8. Import to Neo4j
python neo4j_csv_import.py

# 9. Push embeddings
python scripts/push_embeddings_to_neo4j.py

# Done!
echo "🎉 Pipeline complete!"
```

---

## 📋 One-Liner (All Steps)

**For copy-paste execution:**

```bash
cd ~/Documents/capstoneg5/brats-3net && \
python scripts/build_train_pt_from_nii.py && \
python scripts/build_kg_3d_from_pt.py && \
python scripts/build_similarity_edges.py && \
python scripts/build_lesion_index.py && \
python scripts/check_embedding_separation.py && \
python scripts/inspect_kg_3d.py && \
python scripts/inspect_similarity_edges.py && \
python scripts/export_kg_to_csv.py && \
docker compose -f docker/docker-compose.yml up -d neo4j && \
sleep 30 && \
python neo4j_csv_import.py && \
python scripts/push_embeddings_to_neo4j.py && \
echo "🎉 Pipeline complete! Run 'streamlit run medragx_ui.py' to start UI"
```

---

## 🔍 Quick Verification

After running the pipeline, verify everything is ready:

```bash
# Check files exist
ls -lh data/kg_3d_with_sim.json          # Knowledge graph
ls -lh data/lesion_faiss.index           # FAISS index
ls -lh data/neo4j_import/nodes.csv       # Neo4j CSVs

# Check Neo4j is running
docker ps | grep neo4j

# Check Neo4j has data
# Open http://localhost:7474 and run:
MATCH (l:Lesion) RETURN count(l)
# Should return: 354 (or similar, depending on dataset)

# Test vector index
CALL db.index.vector.queryNodes('lesion_embedding_idx', 5, [0.1, 0.2, ...])
YIELD node, score
RETURN node.id, score LIMIT 5
```

---

## ⏱️ Expected Runtime

**On CPU (16 GB RAM, AMD Ryzen 5 5600H):**
- Step 1 (Preprocessing): ~30-45 minutes
- Step 2 (Test embeddings): ~5 minutes (optional)
- Step 3 (Build KG): ~20-30 minutes
- Step 4 (Similarity): ~10-15 minutes
- Step 5 (FAISS): ~2-5 minutes
- Step 6 (Quality checks): ~2-3 minutes
- Step 7 (Export CSV): ~1-2 minutes
- Step 8 (Start Neo4j): ~30 seconds
- Step 9 (Import): ~5-10 minutes
- Step 10 (Push embeddings): ~5-10 minutes

**Total: ~90-120 minutes (1.5-2 hours)**

**On GPU:**
Significantly faster (30-45 minutes total)

---

## 🛠️ Troubleshooting

### Issue: "Directory brats-3net not found"
```bash
# Make sure you're in the right location
cd ~/Documents/capstoneg5
ls -la  # Should see brats-3net directory
```

### Issue: "Script not found"
```bash
# Make sure scripts are in the right place
ls brats-3net/scripts/build_train_pt_from_nii.py
```

### Issue: "Permission denied"
```bash
# Make bash script executable
chmod +x run_complete_pipeline.sh
```

### Issue: "Docker not found"
```bash
# Install Docker Desktop or Docker Engine
# Mac: brew install --cask docker
# Ubuntu: sudo apt install docker.io docker-compose
```

### Issue: "Neo4j won't start"
```bash
# Check if port 7474/7687 is in use
lsof -i :7474
lsof -i :7687

# Kill existing Neo4j
docker stop neo4j
docker rm neo4j

# Restart
docker compose -f docker/docker-compose.yml up -d neo4j
```

### Issue: "Import to Neo4j failed"
```bash
# Check Neo4j logs
docker logs neo4j

# Verify CSV files exist
ls -lh data/neo4j_import/

# Try manual import via browser
# Open http://localhost:7474
```

---

## 🎯 After Pipeline Completes

### Test the System

**1. Command-line RAG query:**
```bash
cd ~/Documents/capstoneg5/brats-3net
python medrag_query_system.py
```

**2. Launch Web UI:**
```bash
streamlit run medragx_ui.py
```
Opens at: http://localhost:8501

**3. Test Neo4j queries:**
```bash
# Open browser: http://localhost:7474
# Username: neo4j
# Password: medrag_password_123

# Run test query:
MATCH (l:Lesion)
CALL db.index.vector.queryNodes('lesion_embedding_idx', 10, l.embedding)
YIELD node, score
RETURN node.id, score
LIMIT 10
```

---

## 📊 What Gets Created

```
brats-3net/
├── data/
│   ├── processed/
│   │   └── train/              # ✅ Step 1: .pt tensors
│   ├── kg_3d_with_sim.json     # ✅ Step 3: Knowledge graph
│   ├── lesion_faiss.index      # ✅ Step 5: FAISS index
│   ├── lesion_faiss.index.meta # ✅ Step 5: FAISS metadata
│   └── neo4j_import/
│       ├── nodes.csv           # ✅ Step 7: Neo4j nodes
│       └── edges.csv           # ✅ Step 7: Neo4j edges
└── vector_db/                  # ✅ Vector store (if using)
```

---

## 🎉 Summary

**Choose your method:**

| Method | Best For | Pros | Cons |
|--------|----------|------|------|
| **Bash Script** | Most users | Interactive, user-friendly | Requires bash |
| **Python Script** | Cross-platform | Works everywhere | Still interactive |
| **One-liner** | Automation | No interaction needed | No progress feedback |

**Recommendation:** Use the **bash script** for first run, then use **one-liner** for re-runs.

---

## 🚀 Quick Start (TL;DR)

```bash
# Download scripts to brats-3net directory
cd ~/Documents/capstoneg5/brats-3net

# Run pipeline
chmod +x run_complete_pipeline.sh
./run_complete_pipeline.sh

# Wait ~90 minutes

# Launch UI when done
streamlit run medragx_ui.py
```

**That's it!** Your complete MedRAG-X system will be ready.
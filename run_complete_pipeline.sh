#!/bin/bash
# MedRAG-X Complete Pipeline Automation
# Runs entire preprocessing → Neo4j deployment pipeline

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_DIR="brats-3net"
DOCKER_COMPOSE="docker/docker-compose.yml"

echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║      MedRAG-X Complete Pipeline Automation                ║${NC}"
echo -e "${BLUE}║      End-to-End Processing: NIfTI → Neo4j                  ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Function to print step header
print_step() {
    echo ""
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}STEP $1: $2${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

# Function to check if step should be skipped
should_skip() {
    if [ -f "$1" ]; then
        echo -e "${YELLOW}⚠️  Output file $1 exists. Skip this step? (y/n)${NC}"
        read -r response
        if [[ "$response" =~ ^[Yy]$ ]]; then
            echo -e "${YELLOW}⏭️  Skipped${NC}"
            return 0
        fi
    fi
    return 1
}

# Change to project directory
cd "$PROJECT_DIR" || { echo -e "${RED}❌ Directory $PROJECT_DIR not found${NC}"; exit 1; }

# ============================================================================
# STEP 1: Preprocessing (NIfTI → .pt)
# ============================================================================
print_step "1" "Preprocessing (NIfTI → .pt tensors)"

if should_skip "data/processed/train"; then
    :
else
    echo -e "${YELLOW}⏳ Converting NIfTI files to PyTorch tensors...${NC}"
    python scripts/build_train_pt_from_nii.py
    echo -e "${GREEN}✅ Preprocessing complete${NC}"
fi

# ============================================================================
# STEP 2: Test Embedding Pipeline (Optional)
# ============================================================================
print_step "2" "Test Embedding Pipeline (Optional)"

echo -e "${YELLOW}Run embedding pipeline test? (y/n)${NC}"
read -r run_test

if [[ "$run_test" =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}⏳ Testing embedding generation...${NC}"
    python scripts/embed_from_pt.py
    echo -e "${GREEN}✅ Embedding test complete${NC}"
else
    echo -e "${YELLOW}⏭️  Skipped embedding test${NC}"
fi

# ============================================================================
# STEP 3: Build Knowledge Graph with Embeddings
# ============================================================================
print_step "3" "Build Knowledge Graph with Embeddings"

if should_skip "data/kg_3d_with_sim.json"; then
    :
else
    echo -e "${YELLOW}⏳ Building 3D knowledge graph...${NC}"
    python scripts/build_kg_3d_from_pt.py
    echo -e "${GREEN}✅ Knowledge graph built${NC}"
fi

# ============================================================================
# STEP 4: Add Similarity Edges
# ============================================================================
print_step "4" "Add Similarity Edges"

echo -e "${YELLOW}⏳ Computing lesion similarities and adding edges...${NC}"
python scripts/build_similarity_edges.py
echo -e "${GREEN}✅ Similarity edges added${NC}"

# ============================================================================
# STEP 5: Build FAISS Index
# ============================================================================
print_step "5" "Build FAISS Index for Fast Retrieval"

if should_skip "data/lesion_faiss.index"; then
    :
else
    echo -e "${YELLOW}⏳ Building FAISS vector index...${NC}"
    python scripts/build_lesion_index.py
    echo -e "${GREEN}✅ FAISS index built${NC}"
fi

# ============================================================================
# STEP 6: Quality Checks
# ============================================================================
print_step "6" "Quality Checks"

echo -e "${YELLOW}⏳ Running quality checks...${NC}"

echo -e "${BLUE}   6a. Checking embedding separation...${NC}"
python scripts/check_embedding_separation.py

echo -e "${BLUE}   6b. Inspecting knowledge graph...${NC}"
python scripts/inspect_kg_3d.py

echo -e "${BLUE}   6c. Inspecting similarity edges...${NC}"
python scripts/inspect_similarity_edges.py

echo -e "${GREEN}✅ Quality checks complete${NC}"

# ============================================================================
# STEP 7: Export for Neo4j
# ============================================================================
print_step "7" "Export Knowledge Graph to CSV (for Neo4j)"

if should_skip "data/neo4j_import/nodes.csv"; then
    :
else
    echo -e "${YELLOW}⏳ Exporting knowledge graph to CSV format...${NC}"
    python scripts/export_kg_to_csv.py
    echo -e "${GREEN}✅ CSV export complete${NC}"
fi

# ============================================================================
# STEP 8: Start Neo4j
# ============================================================================
print_step "8" "Start Neo4j Database"

echo -e "${YELLOW}⏳ Checking Neo4j status...${NC}"

if docker ps | grep -q neo4j; then
    echo -e "${GREEN}✅ Neo4j is already running${NC}"
else
    echo -e "${YELLOW}⏳ Starting Neo4j container...${NC}"
    docker compose -f "$DOCKER_COMPOSE" up -d neo4j

    echo -e "${YELLOW}⏳ Waiting for Neo4j to be ready (30 seconds)...${NC}"
    sleep 30

    echo -e "${GREEN}✅ Neo4j started${NC}"
    echo -e "${BLUE}   Access at: http://localhost:7474${NC}"
    echo -e "${BLUE}   Bolt: bolt://localhost:7687${NC}"
fi

# ============================================================================
# STEP 9: Import CSVs to Neo4j (Manual/Automated)
# ============================================================================
print_step "9" "Import CSVs to Neo4j"

echo -e "${YELLOW}Choose import method:${NC}"
echo -e "  ${BLUE}1)${NC} Manual import (opens browser, you run Cypher queries)"
echo -e "  ${BLUE}2)${NC} Automated import (uses neo4j_csv_import.py script)"
echo -e "  ${BLUE}3)${NC} Skip (already imported)"
read -r import_choice

case $import_choice in
    1)
        echo -e "${YELLOW}📋 Manual Import Instructions:${NC}"
        echo ""
        echo -e "${BLUE}1. Open Neo4j Browser: http://localhost:7474${NC}"
        echo -e "${BLUE}2. Login with:${NC}"
        echo -e "   Username: neo4j"
        echo -e "   Password: medrag_password_123"
        echo ""
        echo -e "${BLUE}3. Run these Cypher queries:${NC}"
        echo ""
        echo -e "${GREEN}// Import Patients${NC}"
        echo "LOAD CSV WITH HEADERS FROM 'file:///patients.csv' AS row"
        echo "CREATE (p:Patient {id: row.id, name: row.name});"
        echo ""
        echo -e "${GREEN}// Import Lesions${NC}"
        echo "LOAD CSV WITH HEADERS FROM 'file:///lesions.csv' AS row"
        echo "CREATE (l:Lesion {id: row.id, patient_id: row.patient_id});"
        echo ""
        echo -e "${GREEN}// Create relationships${NC}"
        echo "MATCH (p:Patient), (l:Lesion)"
        echo "WHERE p.id = l.patient_id"
        echo "CREATE (p)-[:HAS_LESION]->(l);"
        echo ""
        echo -e "${YELLOW}Press Enter when import is complete...${NC}"
        read -r
        ;;
    2)
        echo -e "${YELLOW}⏳ Running automated CSV import...${NC}"
        python neo4j_csv_import.py
        echo -e "${GREEN}✅ CSV import complete${NC}"
        ;;
    3)
        echo -e "${YELLOW}⏭️  Skipped CSV import${NC}"
        ;;
    *)
        echo -e "${RED}❌ Invalid choice. Skipping import.${NC}"
        ;;
esac

# ============================================================================
# STEP 10: Push Embeddings to Neo4j
# ============================================================================
print_step "10" "Push Embeddings to Neo4j"

echo -e "${YELLOW}⏳ Uploading embeddings to Neo4j vector index...${NC}"
python scripts/push_embeddings_to_neo4j.py
echo -e "${GREEN}✅ Embeddings pushed to Neo4j${NC}"

# ============================================================================
# COMPLETION
# ============================================================================
echo ""
echo -e "${GREEN}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║                   🎉 PIPELINE COMPLETE! 🎉                 ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${BLUE}✅ All steps completed successfully!${NC}"
echo ""
echo -e "${YELLOW}📊 System Status:${NC}"
echo -e "   ${GREEN}✓${NC} Preprocessing complete"
echo -e "   ${GREEN}✓${NC} Knowledge graph built"
echo -e "   ${GREEN}✓${NC} FAISS index ready"
echo -e "   ${GREEN}✓${NC} Neo4j database populated"
echo -e "   ${GREEN}✓${NC} Vector embeddings indexed"
echo ""
echo -e "${YELLOW}🚀 Next Steps:${NC}"
echo -e "   ${BLUE}1.${NC} Test RAG queries: ${GREEN}python medrag_query_system.py${NC}"
echo -e "   ${BLUE}2.${NC} Launch UI: ${GREEN}streamlit run medragx_ui.py${NC}"
echo -e "   ${BLUE}3.${NC} Neo4j Browser: ${GREEN}http://localhost:7474${NC}"
echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
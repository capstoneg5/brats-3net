#!/usr/bin/env python3
"""
MedRAG-X Complete Pipeline Automation (Python Version)
Runs entire preprocessing → Neo4j deployment pipeline
"""

import subprocess
import sys
import time
from pathlib import Path


class Colors:
    """ANSI color codes"""
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    NC = '\033[0m'


def run_command(cmd: str, desc: str):
    """Run command with error handling"""
    print(f"{Colors.YELLOW}⏳ {desc}...{Colors.NC}")
    try:
        subprocess.run(cmd, shell=True, check=True)
        print(f"{Colors.GREEN}✅ Complete{Colors.NC}")
    except subprocess.CalledProcessError as e:
        print(f"{Colors.RED}❌ Failed: {e}{Colors.NC}")
        sys.exit(1)


# Header
print(f"{Colors.BLUE}MedRAG-X Complete Pipeline{Colors.NC}\n")

# Step 1
print(f"{Colors.GREEN}[1/10] Preprocessing{Colors.NC}")
run_command("python scripts/build_train_pt_from_nii.py", "NIfTI → PyTorch")

# Step 2
print(f"\n{Colors.GREEN}[2/10] Test Embeddings (optional){Colors.NC}")
if input("Run test? (y/n): ").lower() == 'y':
    run_command("python scripts/embed_from_pt.py", "Testing embeddings")

# Step 3
print(f"\n{Colors.GREEN}[3/10] Build Knowledge Graph{Colors.NC}")
run_command("python scripts/build_kg_3d_from_pt.py", "Building KG")

# Step 4
print(f"\n{Colors.GREEN}[4/10] Add Similarity Edges{Colors.NC}")
run_command("python scripts/build_similarity_edges.py", "Adding edges")

# Step 5
print(f"\n{Colors.GREEN}[5/10] Build FAISS Index{Colors.NC}")
run_command("python scripts/build_lesion_index.py", "Building index")

# Step 6
print(f"\n{Colors.GREEN}[6/10] Quality Checks{Colors.NC}")
run_command("python scripts/check_embedding_separation.py", "Check embeddings")
run_command("python scripts/inspect_kg_3d.py", "Inspect KG")
run_command("python scripts/inspect_similarity_edges.py", "Inspect edges")

# Step 7
print(f"\n{Colors.GREEN}[7/10] Export to CSV{Colors.NC}")
run_command("python scripts/export_kg_to_csv.py", "Exporting")

# Step 8
print(f"\n{Colors.GREEN}[8/10] Start Neo4j{Colors.NC}")
if subprocess.run("docker ps | grep neo4j", shell=True, capture_output=True).returncode != 0:
    run_command("docker compose -f docker/docker-compose.yml up -d neo4j", "Starting Neo4j")
    print(f"{Colors.YELLOW}Waiting 30s...{Colors.NC}")
    time.sleep(30)
else:
    print(f"{Colors.GREEN}✅ Already running{Colors.NC}")

# Step 9
print(f"\n{Colors.GREEN}[9/10] Import to Neo4j{Colors.NC}")
run_command("python neo4j_csv_import.py", "Importing CSVs")

# Step 10
print(f"\n{Colors.GREEN}[10/10] Push Embeddings{Colors.NC}")
run_command("python scripts/push_embeddings_to_neo4j.py", "Uploading embeddings")

# Done
print(f"\n{Colors.GREEN}🎉 PIPELINE COMPLETE!{Colors.NC}")
print(f"\n{Colors.YELLOW}Next:{Colors.NC}")
print(f"  python medrag_query_system.py")
print(f"  streamlit run medragx_ui.py")
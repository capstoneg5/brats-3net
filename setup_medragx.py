#!/usr/bin/env python3
"""
MedRAG-X Setup Script
Configures data paths and validates environment
"""

import os
import sys
from pathlib import Path
import json


class Colors:
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    NC = '\033[0m'


def find_brats_data():
    """Find BraTS data directory"""

    print(f"{Colors.BLUE}🔍 Searching for BraTS data...{Colors.NC}\n")

    # Common locations to check
    possible_locations = [
        Path.home() / "/Documents/capstoneg5/brats-3net/artifacts/processed/train",
        Path.home() / "Documents/capstoneg5/data/train",
        Path.home() / "Documents/capstoneg5/BraTS20_Training_001",
        Path.home() / "Downloads/BraTS20",
        Path.home() / "Desktop/BraTS20",
        Path.cwd() / "data/brats/train",
        Path.cwd() / "data/train",
        Path.cwd() / "../data/brats/train",
    ]

    # Check each location
    found_locations = []
    for loc in possible_locations:
        if loc.exists() and loc.is_dir():
            # Check if it contains BraTS patient folders
            subdirs = [d for d in loc.iterdir() if d.is_dir() and 'BraTS' in d.name]
            if subdirs:
                found_locations.append(loc)
                print(f"{Colors.GREEN}✅ Found: {loc}{Colors.NC}")
                print(f"   Contains {len(subdirs)} patient folders")

    if not found_locations:
        print(f"{Colors.RED}❌ No BraTS data found in common locations{Colors.NC}\n")
        return None

    if len(found_locations) == 1:
        return found_locations[0]

    # Multiple locations found, ask user
    print(f"\n{Colors.YELLOW}Multiple locations found. Choose one:{Colors.NC}")
    for i, loc in enumerate(found_locations, 1):
        print(f"  {i}) {loc}")

    while True:
        try:
            choice = int(input(f"\n{Colors.YELLOW}Enter number (1-{len(found_locations)}): {Colors.NC}"))
            if 1 <= choice <= len(found_locations):
                return found_locations[choice - 1]
        except (ValueError, KeyboardInterrupt):
            pass


def update_config_paths(brats_train_path: Path):
    """Update config/paths.py with correct data location"""

    config_file = Path("config/paths.py")

    if not config_file.exists():
        print(f"{Colors.RED}❌ config/paths.py not found{Colors.NC}")
        return False

    # Read current config
    with open(config_file, 'r') as f:
        content = f.read()

    # Determine the base data directory
    if 'brats/train' in str(brats_train_path):
        data_root = brats_train_path.parent.parent
    else:
        data_root = brats_train_path.parent

    # Create new config content
    new_content = f'''"""
Configuration paths for MedRAG-X project
Auto-configured by setup script
"""

from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent.parent

# Data directories (AUTO-CONFIGURED)
DATA_ROOT = Path("{data_root}")
BRATS_TRAIN_RAW = Path("{brats_train_path}")
BRATS_VAL_RAW = DATA_ROOT / "brats" / "val"

# Processed data
PROCESSED_ROOT = DATA_ROOT / "processed"
TRAIN_PROCESSED = PROCESSED_ROOT / "train"
VAL_PROCESSED = PROCESSED_ROOT / "val"

# Outputs
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
ARTIFACTS_DIR = OUTPUT_ROOT / "artifacts"
VECTOR_DB_DIR = OUTPUT_ROOT / "vector_db"

# Knowledge graphs
KG_DIR = DATA_ROOT / "kg"
KG_3D_PATH = DATA_ROOT / "kg_3d_with_sim.json"

# Embeddings
EMBEDDINGS_DIR = DATA_ROOT / "embeddings"
LESION_EMBEDDINGS = EMBEDDINGS_DIR / "lesions.jsonl"

# FAISS indices
FAISS_INDEX_PATH = DATA_ROOT / "lesion_faiss.index"
FAISS_META_PATH = DATA_ROOT / "lesion_faiss.index.meta"

# Neo4j import
NEO4J_IMPORT_DIR = DATA_ROOT / "neo4j_import"

# Create directories
for path in [
    PROCESSED_ROOT, TRAIN_PROCESSED, VAL_PROCESSED,
    OUTPUT_ROOT, ARTIFACTS_DIR, VECTOR_DB_DIR,
    KG_DIR, EMBEDDINGS_DIR, NEO4J_IMPORT_DIR
]:
    path.mkdir(parents=True, exist_ok=True)
'''

    # Backup original
    backup_file = config_file.with_suffix('.py.backup')
    if not backup_file.exists():
        with open(backup_file, 'w') as f:
            f.write(content)
        print(f"{Colors.YELLOW}📋 Backed up original config to {backup_file}{Colors.NC}")

    # Write new config
    with open(config_file, 'w') as f:
        f.write(new_content)

    print(f"{Colors.GREEN}✅ Updated config/paths.py{Colors.NC}")
    print(f"   DATA_ROOT: {data_root}")
    print(f"   BRATS_TRAIN_RAW: {brats_train_path}")

    return True


def create_directory_structure(data_root: Path):
    """Create required directory structure"""

    print(f"\n{Colors.BLUE}📁 Creating directory structure...{Colors.NC}")

    dirs_to_create = [
        data_root / "processed/train",
        data_root / "processed/val",
        data_root / "kg",
        data_root / "embeddings",
        data_root / "neo4j_import",
        Path("outputs/artifacts"),
        Path("outputs/vector_db"),
    ]

    for dir_path in dirs_to_create:
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"{Colors.GREEN}✅ {dir_path}{Colors.NC}")


def verify_environment():
    """Verify Python packages"""

    print(f"\n{Colors.BLUE}🔍 Verifying environment...{Colors.NC}")

    required_packages = [
        'torch',
        'numpy',
        'nibabel',
        'scipy',
        'monai',
        'faiss',  # or 'faiss-cpu'
        'neo4j',
    ]

    missing = []
    for pkg in required_packages:
        try:
            __import__(pkg)
            print(f"{Colors.GREEN}✅ {pkg}{Colors.NC}")
        except ImportError:
            print(f"{Colors.RED}❌ {pkg}{Colors.NC}")
            missing.append(pkg)

    if missing:
        print(f"\n{Colors.YELLOW}Missing packages: {', '.join(missing)}{Colors.NC}")
        print(f"{Colors.YELLOW}Install with:{Colors.NC}")
        print(f"  pip install {' '.join(missing)}")
        return False

    return True


def main():
    """Main setup flow"""

    print(f"{Colors.BLUE}╔════════════════════════════════════════════╗{Colors.NC}")
    print(f"{Colors.BLUE}║      MedRAG-X Setup & Configuration        ║{Colors.NC}")
    print(f"{Colors.BLUE}╚════════════════════════════════════════════╝{Colors.NC}\n")

    # 1. Find BraTS data
    brats_path = find_brats_data()

    if not brats_path:
        print(f"\n{Colors.YELLOW}Manual configuration required:{Colors.NC}")
        print(f"  1. Download BraTS dataset from Kaggle")
        print(f"  2. Extract to a directory")
        print(f"  3. Enter the path to the training data:")

        manual_path = input(f"\n{Colors.YELLOW}Path: {Colors.NC}").strip()
        brats_path = Path(manual_path)

        if not brats_path.exists():
            print(f"{Colors.RED}❌ Path does not exist: {brats_path}{Colors.NC}")
            sys.exit(1)

    # 2. Update config
    print(f"\n{Colors.BLUE}⚙️  Updating configuration...{Colors.NC}")
    if not update_config_paths(brats_path):
        sys.exit(1)

    # 3. Create directories
    data_root = brats_path.parent.parent if 'brats/train' in str(brats_path) else brats_path.parent
    create_directory_structure(data_root)

    # 4. Verify environment
    if not verify_environment():
        print(f"\n{Colors.YELLOW}⚠️  Some packages missing. Install them before running pipeline.{Colors.NC}")

    # 5. Summary
    print(f"\n{Colors.GREEN}╔════════════════════════════════════════════╗{Colors.NC}")
    print(f"{Colors.GREEN}║          ✅ Setup Complete!                ║{Colors.NC}")
    print(f"{Colors.GREEN}╚════════════════════════════════════════════╝{Colors.NC}\n")

    print(f"{Colors.YELLOW}📊 Configuration Summary:{Colors.NC}")
    print(f"   Data Location: {brats_path}")
    print(f"   Config File: config/paths.py")
    print(f"   Directories: Created")

    print(f"\n{Colors.YELLOW}🚀 Next Steps:{Colors.NC}")
    print(f"   1. Run: {Colors.GREEN}python run_complete_pipeline.py{Colors.NC}")
    print(f"   2. Or: {Colors.GREEN}./run_complete_pipeline.sh{Colors.NC}")

    print(f"\n{Colors.BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{Colors.NC}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}⚠️  Setup cancelled{Colors.NC}")
        sys.exit(1)
    except Exception as e:
        print(f"\n{Colors.RED}❌ Setup failed: {e}{Colors.NC}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
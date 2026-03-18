#!/usr/bin/env bash
# fix_init_files.sh — Run from your brats-3net project root
# Creates missing __init__.py files so 'from src.xxx' imports work

set -e

dirs=(
    "app"
    "src"
    "src/rag"
    "src/data"
    "src/lesion"
    "src/embeddings"
    "src/eval"
    "src/graph"
    "src/guardrails"
    "orchestration"
    "preprocessing"
    "retrival"
    "scripts"
    "models"
    "models/segmentation"
    "models/embeddings"
)

for d in "${dirs[@]}"; do
    if [ -d "$d" ]; then
        init_file="$d/__init__.py"
        if [ ! -f "$init_file" ]; then
            touch "$init_file"
            echo "Created: $init_file"
        else
            echo "Exists:  $init_file"
        fi
    else
        echo "Skip:    $d (directory not found)"
    fi
done

echo ""
echo "Done. All __init__.py files are in place."
echo "Run with: PYTHONPATH=. streamlit run app.py"
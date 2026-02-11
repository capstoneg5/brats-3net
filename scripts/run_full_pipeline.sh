#!/usr/bin/env bash
set -e  # stop on first error

echo "======================================"
echo " Running Capstone End-to-End Pipeline "
echo "======================================"

python -m orchestration.pipeline run \
  --split train \
  --preprocess \
  --use_gt_masks \
  --extract_lesions \
  --train_embed \
  --embed_all \
  --build_neo4j \
  --run_guarded_rag

echo ""
echo "✅ Full pipeline finished successfully."
#!/bin/bash
# Fabric Evaluation Pipeline
# 1. Extract foreground masks (SAM2)
# 2. Run fabric tracking experiments

set -e  # Exit on error

# # ==============================================================================
# # STEP 1: Foreground Mask Extraction
# # ==============================================================================

# echo "=========================================="
# echo "STEP 1: Foreground Mask Extraction"
# echo "=========================================="

# echo "Starting foreground mask extraction for cloth_no_occlusion_back_3sec..."
# for chunk in 0 3 7 12 20; do
#   echo "Processing chunk $chunk..."
#   python obtain_foreground_mask.py \
#     --chunk_path /mnt/mydisk/captured_data_double_arm/cloth_no_occlusion_back_3sec/chunk_${chunk}
# done

# echo "Starting foreground mask extraction for cloth_no_occlusion_back_4sec..."
# for chunk in 8 13; do
#   echo "Processing chunk $chunk..."
#   python obtain_foreground_mask.py \
#     --chunk_path /mnt/mydisk/captured_data_double_arm/cloth_no_occlusion_back_4sec/chunk_${chunk}
# done

# echo "Starting foreground mask extraction for cloth_no_occlusion_front_3sec..."
# for chunk in 2 5 6 7 11 14 17; do
#   echo "Processing chunk $chunk..."
#   python obtain_foreground_mask.py \
#     --chunk_path /mnt/mydisk/captured_data_double_arm/cloth_no_occlusion_front_3sec/chunk_${chunk}
# done

# echo "Starting foreground mask extraction for cloth_no_occlusion_front_4sec..."
# for chunk in 15 21 22 23 27 28; do
#   echo "Processing chunk $chunk..."
#   python obtain_foreground_mask.py \
#     --chunk_path /mnt/mydisk/captured_data_double_arm/cloth_no_occlusion_front_4sec/chunk_${chunk}
# done

# ==============================================================================
# STEP 2: Fabric Tracking Experiments
# ==============================================================================

echo ""
echo "=========================================="
echo "STEP 2: Fabric Tracking Experiments"
echo "=========================================="

# Common parameters
CLIP_SECONDS=10
GRID_ROWS=6
GRID_COLS=6

echo "Running experiments for cloth_no_occlusion_back_3sec..."
for chunk in 0 3 7 12 20; do
  echo "  Experiment: chunk $chunk..."
  python fabric_batch_experiment.py \
    --dataset cloth_no_occlusion_back_3sec \
    --chunk $chunk \
    --clip_seconds $CLIP_SECONDS \
    --grid_rows $GRID_ROWS \
    --grid_cols $GRID_COLS
done

echo "Running experiments for cloth_no_occlusion_back_4sec..."
for chunk in 8 13; do
  echo "  Experiment: chunk $chunk..."
  python fabric_batch_experiment.py \
    --dataset cloth_no_occlusion_back_4sec \
    --chunk $chunk \
    --clip_seconds $CLIP_SECONDS \
    --grid_rows $GRID_ROWS \
    --grid_cols $GRID_COLS
done

echo "Running experiments for cloth_no_occlusion_front_3sec..."
for chunk in 2 5 6 7 11 14 17; do
  echo "  Experiment: chunk $chunk..."
  python fabric_batch_experiment.py \
    --dataset cloth_no_occlusion_front_3sec \
    --chunk $chunk \
    --clip_seconds $CLIP_SECONDS \
    --grid_rows $GRID_ROWS \
    --grid_cols $GRID_COLS
done

echo "Running experiments for cloth_no_occlusion_front_4sec..."
for chunk in 15 21 22 23 27 28; do
  echo "  Experiment: chunk $chunk..."
  python fabric_batch_experiment.py \
    --dataset cloth_no_occlusion_front_4sec \
    --chunk $chunk \
    --clip_seconds $CLIP_SECONDS \
    --grid_rows $GRID_ROWS \
    --grid_cols $GRID_COLS
done

echo ""
echo "=========================================="
echo "ALL EXPERIMENTS COMPLETE!"
echo "=========================================="
echo "Results saved to: ./fabric_evaluation_results/"
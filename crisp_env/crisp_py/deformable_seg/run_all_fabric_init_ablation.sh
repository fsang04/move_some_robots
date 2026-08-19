#!/bin/bash
# Run fabric initialization ablation on all datasets and chunks
# Matches the folder structure of fabric_evaluation_results

set -e

EVAL_FRAMES=50
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "Fabric Initialization Ablation - All Datasets"
echo "Eval frames: $EVAL_FRAMES"
echo "=========================================="

# cloth_no_occlusion_back_3sec chunks
DATASET="cloth_no_occlusion_back_3sec"
CHUNKS=(0 3 7 12 20)
for CHUNK in "${CHUNKS[@]}"; do
    echo ""
    echo ">>> Running $DATASET chunk $CHUNK"
    pixi run -e deform-tracker python fabric_initialization_ablation.py \
        --dataset "$DATASET" \
        --chunk "$CHUNK" \
        --eval_frames "$EVAL_FRAMES" \
        2>&1 | tee -a "fabric_init_ablation_results/${DATASET}/chunk_${CHUNK}/log.txt"
done

# cloth_no_occlusion_back_4sec chunks
DATASET="cloth_no_occlusion_back_4sec"
CHUNKS=(8 13)
for CHUNK in "${CHUNKS[@]}"; do
    echo ""
    echo ">>> Running $DATASET chunk $CHUNK"
    pixi run -e deform-tracker python fabric_initialization_ablation.py \
        --dataset "$DATASET" \
        --chunk "$CHUNK" \
        --eval_frames "$EVAL_FRAMES" \
        2>&1 | tee -a "fabric_init_ablation_results/${DATASET}/chunk_${CHUNK}/log.txt"
done

# cloth_no_occlusion_front_3sec chunks
DATASET="cloth_no_occlusion_front_3sec"
CHUNKS=(2 5 6 7 11 14 17)
for CHUNK in "${CHUNKS[@]}"; do
    echo ""
    echo ">>> Running $DATASET chunk $CHUNK"
    pixi run -e deform-tracker python fabric_initialization_ablation.py \
        --dataset "$DATASET" \
        --chunk "$CHUNK" \
        --eval_frames "$EVAL_FRAMES" \
        2>&1 | tee -a "fabric_init_ablation_results/${DATASET}/chunk_${CHUNK}/log.txt"
done

# cloth_no_occlusion_front_4sec chunks
DATASET="cloth_no_occlusion_front_4sec"
CHUNKS=(15 21 22 23 27 28)
for CHUNK in "${CHUNKS[@]}"; do
    echo ""
    echo ">>> Running $DATASET chunk $CHUNK"
    pixi run -e deform-tracker python fabric_initialization_ablation.py \
        --dataset "$DATASET" \
        --chunk "$CHUNK" \
        --eval_frames "$EVAL_FRAMES" \
        2>&1 | tee -a "fabric_init_ablation_results/${DATASET}/chunk_${CHUNK}/log.txt"
done

echo ""
echo "=========================================="
echo "All fabric initialization ablation done!"
echo "Results in: fabric_init_ablation_results/"
echo "=========================================="

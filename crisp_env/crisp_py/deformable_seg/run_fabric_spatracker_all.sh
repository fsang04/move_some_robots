#!/bin/bash
#
# Run SpaTrackerV2 baseline on ALL fabric datasets
#
# Datasets and their chunks:
#   cloth_no_occlusion_back_3sec: 0, 3, 7, 12, 20
#   cloth_no_occlusion_back_4sec: 8, 13
#   cloth_no_occlusion_front_3sec: 2, 5, 6, 7, 11, 14, 17
#   cloth_no_occlusion_front_4sec: 15, 21, 22, 23, 27, 28
#
# Usage:
#   ./run_fabric_spatracker_all.sh
#   ./run_fabric_spatracker_all.sh 2>&1 | tee fabric_spatracker_all.log

set -e

cd /home/roahmlab/move_some_robots/crisp_env/crisp_py

echo "=========================================="
echo "SpaTrackerV2 Fabric Tracking - All Datasets"
echo "=========================================="

# cloth_no_occlusion_back_3sec: chunks 0, 3, 7, 12, 20
echo ""
echo "=== Dataset: cloth_no_occlusion_back_3sec ==="
for CHUNK in 0 3 7 12 20; do
    echo ""
    echo "--- Processing chunk $CHUNK ---"
    pixi run python deformable_seg/fabric_spatracker.py \
        --dataset cloth_no_occlusion_back_3sec \
        --chunk $CHUNK \
        --clip_seconds 10 \
        --no-mask \
        || echo "WARNING: chunk $CHUNK failed, continuing..."
done

# cloth_no_occlusion_back_4sec: chunks 8, 13
echo ""
echo "=== Dataset: cloth_no_occlusion_back_4sec ==="
for CHUNK in 8 13; do
    echo ""
    echo "--- Processing chunk $CHUNK ---"
    pixi run python deformable_seg/fabric_spatracker.py \
        --dataset cloth_no_occlusion_back_4sec \
        --chunk $CHUNK \
        --clip_seconds 10 \
        --no-mask \
        || echo "WARNING: chunk $CHUNK failed, continuing..."
done

# cloth_no_occlusion_front_3sec: chunks 2, 5, 6, 7, 11, 14, 17
echo ""
echo "=== Dataset: cloth_no_occlusion_front_3sec ==="
for CHUNK in 2 5 6 7 11 14 17; do
    echo ""
    echo "--- Processing chunk $CHUNK ---"
    pixi run python deformable_seg/fabric_spatracker.py \
        --dataset cloth_no_occlusion_front_3sec \
        --chunk $CHUNK \
        --clip_seconds 10 \
        --no-mask \
        || echo "WARNING: chunk $CHUNK failed, continuing..."
done

# cloth_no_occlusion_front_4sec: chunks 15, 21, 22, 23, 27, 28
echo ""
echo "=== Dataset: cloth_no_occlusion_front_4sec ==="
for CHUNK in 15 21 22 23 27 28; do
    echo ""
    echo "--- Processing chunk $CHUNK ---"
    pixi run python deformable_seg/fabric_spatracker.py \
        --dataset cloth_no_occlusion_front_4sec \
        --chunk $CHUNK \
        --clip_seconds 10 \
        --no-mask \
        || echo "WARNING: chunk $CHUNK failed, continuing..."
done

echo ""
echo "=========================================="
echo "All datasets processed!"
echo "=========================================="
echo ""
echo "Results saved to: deformable_seg/fabric_spatracker_results/"
echo ""
echo "To aggregate results, run:"
echo "  pixi run python deformable_seg/aggregate_fabric_spatracker.py"

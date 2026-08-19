#!/bin/bash
# Run Fabric Tracking Projection Ablation on all datasets and chunks
# Compares Full (with interior projection) vs NoProj (no interior projection), CPD disabled in both

set -e  # Exit on error

# Configuration
CLIP_SECONDS=10
GRID_ROWS=6
GRID_COLS=6

echo "=============================================="
echo "FABRIC TRACKING PROJECTION ABLATION - ALL DATA"
echo "=============================================="
echo "Clip duration: ${CLIP_SECONDS}s"
echo "Grid: ${GRID_ROWS}x${GRID_COLS}"
echo "=============================================="

# ============================================
# cloth_no_occlusion_back_3sec
# Chunks: 0, 3, 7, 12, 20
# ============================================
DATASET="cloth_no_occlusion_back_3sec"
echo ""
echo "=============================================="
echo "DATASET: $DATASET"
echo "=============================================="

for chunk in 0 3 7 12 20; do
    echo ""
    echo "--- $DATASET Chunk $chunk ---"
    python fabric_tracking_ablation.py --dataset $DATASET --chunk $chunk --clip_seconds $CLIP_SECONDS --grid_rows $GRID_ROWS --grid_cols $GRID_COLS || {
        echo "Warning: $DATASET chunk $chunk failed, continuing..."
        continue
    }
done

# ============================================
# cloth_no_occlusion_back_4sec
# Chunks: 8, 13
# ============================================
DATASET="cloth_no_occlusion_back_4sec"
echo ""
echo "=============================================="
echo "DATASET: $DATASET"
echo "=============================================="

for chunk in 8 13; do
    echo ""
    echo "--- $DATASET Chunk $chunk ---"
    python fabric_tracking_ablation.py --dataset $DATASET --chunk $chunk --clip_seconds $CLIP_SECONDS --grid_rows $GRID_ROWS --grid_cols $GRID_COLS || {
        echo "Warning: $DATASET chunk $chunk failed, continuing..."
        continue
    }
done

# ============================================
# cloth_no_occlusion_front_3sec
# Chunks: 2, 5, 6, 7, 11, 14, 17
# ============================================
DATASET="cloth_no_occlusion_front_3sec"
echo ""
echo "=============================================="
echo "DATASET: $DATASET"
echo "=============================================="

for chunk in 2 5 6 7 11 14 17; do
    echo ""
    echo "--- $DATASET Chunk $chunk ---"
    python fabric_tracking_ablation.py --dataset $DATASET --chunk $chunk --clip_seconds $CLIP_SECONDS --grid_rows $GRID_ROWS --grid_cols $GRID_COLS || {
        echo "Warning: $DATASET chunk $chunk failed, continuing..."
        continue
    }
done

# ============================================
# cloth_no_occlusion_front_4sec
# Chunks: 15, 21, 22, 23, 27, 28
# ============================================
DATASET="cloth_no_occlusion_front_4sec"
echo ""
echo "=============================================="
echo "DATASET: $DATASET"
echo "=============================================="

for chunk in 15 21 22 23 27 28; do
    echo ""
    echo "--- $DATASET Chunk $chunk ---"
    python fabric_tracking_ablation.py --dataset $DATASET --chunk $chunk --clip_seconds $CLIP_SECONDS --grid_rows $GRID_ROWS --grid_cols $GRID_COLS || {
        echo "Warning: $DATASET chunk $chunk failed, continuing..."
        continue
    }
done

echo ""
echo "=============================================="
echo "ALL FABRIC ABLATIONS COMPLETE!"
echo "=============================================="
echo ""
echo "Output directory: ./fabric_tracking_ablation_results/"
echo ""
echo "Datasets processed:"
echo "  - cloth_no_occlusion_back_3sec: chunks 0, 3, 7, 12, 20"
echo "  - cloth_no_occlusion_back_4sec: chunks 8, 13"
echo "  - cloth_no_occlusion_front_3sec: chunks 2, 5, 6, 7, 11, 14, 17"
echo "  - cloth_no_occlusion_front_4sec: chunks 15, 21, 22, 23, 27, 28"

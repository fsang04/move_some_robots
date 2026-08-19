#!/bin/bash
# Run DLO and BDLO tracking projection ablation on all chunks
# Compares Full (with projection) vs NoProj (no projection), CPD disabled in both

set -e  # Exit on error

# Configuration
CLIP_SECONDS=10
N_KEYPOINTS_DLO=14
N_KEYPOINTS_BDLO=25
KEYPOINTS_PER_SEGMENT="4 4 3 3 5"  # [ee0, ee1, free0, free1, trunk] -> 2+4+19=25 keypoints

# Different chunk ranges for each experiment type
DLO_START=${1:-0}
DLO_END=${2:-19}
BDLO_START=${3:-0}
BDLO_END=${4:-44}
BDLO_FASTER_START=${5:-0}
BDLO_FASTER_END=${6:-19}

echo "=============================================="
echo "DLO & BDLO Tracking Projection Ablation"
echo "=============================================="
echo "DLO chunks: $DLO_START to $DLO_END"
echo "BDLO chunks: $BDLO_START to $BDLO_END"
echo "BDLO Faster chunks: $BDLO_FASTER_START to $BDLO_FASTER_END"
echo "Clip duration: ${CLIP_SECONDS}s"
echo "DLO keypoints: $N_KEYPOINTS_DLO"
echo "BDLO keypoints: $N_KEYPOINTS_BDLO"
echo "BDLO keypoints_per_segment: $KEYPOINTS_PER_SEGMENT"
echo "=============================================="

# ============================================
# DLO Tracking Ablation
# ============================================
echo ""
echo "=============================================="
echo "DLO TRACKING ABLATION (chunks $DLO_START-$DLO_END)"
echo "=============================================="

for chunk in $(seq $DLO_START $DLO_END); do
    echo ""
    echo "--- DLO Chunk $chunk ---"
    python dlo_tracking_ablation.py --chunk $chunk --clip_seconds 15 --n_keypoints $N_KEYPOINTS_DLO || {
        echo "Warning: DLO chunk $chunk failed, continuing..."
        continue
    }
done

echo ""
echo "DLO tracking ablation complete!"

# ============================================
# BDLO Tracking Ablation
# ============================================
echo ""
echo "=============================================="
echo "BDLO TRACKING ABLATION (chunks $BDLO_START-$BDLO_END)"
echo "=============================================="

for chunk in $(seq $BDLO_START $BDLO_END); do
    echo ""
    echo "--- BDLO Chunk $chunk ---"
    python bdlo_tracking_ablation.py --chunk $chunk --clip_seconds $CLIP_SECONDS --n_keypoints $N_KEYPOINTS_BDLO --keypoints_per_segment $KEYPOINTS_PER_SEGMENT || {
        echo "Warning: BDLO chunk $chunk failed, continuing..."
        continue
    }
done

echo ""
echo "BDLO tracking ablation complete!"

# ============================================
# BDLO Faster Tracking Ablation
# ============================================
echo ""
echo "=============================================="
echo "BDLO FASTER TRACKING ABLATION (chunks $BDLO_FASTER_START-$BDLO_FASTER_END)"
echo "=============================================="

for chunk in $(seq $BDLO_FASTER_START $BDLO_FASTER_END); do
    echo ""
    echo "--- BDLO Faster Chunk $chunk ---"
    python bdlo_faster_tracking_ablation.py --chunk $chunk --clip_seconds $CLIP_SECONDS --n_keypoints $N_KEYPOINTS_BDLO --keypoints_per_segment $KEYPOINTS_PER_SEGMENT || {
        echo "Warning: BDLO Faster chunk $chunk failed, continuing..."
        continue
    }
done

echo ""
echo "=============================================="
echo "ALL TRACKING ABLATIONS COMPLETE!"
echo "=============================================="
echo ""
echo "Output directories:"
echo "  - ./dlo_tracking_ablation_results/"
echo "  - ./bdlo_tracking_ablation_results/"
echo "  - ./bdlo_faster_tracking_ablation_results/"
echo ""
echo "To aggregate results:"
echo "  python aggregate_dlo_results.py --input_dir dlo_tracking_ablation_results"
echo "  python aggregate_bdlo_results.py --input_dir bdlo_tracking_ablation_results"

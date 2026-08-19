#!/bin/bash
# Run wire initialization with and without repulsion for all trajectories
# Usage: bash run_wire_init_ablation.sh

set -e  # Exit on error

echo "=============================================="
echo "Wire Initialization Ablation Study"
echo "=============================================="

# Configuration
PIPELINE="streamlined"
METHOD="fps"

# Trajectory-specific keypoints_per_segment
# Format: [ee0, ee1, free0, free1, trunk]
TRAJ1_KPS="3 3 3 2 4"
TRAJ2_KPS="3 3 2 3 4"
TRAJ3_KPS="3 3 3 2 4"


# ============================================
# Trajectory 1
# ============================================
echo ""
echo "=============================================="
echo "Trajectory 1 - WITH repulsion"
echo "=============================================="
python wire_init_main.py --traj 1 --pipeline $PIPELINE --method $METHOD \
    --keypoints_per_segment $TRAJ1_KPS --min_mst_pixels 850

echo ""
echo "=============================================="
echo "Trajectory 1 - WITHOUT repulsion"
echo "=============================================="
python wire_init_main.py --traj 1 --pipeline $PIPELINE --method $METHOD \
    --keypoints_per_segment $TRAJ1_KPS --no_repulsion --min_mst_pixels 850


# ============================================
# Trajectory 2
# ============================================
echo ""
echo "=============================================="
echo "Trajectory 2 - WITH repulsion"
echo "=============================================="
python wire_init_main.py --traj 2 --pipeline $PIPELINE --method $METHOD \
    --keypoints_per_segment $TRAJ2_KPS --min_mst_pixels 960

echo ""
echo "=============================================="
echo "Trajectory 2 - WITHOUT repulsion"
echo "=============================================="
python wire_init_main.py --traj 2 --pipeline $PIPELINE --method $METHOD \
    --keypoints_per_segment $TRAJ2_KPS --no_repulsion --min_mst_pixels 960

# ============================================
# Trajectory 3
# ============================================
echo ""
echo "=============================================="
echo "Trajectory 3 - WITH repulsion"
echo "=============================================="
python wire_init_main.py --traj 3 --pipeline $PIPELINE --method $METHOD \
    --keypoints_per_segment $TRAJ3_KPS --min_mst_pixels 800

echo ""
echo "=============================================="
echo "Trajectory 3 - WITHOUT repulsion"
echo "=============================================="
python wire_init_main.py --traj 3 --pipeline $PIPELINE --method $METHOD \
    --keypoints_per_segment $TRAJ3_KPS --no_repulsion --min_mst_pixels 800

# ============================================
# Summary
# ============================================
echo ""
echo "=============================================="
echo "ABLATION STUDY COMPLETE"
echo "=============================================="
echo "Output directories:"
echo "  - data/arm_traj2/wire_init_output_${PIPELINE}_${METHOD}/"
echo "  - data/arm_traj2/wire_init_output_${PIPELINE}_${METHOD}_no_repulsion/"
echo "  - data/arm_traj3/wire_init_output_${PIPELINE}_${METHOD}/"
echo "  - data/arm_traj3/wire_init_output_${PIPELINE}_${METHOD}_no_repulsion/"
echo ""
echo "Compare edge_length_error_metrics.json in each directory to see the effect of repulsion."

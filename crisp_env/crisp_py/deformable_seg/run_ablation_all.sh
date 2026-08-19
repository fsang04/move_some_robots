#!/bin/bash
# Run ablation study on all trajectories (all frames)

echo "=============================================="
echo "Running Ablation Study on All Trajectories"
echo "=============================================="

# Trajectory 1
echo ""
echo "=============================================="
echo "TRAJECTORY 1 - All Frames"
echo "=============================================="
python wire_tracker_ablation.py --trajectory traj1

# Trajectory 2
echo ""
echo "=============================================="
echo "TRAJECTORY 2 - All Frames"
echo "=============================================="
python wire_tracker_ablation.py --trajectory traj2

# Trajectory 3
echo ""
echo "=============================================="
echo "TRAJECTORY 3 - All Frames"
echo "=============================================="
python wire_tracker_ablation.py --trajectory traj3

echo ""
echo "=============================================="
echo "All ablation studies complete!"
echo "=============================================="
echo "Output directories:"
echo "  - ./data/arm_traj1/ablation_output/"
echo "  - ./data/arm_traj2/ablation_output/"
echo "  - ./data/arm_traj3/ablation_output/"

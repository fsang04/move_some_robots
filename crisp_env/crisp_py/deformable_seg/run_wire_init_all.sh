#!/bin/bash
# Run wire initialization for all trajectories with both FPS and GMM methods

set -e  # Exit on error

echo "=========================================="
echo "Wire Initialization - All Trajectories"
echo "=========================================="

cd /home/yehengz/deformable_seg

# Trajectory 1
echo ""
echo "========== Trajectory 1 - FPS =========="
python wire_init_main.py --traj 1 --method fps --min_mst_pixels 800

echo ""
echo "========== Trajectory 1 - GMM =========="
python wire_init_main.py --traj 1 --method gmm --min_mst_pixels 800

# Trajectory 2
echo ""
echo "========== Trajectory 2 - FPS =========="
python wire_init_main.py --traj 2 --method fps

echo ""
echo "========== Trajectory 2 - GMM =========="
python wire_init_main.py --traj 2 --method gmm

# Trajectory 3
echo ""
echo "========== Trajectory 3 - FPS =========="
python wire_init_main.py --traj 3 --method fps --min_mst_pixels 800

echo ""
echo "========== Trajectory 3 - GMM =========="
python wire_init_main.py --traj 3 --method gmm --min_mst_pixels 800

echo ""
echo "=========================================="
echo "All runs complete!"
echo "=========================================="
echo ""
echo "Output directories:"
echo "  Traj 1 FPS: ./data/arm_traj1/wire_init_output_fps"
echo "  Traj 1 GMM: ./data/arm_traj1/wire_init_output_gmm"
echo "  Traj 2 FPS: ./data/arm_traj2/wire_init_output_fps"
echo "  Traj 2 GMM: ./data/arm_traj2/wire_init_output_gmm"
echo "  Traj 3 FPS: ./data/arm_traj3/wire_init_output_fps"
echo "  Traj 3 GMM: ./data/arm_traj3/wire_init_output_gmm"

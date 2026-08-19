#!/bin/bash
# Run wire tracking and post-processing for traj1, traj2 and traj3
# Usage: screen -S tracking bash run_tracking_traj2_traj3.sh

cd /home/yehengz/deformable_seg

echo "=============================================="
echo "Starting Wire Tracking Pipeline"
echo "=============================================="

# Activate conda environment
source ~/miniconda3/bin/activate
conda activate unidepth

# Run traj2
echo ""
echo "=============================================="
echo "Running Trajectory 2 - Tracking"
echo "=============================================="
python wire_tracking_main.py --traj 2

echo ""
echo "=============================================="
echo "Running Trajectory 2 - Post Processing (sigma=2)"
echo "=============================================="
python wire_tracking_post_processing.py --traj 2 --sigma 2

echo ""
echo "=============================================="
echo "Running Trajectory 2 - Post Processing (sigma=5)"
echo "=============================================="
python wire_tracking_post_processing.py --traj 2 --sigma 5

# Run traj3
echo ""
echo "=============================================="
echo "Running Trajectory 3 - Tracking"
echo "=============================================="
python wire_tracking_main.py --traj 3

echo ""
echo "=============================================="
echo "Running Trajectory 3 - Post Processing (sigma=2)"
echo "=============================================="
python wire_tracking_post_processing.py --traj 3 --sigma 2

echo ""
echo "=============================================="
echo "Running Trajectory 3 - Post Processing (sigma=5)"
echo "=============================================="
python wire_tracking_post_processing.py --traj 3 --sigma 5


# Run traj1 post-processing only (tracking already done)
echo ""
echo "=============================================="
echo "Running Trajectory 1 - Post Processing (sigma=5)"
echo "=============================================="
python wire_tracking_post_processing.py --traj 1 --sigma 5

echo ""
echo "=============================================="
echo "All Done!"
echo "=============================================="

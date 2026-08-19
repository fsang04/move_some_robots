#!/bin/bash
# Run post-processing for all trajectories with sigma=3

echo "========================================"
echo "Running post-processing for all trajectories"
echo "Sigma: 3"
echo "========================================"

echo ""
echo "========================================"
echo "Trajectory 1"
echo "========================================"
python wire_tracking_post_processing.py --traj 1 --sigma 3

echo ""
echo "========================================"
echo "Trajectory 2"
echo "========================================"
python wire_tracking_post_processing.py --traj 2 --sigma 3

echo ""
echo "========================================"
echo "Trajectory 3"
echo "========================================"
python wire_tracking_post_processing.py --traj 3 --sigma 3

echo ""
echo "========================================"
echo "All post-processing complete!"
echo "========================================"

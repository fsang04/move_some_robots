#!/bin/bash
# Run CDCPD benchmark on all trajectories
# Usage: ./run_cdcpd_benchmark.sh

set -e

echo "========================================================================"
echo "CDCPD vs WireTracker BENCHMARK - ALL TRAJECTORIES"
echo "========================================================================"
echo "Start time: $(date)"
echo ""

cd /home/yehengz/deformable_seg

for traj in traj1 traj2 traj3; do
    echo ""
    echo ">>> Running CDCPD benchmark on $traj"
    echo "------------------------------------------------------------------------"
    python wire_tracking_cdcpd_benchmark.py --trajectory $traj
    echo ">>> Completed $traj"
    echo ""
done

echo "========================================================================"
echo "ALL BENCHMARKS COMPLETED"
echo "========================================================================"
echo "End time: $(date)"
echo ""
echo "Output locations:"
echo "  - traj1: ./data/arm_traj1/cdcpd_benchmark/"
echo "  - traj2: ./data/arm_traj2/cdcpd_benchmark/"
echo "  - traj3: ./data/arm_traj3/cdcpd_benchmark/"
echo ""
echo "Done!"

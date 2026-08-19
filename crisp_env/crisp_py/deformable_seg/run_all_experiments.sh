#!/bin/bash
# Run all wire tracking experiments and ablation studies
# Usage: ./run_all_experiments.sh

set -e  # Exit on error

echo "========================================================================"
echo "WIRE TRACKING EXPERIMENTS - ALL TRAJECTORIES"
echo "========================================================================"
echo "Start time: $(date)"
echo ""

cd /home/yehengz/deformable_seg

# ============================================================================
# MAIN TRACKING (wire_tracking_main.py)
# ============================================================================

echo "========================================================================"
echo "RUNNING MAIN TRACKING"
echo "========================================================================"

for traj in 1 2 3; do
    echo ""
    echo ">>> Running wire_tracking_main.py --traj $traj"
    echo "------------------------------------------------------------------------"
    python wire_tracking_main.py --traj $traj
    echo ">>> Completed traj $traj"
    echo ""
done

# ============================================================================
# ABLATION STUDIES (wire_tracker_ablation.py)
# ============================================================================

echo "========================================================================"
echo "RUNNING ABLATION STUDIES"
echo "========================================================================"

for traj in traj1 traj2 traj3; do
    echo ""
    echo ">>> Running wire_tracker_ablation.py --trajectory $traj"
    echo "------------------------------------------------------------------------"
    python wire_tracker_ablation.py --trajectory $traj
    echo ">>> Completed ablation $traj"
    echo ""
done

# ============================================================================
# SUMMARY
# ============================================================================

echo "========================================================================"
echo "ALL EXPERIMENTS COMPLETED"
echo "========================================================================"
echo "End time: $(date)"
echo ""
echo "Output locations:"
echo "  - Main tracking: ./wire_output/"
echo "  - Ablation traj1: ./data/arm_traj1/ablation_output/"
echo "  - Ablation traj2: ./data/arm_traj2/ablation_output/"
echo "  - Ablation traj3: ./data/arm_traj3/ablation_output/"
echo ""
echo "Done!"

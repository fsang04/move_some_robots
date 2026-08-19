#!/bin/bash
# Run all fabric tracking experiments
# This script runs the main tracker, CDCPD benchmark, and ablation study

set -e  # Exit on error

cd /home/yehengz/deformable_seg

echo "========================================"
echo "Running Fabric Tracking Experiments"
echo "========================================"
echo ""

# Run main fabric tracking
echo "----------------------------------------"
echo "[1/4] Running Fabric Tracking Main..."
echo "----------------------------------------"
python fabric_tracking_main.py
echo ""

# # Run CDCPD benchmark
# echo "----------------------------------------"
# echo "[2/4] Running CDCPD Benchmark..."
# echo "----------------------------------------"
# python fabric_tracking_cdcpd_benchmark.py
# echo ""

# # Run ablation study
# echo "----------------------------------------"
# echo "[3/4] Running Ablation Study..."
# echo "----------------------------------------"
# python fabric_tracker_ablation.py
# echo ""

# Run post-processing with sigma=3.0
echo "----------------------------------------"
echo "[4/4] Running Post-Processing (sigma=3.0)..."
echo "----------------------------------------"
python fabric_tracking_post_processing.py --sigma 3.0
echo ""

echo "========================================"
echo "All fabric tracking experiments complete!"
echo "========================================"
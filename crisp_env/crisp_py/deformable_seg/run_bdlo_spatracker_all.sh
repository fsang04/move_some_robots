#!/bin/bash
# Run SpaTrackerV2 on all BDLO datasets
# Usage: ./run_bdlo_spatracker_all.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

echo "=========================================="
echo "BDLO SpaTrackerV2 - All Chunks"
echo "=========================================="

# 4sec dataset: chunks 0-44
echo ""
echo "=== Dataset: 4sec (bdlo_no_contact_4sec) ==="
echo "Chunks: 0-44"
echo ""

for chunk in $(seq 0 44); do
    echo "----------------------------------------"
    echo "Processing 4sec chunk $chunk / 44"
    echo "----------------------------------------"
    pixi run -e spatracker python deformable_seg/bdlo_spatracker.py --dataset 4sec --chunk $chunk
done

# 2sec dataset: chunks 0-19 (note: config says 20 chunks, so 0-19)
echo ""
echo "=== Dataset: 2sec (bdlo_no_contact_2sec) ==="
echo "Chunks: 0-19"
echo ""

for chunk in $(seq 0 19); do
    echo "----------------------------------------"
    echo "Processing 2sec chunk $chunk / 19"
    echo "----------------------------------------"
    pixi run -e spatracker python deformable_seg/bdlo_spatracker.py --dataset 2sec --chunk $chunk
done

echo ""
echo "=========================================="
echo "DONE - All BDLO chunks processed"
echo "=========================================="

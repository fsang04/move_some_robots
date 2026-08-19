#!/bin/bash
# Evaluate CoTracker results on all BDLO datasets
# Run from: /home/roahmlab/move_some_robots/crisp_env/crisp_py/deformable_seg/

set -e

# Mode (offline or online)
MODE=${1:-offline}

echo "=========================================="
echo "Evaluating CoTracker results on BDLO"
echo "Mode: $MODE"
echo "=========================================="

# bdlo1 chunks (0-44, some may not exist)
echo ""
echo "=== Processing bdlo1 ==="
for chunk in {0..44}; do
    chunk_dir="/home/roahmlab/move_some_robots/crisp_env/crisp_py/deformable_seg/bdlo1_cotracker_results/chunk_$chunk"
    if [ -d "$chunk_dir" ]; then
        echo ""
        echo ">>> Evaluating bdlo1 chunk_$chunk ($MODE)"
        python evaluate_bdlo_cotracker.py --dataset bdlo1 --chunk "$chunk" --mode "$MODE" || echo "  [SKIP] Failed or missing data"
    fi
done

# bdlo1_faster chunks (0-19)
echo ""
echo "=== Processing bdlo1_faster ==="
for chunk in {0..19}; do
    chunk_dir="/home/roahmlab/move_some_robots/crisp_env/crisp_py/deformable_seg/bdlo1_faster_cotracker_results/chunk_$chunk"
    if [ -d "$chunk_dir" ]; then
        echo ""
        echo ">>> Evaluating bdlo1_faster chunk_$chunk ($MODE)"
        python evaluate_bdlo_cotracker.py --dataset bdlo1_faster --chunk "$chunk" --mode "$MODE" || echo "  [SKIP] Failed or missing data"
    fi
done

echo ""
echo "=========================================="
echo "All BDLO evaluations complete!"
echo ""
echo "To aggregate results, run:"
echo "  python evaluate_bdlo_cotracker.py --dataset all --mode $MODE"
echo "=========================================="

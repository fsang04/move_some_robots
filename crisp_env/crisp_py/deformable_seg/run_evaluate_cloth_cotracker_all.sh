#!/bin/bash
# Evaluate CoTracker results on all cloth datasets
# Run from: /home/roahmlab/move_some_robots/crisp_env/crisp_py/deformable_seg/

set -e

# Datasets and chunks (same as run_cloth_cotracker_all.sh)
declare -A CHUNKS
CHUNKS["cloth_no_occlusion_back_3sec"]="0 3 7 12 20"
CHUNKS["cloth_no_occlusion_back_4sec"]="8 13"
CHUNKS["cloth_no_occlusion_front_3sec"]="2 5 6 7 11 14 17"
CHUNKS["cloth_no_occlusion_front_4sec"]="15 21 22 23 27 28"

# Mode (offline or online)
MODE=${1:-offline}

echo "=========================================="
echo "Evaluating CoTracker results on cloth"
echo "Mode: $MODE"
echo "=========================================="

for dataset in "${!CHUNKS[@]}"; do
    echo ""
    echo "Dataset: $dataset"
    echo "Chunks: ${CHUNKS[$dataset]}"
    
    for chunk in ${CHUNKS[$dataset]}; do
        echo ""
        echo ">>> Evaluating $dataset chunk_$chunk ($MODE)"
        python evaluate_cloth_cotracker.py --dataset "$dataset" --chunk "$chunk" --mode "$MODE"
    done
done

echo ""
echo "=========================================="
echo "All evaluations complete!"
echo "=========================================="

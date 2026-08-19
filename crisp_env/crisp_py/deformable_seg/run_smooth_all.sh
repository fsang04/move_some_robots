#!/bin/bash
# Run smooth_bdlo_keypoints.py on all clip directories under bdlo1_faster_free_ee_evaluation_results

BASE_DIR="$(cd "$(dirname "$0")" && pwd)/bdlo1_faster_free_ee_evaluation_results"
SMOOTH_SCRIPT="$(cd "$(dirname "$0")" && pwd)/smooth_bdlo_keypoints.py"
VIS_SCRIPT="$(cd "$(dirname "$0")" && pwd)/visualize_smoothed.py"

for clip_dir in "$BASE_DIR"/chunk_*/clip_*; do
    if [ -f "$clip_dir/3d_keypoints.npz" ]; then
        echo "Processing: $clip_dir"
        python "$SMOOTH_SCRIPT" --input_dir "$clip_dir"
        # python "$VIS_SCRIPT" --input_dir "$clip_dir"
    else
        echo "Skipping (no 3d_keypoints.npz): $clip_dir"
    fi
done

echo "All done!"
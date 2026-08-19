#!/bin/bash
# Run smooth_fabric_keypoints.py on all clip directories under fabric_evaluation_results

BASE_DIR="$(cd "$(dirname "$0")" && pwd)/fabric_evaluation_results"
SMOOTH_SCRIPT="$(cd "$(dirname "$0")" && pwd)/smooth_fabric_keypoints.py"

for clip_dir in "$BASE_DIR"/*/chunk_*/clip_*; do
    if [ -f "$clip_dir/3d_keypoints.npz" ]; then
        echo "Processing: $clip_dir"
        python "$SMOOTH_SCRIPT" --input_dir "$clip_dir"
    else
        echo "Skipping (no 3d_keypoints.npz): $clip_dir"
    fi
done

echo "All done!"

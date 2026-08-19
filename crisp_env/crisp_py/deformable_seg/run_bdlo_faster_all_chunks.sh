#!/bin/bash
# Run BDLO faster batch experiment on chunks 0-19, excluding corrupted clips
#
# Corrupted clips (chunk, clip):
#   chunk_0:  clip_1
#   chunk_6:  clip_0
#   chunk_11: clip_1
#   chunk_14: clip_0
#   chunk_15: clip_1
#   chunk_17: clip_0, clip_1
#   chunk_18: clip_1

set -e

cd /home/roahmlab/move_some_robots/crisp_env/crisp_py/deformable_seg

echo "=============================================="
echo "  BDLO Faster Batch Experiment (Chunks 0-19)"
echo "=============================================="
echo ""

# Define corrupted clips per chunk
declare -A SKIP_CLIPS
SKIP_CLIPS[0]="1"
SKIP_CLIPS[6]="0"
SKIP_CLIPS[11]="1"
SKIP_CLIPS[14]="0"
SKIP_CLIPS[15]="1"
SKIP_CLIPS[17]="0 1"
SKIP_CLIPS[18]="1"

# Run each chunk
for chunk in {0..19}; do
    echo ""
    echo "======================================================"
    echo "  Processing chunk $chunk"
    echo "======================================================"
    
    skip_arg=""
    if [[ -n "${SKIP_CLIPS[$chunk]}" ]]; then
        skip_arg="--skip_clips ${SKIP_CLIPS[$chunk]}"
        echo "  Skipping clips: ${SKIP_CLIPS[$chunk]}"
    fi
    
    pixi run -e deform-tracker python bdlo1_faster_batch_experiment.py \
        --chunk $chunk \
        --clip_seconds 10 \
        --n_keypoints 25 \
        --keypoints_per_segment 4 4 3 3 5 \
        $skip_arg
    
    echo "  Chunk $chunk complete!"
done

echo ""
echo "=============================================="
echo "  ALL CHUNKS COMPLETE!"
echo "=============================================="
echo "Results saved to: bdlo1_faster_free_ee_evaluation_results/"

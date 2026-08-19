#!/bin/bash
# Run DLO SpaTracker baseline on all chunks (0-19)

# Chunks available in dlo1_evaluation_results
CHUNKS=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19)

# Default clip duration is 15s (450 frames at 30fps)
CLIP_SECONDS=${CLIP_SECONDS:-15}
SEGMENT_SECONDS=${SEGMENT_SECONDS:-0}

echo "===== DLO SpaTracker Baseline ====="
echo "Chunks: ${CHUNKS[@]}"
echo "Clip duration: ${CLIP_SECONDS}s"
echo "Segment duration: ${SEGMENT_SECONDS}s"
echo "===================================="

for chunk in "${CHUNKS[@]}"; do
    echo ""
    echo "====== Processing chunk_${chunk} ======"
    
    # Check if evaluation results exist
    if [ ! -d "dlo1_evaluation_results/chunk_${chunk}" ]; then
        echo "Skipping chunk_${chunk}: no evaluation results"
        continue
    fi
    
    python dlo_spatracker.py --chunk ${chunk} \
        --clip_seconds ${CLIP_SECONDS} \
        --segment_seconds ${SEGMENT_SECONDS} \
    
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed on chunk_${chunk}"
    fi
done

echo ""
echo "===== Done ====="
echo "Run: python aggregate_dlo_spatracker.py"

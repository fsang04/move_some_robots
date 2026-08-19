#!/bin/bash
# Run DLO CoTracker evaluation on all chunks

# Evaluate each chunk (0-19)
for chunk in {0..19}; do
    echo "=============================================="
    echo "Evaluating DLO chunk $chunk"
    echo "=============================================="
    python evaluate_dlo_cotracker.py --chunk $chunk --mode offline
done

# Aggregate all results
echo "=============================================="
echo "Aggregating all results"
echo "=============================================="
python evaluate_dlo_cotracker.py --all --mode offline

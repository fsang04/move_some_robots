#!/usr/bin/env python3
"""Aggregate BDLO SpaTracker evaluation results, excluding corrupted data."""

import csv
import numpy as np
from pathlib import Path

# Base directories for BDLO SpaTracker results
BASE_DIRS = [
    Path(__file__).parent / 'bdlo1_spatracker_results',
    Path(__file__).parent / 'bdlo1_faster_spatracker_results',
]

# Corrupted clips to exclude (in bdlo1_faster_* only)
CORRUPTED_FASTER = {
    ('chunk_0', 'clip_0'),
    ('chunk_0', 'clip_1'),
    ('chunk_6', 'clip_0'),
    ('chunk_11', 'clip_0'),
    ('chunk_11', 'clip_1'),
    ('chunk_14', 'clip_0'),
    ('chunk_15', 'clip_1'),
    ('chunk_17', 'clip_0'),
    ('chunk_17', 'clip_1'),
    ('chunk_18', 'clip_1'),
}

def collect_all_frames(base_dirs):
    """Collect all per_frame.csv data from multiple base directories."""
    all_rows = []
    skipped = []
    
    for base_dir in base_dirs:
        base_path = Path(base_dir)
        is_faster = 'faster' in str(base_dir)
        
        if not base_path.exists():
            print(f"Warning: {base_dir} does not exist")
            continue
            
        chunk_dirs = sorted([d for d in base_path.iterdir() if d.is_dir() and d.name.startswith('chunk_')],
                           key=lambda x: int(x.name.split('_')[1]))
        
        for chunk_dir in chunk_dirs:
            clip_dirs = sorted([d for d in chunk_dir.iterdir() if d.is_dir() and d.name.startswith('clip_')],
                              key=lambda x: int(x.name.split('_')[1]))
            
            for clip_dir in clip_dirs:
                # Skip corrupted clips in faster folder
                if is_faster and (chunk_dir.name, clip_dir.name) in CORRUPTED_FASTER:
                    skipped.append(f"{base_dir.name}/{chunk_dir.name}/{clip_dir.name}")
                    continue
                    
                csv_path = clip_dir / 'per_frame.csv'
                if csv_path.exists():
                    with open(csv_path, 'r') as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            row['_source'] = str(base_dir.name)
                            row['_chunk'] = chunk_dir.name
                            row['_clip'] = clip_dir.name
                            all_rows.append(row)
    
    return all_rows, skipped

def compute_stats(rows):
    """Compute statistics for SpaTracker results."""
    # SpaTracker columns: EdgePctMean, EdgeRMSE_mm, CD_mm, F_10mm
    metrics = {
        'EdgePctMean': 'EdgePctMean',
        'EdgeRMSE': 'EdgeRMSE_mm',
        'CD': 'CD_mm',
        'F@10mm': 'F_10mm',
    }
    
    results = {}
    for display_name, col_name in metrics.items():
        values = [float(r[col_name]) for r in rows if col_name in r and r[col_name]]
        if values:
            arr = np.array(values)
            results[display_name] = {
                'mean': np.mean(arr),
                'std': np.std(arr),
                'count': len(arr)
            }
    
    return results

def main():
    print("Collecting frames from BDLO SpaTracker results (excluding corrupted data)...")
    all_rows, skipped = collect_all_frames(BASE_DIRS)
    
    if skipped:
        print(f"\nSkipped {len(skipped)} corrupted clips:")
        for s in skipped:
            print(f"  - {s}")
    
    clips = set((row['_source'], row['_chunk'], row['_clip']) for row in all_rows)
    chunks = set((row['_source'], row['_chunk']) for row in all_rows)
    
    results = compute_stats(all_rows)
    n_frames = len(all_rows)
    
    print(f"\n{'='*80}")
    print(f"BDLO SpaTracker RESULTS (excluding corrupted data)")
    print(f"Total frames: {n_frames}, Total clips: {len(clips)}, Total chunks: {len(chunks)}")
    print(f"{'='*80}")
    
    if results:
        edge_pct = f"{results['EdgePctMean']['mean']:5.2f} ± {results['EdgePctMean']['std']:5.2f}%"
        edge_rmse = f"{results['EdgeRMSE']['mean']:5.2f} ± {results['EdgeRMSE']['std']:5.2f} mm"
        cd = f"{results['CD']['mean']:5.2f} ± {results['CD']['std']:5.2f} mm"
        f10 = f"{results['F@10mm']['mean']:5.1f}%"
        
        print(f"{'Metric':<20} | {'Value':<25}")
        print("-"*50)
        print(f"{'Edge%':<20} | {edge_pct:<25}")
        print(f"{'EdgeRMSE (mm)':<20} | {edge_rmse:<25}")
        print(f"{'CD (mm)':<20} | {cd:<25}")
        print(f"{'F@10mm':<20} | {f10:<25}")
    else:
        print("No data found!")

if __name__ == '__main__':
    main()

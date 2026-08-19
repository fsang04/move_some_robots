#!/usr/bin/env python3
"""Aggregate DLO evaluation results across all chunks and clips."""

import csv
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).parent / 'dlo1_evaluation_results'

def collect_all_frames(base_dir):
    """Collect all per_frame.csv data from base directory."""
    all_rows = []
    base_path = Path(base_dir)
    
    if not base_path.exists():
        print(f"Warning: {base_dir} does not exist")
        return all_rows
        
    chunk_dirs = sorted([d for d in base_path.iterdir() if d.is_dir() and d.name.startswith('chunk_')],
                       key=lambda x: int(x.name.split('_')[1]))
    
    for chunk_dir in chunk_dirs:
        clip_dirs = sorted([d for d in chunk_dir.iterdir() if d.is_dir() and d.name.startswith('clip_')],
                          key=lambda x: int(x.name.split('_')[1]))
        
        for clip_dir in clip_dirs:
            csv_path = clip_dir / 'per_frame.csv'
            if csv_path.exists():
                with open(csv_path, 'r') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        row['_chunk'] = chunk_dir.name
                        row['_clip'] = clip_dir.name
                        all_rows.append(row)
    
    return all_rows

def compute_frame_weighted_stats(rows, methods=['Full', 'NoSnap', 'NoGeometry', 'CDCPD']):
    """Compute frame-weighted statistics for each method."""
    metrics = ['EdgePctMean', 'EdgeRMSE', 'Edge<5%', 'CD', 'F@10mm']
    
    results = {}
    for method in methods:
        method_rows = [r for r in rows if r['Method'] == method]
        if not method_rows:
            continue
            
        results[method] = {}
        for metric in metrics:
            values = [float(r[metric]) for r in method_rows if metric in r]
            if values:
                arr = np.array(values)
                results[method][metric] = {
                    'mean': np.mean(arr),
                    'std': np.std(arr),
                    'count': len(arr)
                }
    
    return results

def main():
    print("Collecting frames from DLO evaluation results...")
    all_rows = collect_all_frames(BASE_DIR)
    
    # Count unique frames per method
    method_counts = {}
    for row in all_rows:
        method = row['Method']
        method_counts[method] = method_counts.get(method, 0) + 1
    
    clips = set((row['_chunk'], row['_clip']) for row in all_rows)
    chunks = set(row['_chunk'] for row in all_rows)
    
    results = compute_frame_weighted_stats(all_rows)
    n_frames = method_counts.get('Full', 0)
    
    print(f"\n{'='*100}")
    print(f"DLO RESULTS")
    print(f"Total frames: {n_frames}, Total clips: {len(clips)}, Total chunks: {len(chunks)}")
    print(f"{'='*100}")
    print(f"{'Method':<12} | {'Edge%':<18} | {'EdgeRMSE (mm)':<18} | {'Edge<5%':<10} | {'CD (mm)':<18} | {'F@10mm':<10}")
    print("-"*100)
    
    for method in ['Full', 'NoSnap', 'NoGeometry', 'CDCPD']:
        if method in results:
            r = results[method]
            edge_pct = f"{r['EdgePctMean']['mean']:5.2f} ± {r['EdgePctMean']['std']:5.2f}%"
            edge_rmse = f"{r['EdgeRMSE']['mean']:5.2f} ± {r['EdgeRMSE']['std']:5.2f} mm"
            edge_5 = f"{r['Edge<5%']['mean']:5.1f}%"
            cd = f"{r['CD']['mean']:5.2f} ± {r['CD']['std']:5.2f} mm"
            f10 = f"{r['F@10mm']['mean']:5.1f}%"
            print(f"{method:<12} | {edge_pct:<18} | {edge_rmse:<18} | {edge_5:<10} | {cd:<18} | {f10:<10}")

if __name__ == '__main__':
    main()

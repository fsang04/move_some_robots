#!/usr/bin/env python3
"""Aggregate BDLO tracking ablation results across both folders, excluding corrupted data."""

import csv
import numpy as np
from pathlib import Path

# Base directories for BDLO tracking ablation results
BASE_DIRS = [
    Path(__file__).parent / 'bdlo_tracking_ablation_results',
    Path(__file__).parent / 'bdlo_faster_tracking_ablation_results',
]

# Corrupted clips to exclude (in bdlo_faster_tracking_ablation_results only)
# These correspond to corrupted data in bdlo1_faster_evaluation_results
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
    sources = {}
    
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
                    skipped.append(f"{base_path.name}/{chunk_dir.name}/{clip_dir.name}")
                    continue
                    
                csv_path = clip_dir / 'per_frame.csv'
                if csv_path.exists():
                    source_key = base_path.name
                    if source_key not in sources:
                        sources[source_key] = {'chunks': set(), 'clips': 0, 'frames': 0}
                    sources[source_key]['chunks'].add(chunk_dir.name)
                    sources[source_key]['clips'] += 1
                    
                    with open(csv_path, 'r') as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            row['_source'] = source_key
                            row['_chunk'] = chunk_dir.name
                            row['_clip'] = clip_dir.name
                            all_rows.append(row)
                            sources[source_key]['frames'] += 1
    
    return all_rows, skipped, sources


def compute_stats(rows, methods=['Full', 'NoProj']):
    """Compute frame-weighted statistics for each method."""
    results = {}
    
    for method in methods:
        method_rows = [r for r in rows if r.get('Method') == method]
        if not method_rows:
            continue
            
        results[method] = {'count': len(method_rows)}
        
        # Numeric metrics to aggregate
        metrics = ['EdgePctMean', 'EdgePctStd', 'EdgePctMax', 'EdgeRMSE', 
                   'PosRMSE', 'CD', 'Pred2Ref', 'Ref2Pred', 
                   'F@2mm', 'F@5mm', 'F@10mm']
        
        for metric in metrics:
            values = []
            for r in method_rows:
                if metric in r:
                    try:
                        values.append(float(r[metric]))
                    except (ValueError, TypeError):
                        pass
            
            if values:
                arr = np.array(values)
                results[method][metric] = {
                    'mean': np.mean(arr),
                    'std': np.std(arr),
                    'median': np.median(arr),
                    'min': np.min(arr),
                    'max': np.max(arr),
                }
        
        # Compute Edge<5% (percentage of frames where EdgePctMean < 5%)
        edge_pct_values = []
        for r in method_rows:
            if 'EdgePctMean' in r:
                try:
                    edge_pct_values.append(float(r['EdgePctMean']))
                except (ValueError, TypeError):
                    pass
        if edge_pct_values:
            arr = np.array(edge_pct_values)
            results[method]['Edge<5%'] = np.mean(arr < 5.0) * 100  # Percentage
    
    return results


def main():
    print("=" * 100)
    print("BDLO Tracking Ablation Results Aggregation")
    print("=" * 100)
    
    print("\nCollecting frames from BDLO tracking ablation results (excluding corrupted data)...")
    all_rows, skipped, sources = collect_all_frames(BASE_DIRS)
    
    if skipped:
        print(f"\nSkipped {len(skipped)} corrupted clips:")
        for s in skipped:
            print(f"  - {s}")
    
    print(f"\nData sources:")
    total_chunks = 0
    total_clips = 0
    total_frames = 0
    for source, info in sources.items():
        n_chunks = len(info['chunks'])
        print(f"  {source}: {n_chunks} chunks, {info['clips']} clips, {info['frames']} frames")
        total_chunks += n_chunks
        total_clips += info['clips']
        total_frames += info['frames']
    
    print(f"\nTotal: {total_chunks} chunks, {total_clips} clips, {total_frames} frames")
    
    # Get unique methods
    methods = sorted(set(r.get('Method', '') for r in all_rows if r.get('Method')))
    print(f"\nMethods found: {methods}")
    
    # Compute statistics
    results = compute_stats(all_rows, methods)
    
    # Print summary table
    print(f"\n{'='*140}")
    print(f"{'Method':<12} | {'Frames':>8} | {'EdgePct%':>12} | {'Edge<5%':>8} | {'EdgeRMSE':>12} | {'PosRMSE':>12} | {'CD':>12} | {'F@5mm':>10} | {'F@10mm':>10}")
    print(f"{'-'*140}")
    
    for method in methods:
        if method not in results:
            continue
        r = results[method]
        
        edge_pct = r.get('EdgePctMean', {})
        edge_under5 = r.get('Edge<5%', 0)
        edge_rmse = r.get('EdgeRMSE', {})
        pos_rmse = r.get('PosRMSE', {})
        cd = r.get('CD', {})
        f5 = r.get('F@5mm', {})
        f10 = r.get('F@10mm', {})
        
        print(f"{method:<12} | {r['count']:>8} | "
              f"{edge_pct.get('mean', 0):>5.2f}±{edge_pct.get('std', 0):>5.2f} | "
              f"{edge_under5:>7.1f}% | "
              f"{edge_rmse.get('mean', 0):>5.2f}±{edge_rmse.get('std', 0):>5.2f} | "
              f"{pos_rmse.get('mean', 0):>5.2f}±{pos_rmse.get('std', 0):>5.2f} | "
              f"{cd.get('mean', 0):>5.2f}±{cd.get('std', 0):>5.2f} | "
              f"{f5.get('mean', 0):>9.2f}% | "
              f"{f10.get('mean', 0):>9.2f}%")
    
    print(f"{'='*140}")
    
    # Detailed stats
    print(f"\nDetailed Statistics (mean ± std):")
    print(f"{'-'*100}")
    
    for method in methods:
        if method not in results:
            continue
        r = results[method]
        print(f"\n  {method} ({r['count']} frames):")
        
        # Edge<5% 
        edge_under5 = r.get('Edge<5%', 0)
        print(f"    {'Edge<5%':<14}: {edge_under5:>8.1f}%")
        
        for metric in ['EdgePctMean', 'EdgeRMSE', 'PosRMSE', 'CD', 'F@2mm', 'F@5mm', 'F@10mm']:
            if metric in r:
                m = r[metric]
                unit = "%" if "Pct" in metric or "F@" in metric else "mm"
                print(f"    {metric:<14}: {m['mean']:>8.3f} ± {m['std']:>7.3f} (median: {m['median']:>8.3f}) {unit}")
    
    # Save to CSV
    output_dir = Path(__file__).parent
    
    # Per-frame aggregated CSV
    output_csv = output_dir / 'bdlo_tracking_ablation_aggregate.csv'
    with open(output_csv, 'w', newline='') as f:
        fieldnames = ['Method', 'NumFrames', 
                      'EdgePctMean_mean', 'EdgePctMean_std', 'Edge<5%',
                      'EdgeRMSE_mean', 'EdgeRMSE_std',
                      'PosRMSE_mean', 'PosRMSE_std',
                      'CD_mean', 'CD_std',
                      'F@2mm_mean', 'F@5mm_mean', 'F@10mm_mean']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for method in methods:
            if method not in results:
                continue
            r = results[method]
            row = {
                'Method': method,
                'NumFrames': r['count'],
                'Edge<5%': f"{r.get('Edge<5%', 0):.2f}",
            }
            for metric in ['EdgePctMean', 'EdgeRMSE', 'PosRMSE', 'CD']:
                if metric in r:
                    row[f'{metric}_mean'] = f"{r[metric]['mean']:.4f}"
                    row[f'{metric}_std'] = f"{r[metric]['std']:.4f}"
            for metric in ['F@2mm', 'F@5mm', 'F@10mm']:
                if metric in r:
                    row[f'{metric}_mean'] = f"{r[metric]['mean']:.4f}"
            writer.writerow(row)
    
    print(f"\nSaved aggregate results to: {output_csv}")
    
    # Save corrupted clips info
    corrupted_info_path = output_dir / 'bdlo_tracking_ablation_corrupted_clips.txt'
    with open(corrupted_info_path, 'w') as f:
        f.write("Corrupted clips excluded from bdlo_faster_tracking_ablation_results:\n")
        f.write("=" * 60 + "\n\n")
        for chunk, clip in sorted(CORRUPTED_FASTER):
            f.write(f"  {chunk}/{clip}\n")
        f.write(f"\nTotal: {len(CORRUPTED_FASTER)} clips excluded\n")
    print(f"Saved corrupted clips info to: {corrupted_info_path}")


if __name__ == "__main__":
    main()

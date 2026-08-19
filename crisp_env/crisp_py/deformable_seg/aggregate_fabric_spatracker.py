#!/usr/bin/env python3
"""
Aggregate SpaTrackerV2 results from all fabric chunks.
Output format matches fabric_all_results_summary.txt for comparison.

Usage:
    pixi run python deformable_seg/aggregate_fabric_spatracker.py
"""

import os
import re
import numpy as np
from pathlib import Path

# Fabric datasets and their chunks
DATASETS = {
    "cloth_no_occlusion_back_3sec": [0, 3, 7, 12, 20],
    "cloth_no_occlusion_back_4sec": [8, 13],
    "cloth_no_occlusion_front_3sec": [2, 5, 6, 7, 11, 14, 17],
    "cloth_no_occlusion_front_4sec": [15, 21, 22, 23, 27, 28],
}

RESULTS_DIR = Path("/home/roahmlab/move_some_robots/crisp_env/crisp_py/deformable_seg/fabric_spatracker_results")

def parse_chunk_summary(summary_path: Path) -> dict:
    """
    Parse chunk_summary.txt file and extract metrics.
    
    Format in chunk_summary.txt:
    SpaTracker   | 18.21 ±  8.08%  | 22.00 ±  9.94 mm | ... 
    
    Returns dict with:
        - num_frames: int
        - edge_pct_mean, edge_pct_std
        - edge_rmse_mean, edge_rmse_std
        - pos_rmse_mean, pos_rmse_std
        - cd_mean, cd_std
        - prec_2mm, prec_5mm, prec_10mm
        - rec_2mm, rec_5mm, rec_10mm
        - f_2mm, f_5mm, f_10mm
    """
    if not summary_path.exists():
        return None
    
    with open(summary_path, 'r') as f:
        content = f.read()
    
    metrics = {}
    
    # Parse number of frames - "Total Frames: 600"
    m = re.search(r'Total Frames:\s*(\d+)', content)
    if m:
        metrics['num_frames'] = int(m.group(1))
    else:
        metrics['num_frames'] = 0
    
    # Parse SpaTracker row in FRAME-WEIGHTED SUMMARY table
    # Format: SpaTracker   | 18.21 ±  8.08%  | 22.00 ±  9.94 mm |   0.2%  | 62.03 ± 28.13 mm |   0.0%  | 60.83 ± 30.46 mm |  11.6%
    main_row = re.search(
        r'SpaTracker\s+\|\s*([\d.]+)\s*±\s*([\d.]+)%?\s*\|\s*([\d.]+)\s*±\s*([\d.]+)\s*mm\s*\|\s*([\d.]+)%?\s*\|\s*([\d.]+)\s*±\s*([\d.]+)\s*mm\s*\|\s*([\d.]+)%?\s*\|\s*([\d.]+)\s*±\s*([\d.]+)\s*mm\s*\|\s*([\d.]+)%?',
        content
    )
    if main_row:
        metrics['edge_pct_mean'] = float(main_row.group(1))
        metrics['edge_pct_std'] = float(main_row.group(2))
        metrics['edge_rmse_mean'] = float(main_row.group(3))
        metrics['edge_rmse_std'] = float(main_row.group(4))
        # group 5 is <5%
        metrics['pos_rmse_mean'] = float(main_row.group(6))
        metrics['pos_rmse_std'] = float(main_row.group(7))
        # group 8 is <5mm
        metrics['cd_mean'] = float(main_row.group(9))
        metrics['cd_std'] = float(main_row.group(10))
        metrics['f_10mm'] = float(main_row.group(11))
    
    # Parse Precision/Recall/F-Score row
    # Format: SpaTracker   |      0.5% |      4.6% |     13.8% |      0.4% |      3.7% |     10.0% |    0.5% |    4.1% |   11.6%
    prf_row = re.search(
        r'SpaTracker\s+\|\s*([\d.]+)%\s*\|\s*([\d.]+)%\s*\|\s*([\d.]+)%\s*\|\s*([\d.]+)%\s*\|\s*([\d.]+)%\s*\|\s*([\d.]+)%\s*\|\s*([\d.]+)%\s*\|\s*([\d.]+)%\s*\|\s*([\d.]+)%',
        content
    )
    if prf_row:
        metrics['prec_2mm'] = float(prf_row.group(1))
        metrics['prec_5mm'] = float(prf_row.group(2))
        metrics['prec_10mm'] = float(prf_row.group(3))
        metrics['rec_2mm'] = float(prf_row.group(4))
        metrics['rec_5mm'] = float(prf_row.group(5))
        metrics['rec_10mm'] = float(prf_row.group(6))
        metrics['f_2mm'] = float(prf_row.group(7))
        metrics['f_5mm'] = float(prf_row.group(8))
        metrics['f_10mm'] = float(prf_row.group(9))
    
    return metrics


def weighted_mean_std(values: list, weights: list) -> tuple:
    """Compute weighted mean and pooled std."""
    values = np.array(values)
    weights = np.array(weights)
    
    if weights.sum() == 0:
        return 0.0, 0.0
    
    w_mean = np.average(values, weights=weights)
    # Pooled variance
    variance = np.average((values - w_mean)**2, weights=weights)
    w_std = np.sqrt(variance)
    
    return w_mean, w_std


def aggregate_dataset(dataset_name: str, chunk_ids: list) -> dict:
    """Aggregate metrics for a single dataset."""
    all_metrics = []
    
    for chunk_id in chunk_ids:
        summary_path = RESULTS_DIR / dataset_name / f"chunk_{chunk_id}" / "chunk_summary.txt"
        m = parse_chunk_summary(summary_path)
        if m is not None and m.get('num_frames', 0) > 0:
            all_metrics.append(m)
    
    if not all_metrics:
        return None
    
    # Weights are number of frames
    weights = [m['num_frames'] for m in all_metrics]
    total_frames = sum(weights)
    
    result = {'num_frames': total_frames, 'num_chunks': len(all_metrics)}
    
    # Compute weighted averages
    for key in ['edge_pct_mean', 'edge_rmse_mean', 'pos_rmse_mean', 'cd_mean',
                'prec_2mm', 'prec_5mm', 'prec_10mm',
                'rec_2mm', 'rec_5mm', 'rec_10mm',
                'f_2mm', 'f_5mm', 'f_10mm']:
        if key in all_metrics[0]:
            vals = [m[key] for m in all_metrics]
            result[key] = np.average(vals, weights=weights)
    
    # For std, compute pooled std
    for key_mean, key_std in [('edge_pct_mean', 'edge_pct_std'), 
                              ('edge_rmse_mean', 'edge_rmse_std'),
                              ('pos_rmse_mean', 'pos_rmse_std'),
                              ('cd_mean', 'cd_std')]:
        if key_std in all_metrics[0]:
            stds = [m[key_std] for m in all_metrics]
            result[key_std] = np.average(stds, weights=weights)
    
    return result


def format_metric(value, std=None, precision=2):
    """Format metric with optional std."""
    if std is not None:
        return f"{value:.{precision}f} ± {std:.{precision}f}"
    return f"{value:.{precision}f}"


def main():
    print("=" * 70)
    print("SpaTrackerV2 Fabric Tracking Results - Aggregated Summary")
    print("=" * 70)
    print()
    
    # Collect all results
    dataset_results = {}
    all_metrics = []
    all_weights = []
    
    for dataset_name, chunk_ids in DATASETS.items():
        result = aggregate_dataset(dataset_name, chunk_ids)
        if result is not None:
            dataset_results[dataset_name] = result
            # For overall aggregation, collect per-chunk metrics
            for chunk_id in chunk_ids:
                summary_path = RESULTS_DIR / dataset_name / f"chunk_{chunk_id}" / "chunk_summary.txt"
                m = parse_chunk_summary(summary_path)
                if m is not None and m.get('num_frames', 0) > 0:
                    all_metrics.append(m)
                    all_weights.append(m['num_frames'])
    
    if not dataset_results:
        print("No results found!")
        return
    
    # Print per-dataset results
    print("-" * 70)
    print("Per-Dataset Results:")
    print("-" * 70)
    
    for dataset_name, result in dataset_results.items():
        print(f"\n{dataset_name} ({result['num_chunks']} chunks, {result['num_frames']} frames):")
        print(f"  Edge% (Mean):      {format_metric(result.get('edge_pct_mean', 0), result.get('edge_pct_std', 0))}")
        print(f"  Edge RMSE (mm):    {format_metric(result.get('edge_rmse_mean', 0), result.get('edge_rmse_std', 0))}")
        print(f"  Pos RMSE (mm):     {format_metric(result.get('pos_rmse_mean', 0), result.get('pos_rmse_std', 0))}")
        print(f"  CD (mm):           {format_metric(result.get('cd_mean', 0), result.get('cd_std', 0))}")
        print(f"  Prec @2/5/10mm:    {result.get('prec_2mm', 0):.2f} / {result.get('prec_5mm', 0):.2f} / {result.get('prec_10mm', 0):.2f}")
        print(f"  Rec @2/5/10mm:     {result.get('rec_2mm', 0):.2f} / {result.get('rec_5mm', 0):.2f} / {result.get('rec_10mm', 0):.2f}")
        print(f"  F @2/5/10mm:       {result.get('f_2mm', 0):.2f} / {result.get('f_5mm', 0):.2f} / {result.get('f_10mm', 0):.2f}")
    
    # Compute overall metrics
    print("\n" + "=" * 70)
    print("OVERALL RESULTS (All Datasets Combined)")
    print("=" * 70)
    
    total_chunks = sum(r['num_chunks'] for r in dataset_results.values())
    total_frames = sum(all_weights)
    
    print(f"\nTotal: {total_chunks} chunks, {total_frames} frames")
    print()
    
    # Compute weighted averages
    weights_np = np.array(all_weights)
    
    def weighted_avg(key):
        vals = [m[key] for m in all_metrics if key in m]
        if not vals:
            return 0.0
        return np.average(vals, weights=weights_np[:len(vals)])
    
    edge_pct = weighted_avg('edge_pct_mean')
    edge_pct_std = weighted_avg('edge_pct_std')
    edge_rmse = weighted_avg('edge_rmse_mean')
    edge_rmse_std = weighted_avg('edge_rmse_std')
    pos_rmse = weighted_avg('pos_rmse_mean')
    pos_rmse_std = weighted_avg('pos_rmse_std')
    cd = weighted_avg('cd_mean')
    cd_std = weighted_avg('cd_std')
    
    prec_2mm = weighted_avg('prec_2mm')
    prec_5mm = weighted_avg('prec_5mm')
    prec_10mm = weighted_avg('prec_10mm')
    rec_2mm = weighted_avg('rec_2mm')
    rec_5mm = weighted_avg('rec_5mm')
    rec_10mm = weighted_avg('rec_10mm')
    f_2mm = weighted_avg('f_2mm')
    f_5mm = weighted_avg('f_5mm')
    f_10mm = weighted_avg('f_10mm')
    
    # Compute <5% and <5mm percentages
    edge_under_5pct = sum(1 for m in all_metrics if m.get('edge_rmse_mean', 100) < 5) / len(all_metrics) * 100
    pos_under_5mm = sum(1 for m in all_metrics if m.get('pos_rmse_mean', 100) < 5) / len(all_metrics) * 100
    
    # Print in same format as fabric_all_results_summary.txt
    print("SpaTrackerV2 Baseline:")
    print(f"  Edge% (Mean):      {format_metric(edge_pct, edge_pct_std)}")
    print(f"  Edge RMSE (mm):    {format_metric(edge_rmse, edge_rmse_std)}  (<5%: {edge_under_5pct:.1f}%)")
    print(f"  Pos RMSE (mm):     {format_metric(pos_rmse, pos_rmse_std)}  (<5mm: {pos_under_5mm:.1f}%)")
    print(f"  CD (mm):           {format_metric(cd, cd_std)}")
    print(f"  F@10mm:            {f_10mm:.2f}")
    print()
    print(f"  Precision @2mm:    {prec_2mm:.2f}")
    print(f"  Precision @5mm:    {prec_5mm:.2f}")
    print(f"  Precision @10mm:   {prec_10mm:.2f}")
    print()
    print(f"  Recall @2mm:       {rec_2mm:.2f}")
    print(f"  Recall @5mm:       {rec_5mm:.2f}")
    print(f"  Recall @10mm:      {rec_10mm:.2f}")
    print()
    print(f"  F-Score @2mm:      {f_2mm:.2f}")
    print(f"  F-Score @5mm:      {f_5mm:.2f}")
    print(f"  F-Score @10mm:     {f_10mm:.2f}")
    
    # Save summary to file
    output_path = RESULTS_DIR / "fabric_spatracker_all_results_summary.txt"
    with open(output_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("SpaTrackerV2 Fabric Tracking Results - Aggregated Summary\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("-" * 70 + "\n")
        f.write("Per-Dataset Results:\n")
        f.write("-" * 70 + "\n")
        
        for dataset_name, result in dataset_results.items():
            f.write(f"\n{dataset_name} ({result['num_chunks']} chunks, {result['num_frames']} frames):\n")
            f.write(f"  Edge% (Mean):      {format_metric(result.get('edge_pct_mean', 0), result.get('edge_pct_std', 0))}\n")
            f.write(f"  Edge RMSE (mm):    {format_metric(result.get('edge_rmse_mean', 0), result.get('edge_rmse_std', 0))}\n")
            f.write(f"  Pos RMSE (mm):     {format_metric(result.get('pos_rmse_mean', 0), result.get('pos_rmse_std', 0))}\n")
            f.write(f"  CD (mm):           {format_metric(result.get('cd_mean', 0), result.get('cd_std', 0))}\n")
            f.write(f"  Prec @2/5/10mm:    {result.get('prec_2mm', 0):.2f} / {result.get('prec_5mm', 0):.2f} / {result.get('prec_10mm', 0):.2f}\n")
            f.write(f"  Rec @2/5/10mm:     {result.get('rec_2mm', 0):.2f} / {result.get('rec_5mm', 0):.2f} / {result.get('rec_10mm', 0):.2f}\n")
            f.write(f"  F @2/5/10mm:       {result.get('f_2mm', 0):.2f} / {result.get('f_5mm', 0):.2f} / {result.get('f_10mm', 0):.2f}\n")
        
        f.write("\n" + "=" * 70 + "\n")
        f.write("OVERALL RESULTS (All Datasets Combined)\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Total: {total_chunks} chunks, {total_frames} frames\n\n")
        
        f.write("SpaTrackerV2 Baseline:\n")
        f.write(f"  Edge% (Mean):      {format_metric(edge_pct, edge_pct_std)}\n")
        f.write(f"  Edge RMSE (mm):    {format_metric(edge_rmse, edge_rmse_std)}  (<5%: {edge_under_5pct:.1f}%)\n")
        f.write(f"  Pos RMSE (mm):     {format_metric(pos_rmse, pos_rmse_std)}  (<5mm: {pos_under_5mm:.1f}%)\n")
        f.write(f"  CD (mm):           {format_metric(cd, cd_std)}\n")
        f.write(f"  F@10mm:            {f_10mm:.2f}\n\n")
        f.write(f"  Precision @2mm:    {prec_2mm:.2f}\n")
        f.write(f"  Precision @5mm:    {prec_5mm:.2f}\n")
        f.write(f"  Precision @10mm:   {prec_10mm:.2f}\n\n")
        f.write(f"  Recall @2mm:       {rec_2mm:.2f}\n")
        f.write(f"  Recall @5mm:       {rec_5mm:.2f}\n")
        f.write(f"  Recall @10mm:      {rec_10mm:.2f}\n\n")
        f.write(f"  F-Score @2mm:      {f_2mm:.2f}\n")
        f.write(f"  F-Score @5mm:      {f_5mm:.2f}\n")
        f.write(f"  F-Score @10mm:     {f_10mm:.2f}\n")
    
    print(f"\nSummary saved to: {output_path}")


if __name__ == "__main__":
    main()

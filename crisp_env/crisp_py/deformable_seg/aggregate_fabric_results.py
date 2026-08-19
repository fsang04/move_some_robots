#!/usr/bin/env python3
"""
Aggregate fabric evaluation results across all datasets, chunks, and clips.

Outputs a frame-weighted summary in the exact format of chunk_summary.txt files.
"""

import os
import csv
import numpy as np
from pathlib import Path
from collections import defaultdict


# Results directory
RESULTS_DIR = Path('/home/roahmlab/move_some_robots/crisp_env/crisp_py/deformable_seg/fabric_evaluation_results')

# All datasets
DATASETS = [
    'cloth_no_occlusion_back_3sec',
    'cloth_no_occlusion_back_4sec',
    'cloth_no_occlusion_front_3sec',
    'cloth_no_occlusion_front_4sec',
]

# Methods in order
METHODS = ['Full', 'NoSnap', 'NoGeometry', 'CDCPD']


def collect_all_frames(results_dir: Path, datasets: list = None) -> dict:
    """
    Collect all per-frame data from all datasets, chunks, and clips.
    
    Returns:
        dict with method -> dict of arrays for each metric
    """
    if datasets is None:
        datasets = DATASETS
    
    # Collect data per method
    method_data = {method: defaultdict(list) for method in METHODS}
    metadata = {'datasets': set(), 'chunks': set(), 'clips': set()}
    
    for dataset in datasets:
        dataset_dir = results_dir / dataset
        if not dataset_dir.exists():
            print(f"  Warning: {dataset} not found")
            continue
        
        metadata['datasets'].add(dataset)
        
        # Find all chunk directories
        for chunk_dir in sorted(dataset_dir.iterdir()):
            if not chunk_dir.is_dir() or not chunk_dir.name.startswith('chunk_'):
                continue
            
            metadata['chunks'].add(f"{dataset}/{chunk_dir.name}")
            
            # Find all clip directories
            for clip_dir in sorted(chunk_dir.iterdir()):
                if not clip_dir.is_dir() or not clip_dir.name.startswith('clip_'):
                    continue
                
                per_frame_path = clip_dir / 'per_frame.csv'
                if not per_frame_path.exists():
                    continue
                
                metadata['clips'].add(f"{dataset}/{chunk_dir.name}/{clip_dir.name}")
                
                # Read CSV
                with open(per_frame_path, 'r') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        method = row['Method']
                        if method not in METHODS:
                            continue
                        
                        # Store all numeric columns
                        method_data[method]['EdgePctMean'].append(float(row['EdgePctMean']))
                        method_data[method]['EdgeRMSE'].append(float(row['EdgeRMSE']))
                        method_data[method]['PosRMSE'].append(float(row['PosRMSE']))
                        method_data[method]['Pos<5mm'].append(float(row['Pos<5mm']))
                        method_data[method]['CD'].append(float(row['CD']))
                        method_data[method]['Prec@2mm'].append(float(row['Prec@2mm']))
                        method_data[method]['Prec@5mm'].append(float(row['Prec@5mm']))
                        method_data[method]['Prec@10mm'].append(float(row['Prec@10mm']))
                        method_data[method]['Rec@2mm'].append(float(row['Rec@2mm']))
                        method_data[method]['Rec@5mm'].append(float(row['Rec@5mm']))
                        method_data[method]['Rec@10mm'].append(float(row['Rec@10mm']))
                        method_data[method]['F@2mm'].append(float(row['F@2mm']))
                        method_data[method]['F@5mm'].append(float(row['F@5mm']))
                        method_data[method]['F@10mm'].append(float(row['F@10mm']))
                        method_data[method]['dataset'].append(dataset)
    
    # Convert lists to numpy arrays
    for method in METHODS:
        for key in list(method_data[method].keys()):
            if key != 'dataset':
                method_data[method][key] = np.array(method_data[method][key])
    
    return method_data, metadata


def compute_frame_weighted_summary(method_data: dict, dataset_filter: str = None) -> dict:
    """
    Compute frame-weighted statistics for each method.
    
    Returns:
        dict with statistics per method
    """
    results = {}
    
    for method in METHODS:
        data = method_data[method]
        
        # Filter by dataset if specified
        if dataset_filter:
            mask = np.array([d == dataset_filter for d in data['dataset']])
            if not np.any(mask):
                continue
        else:
            mask = np.ones(len(data['EdgePctMean']), dtype=bool)
        
        n_frames = np.sum(mask)
        
        if n_frames == 0:
            continue
        
        # Edge% statistics
        edge_pct = data['EdgePctMean'][mask]
        edge_pct_mean = np.mean(edge_pct)
        edge_pct_std = np.std(edge_pct)
        
        # EdgeRMSE statistics
        edge_rmse = data['EdgeRMSE'][mask]
        edge_rmse_mean = np.mean(edge_rmse)
        edge_rmse_std = np.std(edge_rmse)
        
        # Edge% < 5% (percentage of frames)
        edge_under_5pct = np.mean(edge_pct < 5.0) * 100
        
        # PosRMSE statistics
        pos_rmse = data['PosRMSE'][mask]
        pos_rmse_mean = np.mean(pos_rmse)
        pos_rmse_std = np.std(pos_rmse)
        
        # Pos < 5mm (from Pos<5mm column - it's actually percentage of keypoints)
        pos_under_5mm = np.mean(data['Pos<5mm'][mask])
        
        # CD statistics
        cd = data['CD'][mask]
        cd_mean = np.mean(cd)
        cd_std = np.std(cd)
        
        # Precision/Recall/F-score
        prec_2mm = np.mean(data['Prec@2mm'][mask])
        prec_5mm = np.mean(data['Prec@5mm'][mask])
        prec_10mm = np.mean(data['Prec@10mm'][mask])
        
        rec_2mm = np.mean(data['Rec@2mm'][mask])
        rec_5mm = np.mean(data['Rec@5mm'][mask])
        rec_10mm = np.mean(data['Rec@10mm'][mask])
        
        f_2mm = np.mean(data['F@2mm'][mask])
        f_5mm = np.mean(data['F@5mm'][mask])
        f_10mm = np.mean(data['F@10mm'][mask])
        
        results[method] = {
            'n_frames': n_frames,
            'edge_pct_mean': edge_pct_mean,
            'edge_pct_std': edge_pct_std,
            'edge_rmse_mean': edge_rmse_mean,
            'edge_rmse_std': edge_rmse_std,
            'edge_under_5pct': edge_under_5pct,
            'pos_rmse_mean': pos_rmse_mean,
            'pos_rmse_std': pos_rmse_std,
            'pos_under_5mm': pos_under_5mm,
            'cd_mean': cd_mean,
            'cd_std': cd_std,
            'prec_2mm': prec_2mm,
            'prec_5mm': prec_5mm,
            'prec_10mm': prec_10mm,
            'rec_2mm': rec_2mm,
            'rec_5mm': rec_5mm,
            'rec_10mm': rec_10mm,
            'f_2mm': f_2mm,
            'f_5mm': f_5mm,
            'f_10mm': f_10mm,
        }
    
    return results


def format_summary(results: dict, title: str, total_frames: int) -> str:
    """
    Format the summary in the exact style of chunk_summary.txt.
    """
    lines = []
    
    # Header
    lines.append(f"{title} ({total_frames} total frames)")
    lines.append("=" * 120)
    lines.append("")
    
    # Frame-weighted summary table
    lines.append("FRAME-WEIGHTED SUMMARY (pooling all frames across clips)")
    lines.append("-" * 120)
    lines.append(f"{'Method':<12} | {'Edge%':<15} | {'EdgeRMSE (mm)':<15} | {'<5%':<8} | {'PosRMSE (mm)':<15} | {'<5mm':<8} | {'CD (mm)':<15} | {'F@10mm':<8}")
    lines.append("-" * 120)
    
    for method in METHODS:
        if method not in results:
            continue
        r = results[method]
        
        edge_pct_str = f"{r['edge_pct_mean']:5.2f} ± {r['edge_pct_std']:5.2f}%"
        edge_rmse_str = f"{r['edge_rmse_mean']:5.2f} ±{r['edge_rmse_std']:5.2f} mm"
        edge_under_str = f"{r['edge_under_5pct']:5.1f}%"
        pos_rmse_str = f"{r['pos_rmse_mean']:5.2f} ±{r['pos_rmse_std']:5.2f} mm"
        pos_under_str = f"{r['pos_under_5mm']:5.1f}%"
        cd_str = f"{r['cd_mean']:5.2f} ±{r['cd_std']:5.2f} mm"
        f10_str = f"{r['f_10mm']:5.1f}%"
        
        lines.append(f"{method:<12} | {edge_pct_str:<15} | {edge_rmse_str:<15} | {edge_under_str:<8} | {pos_rmse_str:<15} | {pos_under_str:<8} | {cd_str:<15} | {f10_str:<8}")
    
    lines.append("")
    
    # Precision/Recall/F-Score table
    lines.append("Precision/Recall/F-Score:")
    lines.append("-" * 100)
    lines.append(f"{'Method':<12} | {'Prec@2mm':<10} | {'Prec@5mm':<10} | {'Prec@10mm':<10} | {'Rec@2mm':<10} | {'Rec@5mm':<10} | {'Rec@10mm':<10} | {'F@2mm':<8} | {'F@5mm':<8} | {'F@10mm':<8}")
    lines.append("-" * 100)
    
    for method in METHODS:
        if method not in results:
            continue
        r = results[method]
        
        lines.append(
            f"{method:<12} | {r['prec_2mm']:8.1f}% | {r['prec_5mm']:8.1f}% | {r['prec_10mm']:8.1f}% | "
            f"{r['rec_2mm']:8.1f}% | {r['rec_5mm']:8.1f}% | {r['rec_10mm']:8.1f}% | "
            f"{r['f_2mm']:6.1f}% | {r['f_5mm']:6.1f}% | {r['f_10mm']:6.1f}%"
        )
    
    return "\n".join(lines)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Aggregate fabric evaluation results')
    parser.add_argument('--dataset', type=str, default=None,
                        help='Specific dataset to aggregate (default: all)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file path (default: print to stdout)')
    parser.add_argument('--per-dataset', action='store_true',
                        help='Also compute per-dataset summaries')
    args = parser.parse_args()
    
    print("Collecting all per-frame data...")
    
    if args.dataset:
        datasets = [args.dataset]
        title = f"Dataset: {args.dataset}"
    else:
        datasets = DATASETS
        title = "All Fabric Datasets Combined"
    
    # Collect all frames
    method_data, metadata = collect_all_frames(RESULTS_DIR, datasets)
    total_frames = len(method_data['Full']['EdgePctMean'])
    
    print(f"  Total frames collected: {total_frames}")
    print(f"  Datasets: {list(metadata['datasets'])}")
    print(f"  Chunks: {len(metadata['chunks'])}")
    print(f"  Clips: {len(metadata['clips'])}")
    
    # Compute summary
    results = compute_frame_weighted_summary(method_data)
    
    # Format and print
    output = format_summary(results, title, total_frames)
    print("\n" + output)
    
    # Per-dataset summaries if requested
    if args.per_dataset and not args.dataset:
        print("\n\n" + "=" * 120)
        print("PER-DATASET SUMMARIES")
        print("=" * 120)
        
        for dataset in DATASETS:
            if dataset not in metadata['datasets']:
                continue
            
            dataset_results = compute_frame_weighted_summary(method_data, dataset_filter=dataset)
            if not dataset_results:
                continue
            
            n_frames = dataset_results['Full']['n_frames']
            dataset_output = format_summary(dataset_results, f"Dataset: {dataset}", n_frames)
            print("\n" + dataset_output)
            output += "\n\n" + dataset_output
    
    # Save to file if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(output)
        print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()

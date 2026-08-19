#!/usr/bin/env python3
"""
Aggregate DLO SpaTracker results across all chunks.

Usage:
    python aggregate_dlo_spatracker.py
"""

import numpy as np
from pathlib import Path
import re


def parse_chunk_summary(summary_path: Path) -> dict:
    """Parse chunk_summary.txt file to extract metrics."""
    content = summary_path.read_text()
    
    result = {
        'frames': 0,
        'clips': 0,
        'edge_pct_mean': 0.0,
        'edge_pct_std': 0.0,
        'edge_rmse_mean': 0.0,
        'edge_rmse_std': 0.0,
        'edge_under_5pct': 0.0,
        'pos_rmse_mean': 0.0,
        'pos_rmse_std': 0.0,
        'pos_under_5mm': 0.0,
        'cd_mean': 0.0,
        'cd_std': 0.0,
        'f_10mm': 0.0,
        'precision_2mm': 0.0,
        'precision_5mm': 0.0,
        'precision_10mm': 0.0,
        'recall_2mm': 0.0,
        'recall_5mm': 0.0,
        'recall_10mm': 0.0,
        'f_2mm': 0.0,
        'f_5mm': 0.0,
    }
    
    # Parse total frames and clips
    frames_match = re.search(r'Total Frames:\s*(\d+)', content)
    if frames_match:
        result['frames'] = int(frames_match.group(1))
    
    clips_match = re.search(r'Number of Clips:\s*(\d+)', content)
    if clips_match:
        result['clips'] = int(clips_match.group(1))
    
    # Parse the table line for SpaTracker
    # Format: SpaTracker | 18.21 ± 8.08%  | 41.25 ± 10.23 mm | ...
    spatracker_match = re.search(
        r'SpaTracker\s*\|\s*([\d.]+)\s*±\s*([\d.]+)%\s*\|\s*'  # Edge%
        r'([\d.]+)\s*±\s*([\d.]+)\s*mm\s*\|\s*'                 # EdgeRMSE
        r'([\d.]+)%\s*\|\s*'                                    # <5%
        r'([\d.]+)\s*±\s*([\d.]+)\s*mm\s*\|\s*'                 # PosRMSE
        r'([\d.]+)%\s*\|\s*'                                    # <5mm
        r'([\d.]+)\s*±\s*([\d.]+)\s*mm\s*\|\s*'                 # CD
        r'([\d.]+)%',                                           # F@10mm
        content
    )
    
    if spatracker_match:
        result['edge_pct_mean'] = float(spatracker_match.group(1))
        result['edge_pct_std'] = float(spatracker_match.group(2))
        result['edge_rmse_mean'] = float(spatracker_match.group(3))
        result['edge_rmse_std'] = float(spatracker_match.group(4))
        result['edge_under_5pct'] = float(spatracker_match.group(5))
        result['pos_rmse_mean'] = float(spatracker_match.group(6))
        result['pos_rmse_std'] = float(spatracker_match.group(7))
        result['pos_under_5mm'] = float(spatracker_match.group(8))
        result['cd_mean'] = float(spatracker_match.group(9))
        result['cd_std'] = float(spatracker_match.group(10))
        result['f_10mm'] = float(spatracker_match.group(11))
    
    # Parse precision/recall line
    prec_rec_match = re.search(
        r'SpaTracker\s*\|\s*([\d.]+)%\s*\|\s*'   # Prec@2mm
        r'([\d.]+)%\s*\|\s*'                     # Prec@5mm
        r'([\d.]+)%\s*\|\s*'                     # Prec@10mm
        r'([\d.]+)%\s*\|\s*'                     # Rec@2mm
        r'([\d.]+)%\s*\|\s*'                     # Rec@5mm
        r'([\d.]+)%\s*\|\s*'                     # Rec@10mm
        r'([\d.]+)%\s*\|\s*'                     # F@2mm
        r'([\d.]+)%\s*\|\s*'                     # F@5mm
        r'([\d.]+)%',                            # F@10mm (duplicate)
        content
    )
    
    if prec_rec_match:
        result['precision_2mm'] = float(prec_rec_match.group(1))
        result['precision_5mm'] = float(prec_rec_match.group(2))
        result['precision_10mm'] = float(prec_rec_match.group(3))
        result['recall_2mm'] = float(prec_rec_match.group(4))
        result['recall_5mm'] = float(prec_rec_match.group(5))
        result['recall_10mm'] = float(prec_rec_match.group(6))
        result['f_2mm'] = float(prec_rec_match.group(7))
        result['f_5mm'] = float(prec_rec_match.group(8))
    
    return result


def main():
    results_dir = Path(__file__).parent / 'dlo_spatracker_results'
    
    if not results_dir.exists():
        print(f"ERROR: Results directory not found: {results_dir}")
        return
    
    # Collect all chunk results
    all_results = []
    
    for chunk_dir in sorted(results_dir.glob('chunk_*')):
        summary_path = chunk_dir / 'chunk_summary.txt'
        if not summary_path.exists():
            print(f"  Skipping {chunk_dir.name}: no chunk_summary.txt")
            continue
        
        result = parse_chunk_summary(summary_path)
        result['chunk'] = chunk_dir.name
        
        if result['frames'] > 0:
            all_results.append(result)
            print(f"  {chunk_dir.name}: {result['frames']} frames, Edge% {result['edge_pct_mean']:.2f}%, F@10mm {result['f_10mm']:.1f}%")
    
    if not all_results:
        print("No results found!")
        return
    
    # Compute weighted aggregates
    total_frames = sum(r['frames'] for r in all_results)
    total_clips = sum(r['clips'] for r in all_results)
    
    def weighted_avg(key):
        return sum(r[key] * r['frames'] for r in all_results) / total_frames
    
    # Also compute edge % < 5% from per_frame CSVs
    edge_pct_values = []
    for chunk_dir in results_dir.glob('chunk_*'):
        for csv_path in chunk_dir.glob('clip_*/per_frame.csv'):
            try:
                with open(csv_path) as f:
                    header = f.readline().strip().split(',')
                    edge_idx = header.index('EdgePctMean')
                    for line in f:
                        parts = line.strip().split(',')
                        if len(parts) > edge_idx:
                            edge_pct_values.append(float(parts[edge_idx]))
            except (ValueError, IndexError):
                pass
    
    edge_under_5_from_csv = np.mean(np.array(edge_pct_values) < 5.0) * 100 if edge_pct_values else 0
    edge_under_10_from_csv = np.mean(np.array(edge_pct_values) < 10.0) * 100 if edge_pct_values else 0
    edge_under_15_from_csv = np.mean(np.array(edge_pct_values) < 15.0) * 100 if edge_pct_values else 0
    
    # Print summary
    print("\n" + "=" * 100)
    print("DLO SpaTracker Results - All Chunks Aggregated")
    print("=" * 100)
    print(f"\nTotal Chunks: {len(all_results)}")
    print(f"Total Clips: {total_clips}")
    print(f"Total Frames: {total_frames}")
    
    print("\n" + "-" * 100)
    print(f"{'Method':<12} | {'Edge%':<16} | {'<5%':<8} | {'EdgeRMSE(mm)':<16} | {'PosRMSE(mm)':<16} | {'CD(mm)':<16} | {'F@10mm':<8}")
    print("-" * 100)
    
    edge_pct = f"{weighted_avg('edge_pct_mean'):.2f} ± {weighted_avg('edge_pct_std'):.2f}%"
    edge_rmse = f"{weighted_avg('edge_rmse_mean'):.2f} ± {weighted_avg('edge_rmse_std'):.2f}"
    pos_rmse = f"{weighted_avg('pos_rmse_mean'):.2f} ± {weighted_avg('pos_rmse_std'):.2f}"
    cd = f"{weighted_avg('cd_mean'):.2f} ± {weighted_avg('cd_std'):.2f}"
    f10 = f"{weighted_avg('f_10mm'):.1f}%"
    edge_under = f"{edge_under_5_from_csv:.1f}%"
    
    print(f"{'SpaTracker':<12} | {edge_pct:<16} | {edge_under:<8} | {edge_rmse:<16} | {pos_rmse:<16} | {cd:<16} | {f10:<8}")
    print("-" * 100)
    
    print("\nEdge % Thresholds (from per-frame CSV):")
    print(f"  <5%:  {edge_under_5_from_csv:.1f}%")
    print(f"  <10%: {edge_under_10_from_csv:.1f}%")
    print(f"  <15%: {edge_under_15_from_csv:.1f}%")
    
    print("\nPrecision/Recall/F-Score:")
    print(f"  Prec@2mm:  {weighted_avg('precision_2mm'):.1f}%")
    print(f"  Prec@5mm:  {weighted_avg('precision_5mm'):.1f}%")
    print(f"  Prec@10mm: {weighted_avg('precision_10mm'):.1f}%")
    print(f"  Rec@2mm:   {weighted_avg('recall_2mm'):.1f}%")
    print(f"  Rec@5mm:   {weighted_avg('recall_5mm'):.1f}%")
    print(f"  Rec@10mm:  {weighted_avg('recall_10mm'):.1f}%")
    print(f"  F@2mm:     {weighted_avg('f_2mm'):.1f}%")
    print(f"  F@5mm:     {weighted_avg('f_5mm'):.1f}%")
    print(f"  F@10mm:    {weighted_avg('f_10mm'):.1f}%")
    
    # Save summary
    output_path = results_dir / 'dlo_spatracker_all_results.txt'
    with open(output_path, 'w') as f:
        f.write("=" * 100 + "\n")
        f.write("DLO SpaTracker Results - All Chunks Aggregated\n")
        f.write("=" * 100 + "\n\n")
        f.write(f"Total Chunks: {len(all_results)}\n")
        f.write(f"Total Clips: {total_clips}\n")
        f.write(f"Total Frames: {total_frames}\n\n")
        
        f.write("-" * 100 + "\n")
        f.write(f"{'Method':<12} | {'Edge%':<16} | {'<5%':<8} | {'EdgeRMSE(mm)':<16} | {'PosRMSE(mm)':<16} | {'CD(mm)':<16} | {'F@10mm':<8}\n")
        f.write("-" * 100 + "\n")
        f.write(f"{'SpaTracker':<12} | {edge_pct:<16} | {edge_under:<8} | {edge_rmse:<16} | {pos_rmse:<16} | {cd:<16} | {f10:<8}\n")
        f.write("-" * 100 + "\n\n")
        
        f.write("Edge % Thresholds (from per-frame CSV):\n")
        f.write(f"  <5%:  {edge_under_5_from_csv:.1f}%\n")
        f.write(f"  <10%: {edge_under_10_from_csv:.1f}%\n")
        f.write(f"  <15%: {edge_under_15_from_csv:.1f}%\n\n")
        
        f.write("Precision/Recall/F-Score:\n")
        f.write(f"  Prec@2mm:  {weighted_avg('precision_2mm'):.1f}%\n")
        f.write(f"  Prec@5mm:  {weighted_avg('precision_5mm'):.1f}%\n")
        f.write(f"  Prec@10mm: {weighted_avg('precision_10mm'):.1f}%\n")
        f.write(f"  Rec@2mm:   {weighted_avg('recall_2mm'):.1f}%\n")
        f.write(f"  Rec@5mm:   {weighted_avg('recall_5mm'):.1f}%\n")
        f.write(f"  Rec@10mm:  {weighted_avg('recall_10mm'):.1f}%\n")
        f.write(f"  F@2mm:     {weighted_avg('f_2mm'):.1f}%\n")
        f.write(f"  F@5mm:     {weighted_avg('f_5mm'):.1f}%\n")
        f.write(f"  F@10mm:    {weighted_avg('f_10mm'):.1f}%\n")
        
        f.write("\n\nPer-Chunk Results:\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Chunk':<12} | {'Frames':<8} | {'Edge%':<10} | {'PosRMSE':<10} | {'CD':<10} | {'F@10mm':<8}\n")
        f.write("-" * 80 + "\n")
        for r in all_results:
            f.write(f"{r['chunk']:<12} | {r['frames']:<8} | {r['edge_pct_mean']:<10.2f} | {r['pos_rmse_mean']:<10.2f} | {r['cd_mean']:<10.2f} | {r['f_10mm']:<8.1f}\n")
    
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()

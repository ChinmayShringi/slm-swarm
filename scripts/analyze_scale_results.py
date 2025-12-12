#!/usr/bin/env python3

"""
Multi-Scale SLM Swarm Results Analysis

Aggregates and analyzes results from scaling experiments (3, 6, 12, 18 workers)
in both heterogeneous and homogeneous configurations.
"""

import argparse
import json
import os
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any


def load_metrics(config_dir: Path) -> Dict[str, Any]:
    """Load metrics from a configuration directory."""
    metrics_file = config_dir / "metrics.csv"
    metadata_file = config_dir / "metadata.txt"
    
    if not metrics_file.exists():
        print(f"Warning: metrics.csv not found in {config_dir}")
        return None
    
    # Load CSV metrics
    df = pd.read_csv(metrics_file)
    metrics = df.iloc[0].to_dict()
    
    # Load metadata
    metadata = {}
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            for line in f:
                if ':' in line:
                    key, value = line.split(':', 1)
                    metadata[key.strip()] = value.strip()
    
    return {
        'metrics': metrics,
        'metadata': metadata,
        'config_name': config_dir.name
    }


def parse_config_name(config_name: str) -> Dict[str, Any]:
    """Parse configuration name to extract swarm type and worker count."""
    parts = config_name.split('-')
    return {
        'swarm_type': parts[0],  # hetero or homo
        'num_workers': int(parts[1]) if len(parts) > 1 else 0
    }


def analyze_results(results_dir: Path) -> pd.DataFrame:
    """Analyze all results and create comparison dataframe."""
    all_data = []
    
    # Find all configuration directories
    config_dirs = [d for d in results_dir.iterdir() if d.is_dir() and (d.name.startswith('hetero-') or d.name.startswith('homo-'))]
    
    if not config_dirs:
        print(f"No experiment directories found in {results_dir}")
        return None
    
    print(f"Found {len(config_dirs)} experiment results")
    
    for config_dir in sorted(config_dirs):
        print(f"Loading {config_dir.name}...")
        data = load_metrics(config_dir)
        if data:
            config_info = parse_config_name(data['config_name'])
            
            row = {
                'Configuration': data['config_name'],
                'Swarm Type': config_info['swarm_type'],
                'Workers': config_info['num_workers'],
                'ROUGE-1': data['metrics'].get('mean_rouge1', 0),
                'ROUGE-2': data['metrics'].get('mean_rouge2', 0),
                'ROUGE-L': data['metrics'].get('mean_rougeL', 0),
                'BERTScore': data['metrics'].get('mean_bertscore', 0),
                'Avg Consensus Similarity': data['metrics'].get('mean_avg_consensus_similarity', 0),
                'Outlier Detection Rate': data['metrics'].get('outlier_detection_rate', 0),
                'Latency p50 (s)': data['metrics'].get('latency_p50', 0),
                'Latency p95 (s)': data['metrics'].get('latency_p95', 0),
                'Latency Mean (s)': data['metrics'].get('latency_mean', 0),
                'Duration (s)': int(data['metadata'].get('Duration', '0').split()[0]) if 'Duration' in data['metadata'] else 0
            }
            all_data.append(row)
    
    df = pd.DataFrame(all_data)
    df = df.sort_values(['Swarm Type', 'Workers'])
    
    return df


def generate_report(df: pd.DataFrame, results_dir: Path):
    """Generate markdown report with analysis."""
    report_file = results_dir / "SCALING_REPORT.md"
    
    with open(report_file, 'w') as f:
        f.write("# Multi-Scale SLM Swarm Evaluation Report\n\n")
        f.write(f"**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Overall summary
        f.write("## Executive Summary\n\n")
        f.write(f"- Total configurations tested: {len(df)}\n")
        f.write(f"- Worker counts: {sorted(df['Workers'].unique())}\n")
        f.write(f"- Swarm types: {', '.join(sorted(df['Swarm Type'].unique()))}\n\n")
        
        # Full results table
        f.write("## Complete Results\n\n")
        f.write(df.to_markdown(index=False, floatfmt=".4f"))
        f.write("\n\n")
        
        # Quality metrics by worker count
        f.write("## Quality Metrics by Worker Count\n\n")
        
        # Heterogeneous swarms
        hetero_df = df[df['Swarm Type'] == 'hetero'].copy()
        if not hetero_df.empty:
            f.write("### Heterogeneous Swarms\n\n")
            f.write("| Workers | ROUGE-1 | ROUGE-2 | ROUGE-L | BERTScore |\n")
            f.write("|---------|---------|---------|---------|----------|\n")
            for _, row in hetero_df.iterrows():
                f.write(f"| {row['Workers']} | {row['ROUGE-1']:.4f} | {row['ROUGE-2']:.4f} | {row['ROUGE-L']:.4f} | {row['BERTScore']:.4f} |\n")
            f.write("\n")
        
        # Homogeneous swarms
        homo_df = df[df['Swarm Type'] == 'homo'].copy()
        if not homo_df.empty:
            f.write("### Homogeneous Swarms\n\n")
            f.write("| Workers | ROUGE-1 | ROUGE-2 | ROUGE-L | BERTScore |\n")
            f.write("|---------|---------|---------|---------|----------|\n")
            for _, row in homo_df.iterrows():
                f.write(f"| {row['Workers']} | {row['ROUGE-1']:.4f} | {row['ROUGE-2']:.4f} | {row['ROUGE-L']:.4f} | {row['BERTScore']:.4f} |\n")
            f.write("\n")
        
        # Latency analysis
        f.write("## Latency Analysis\n\n")
        f.write("### Heterogeneous Swarms\n\n")
        if not hetero_df.empty:
            f.write("| Workers | p50 (s) | p95 (s) | Mean (s) |\n")
            f.write("|---------|---------|---------|----------|\n")
            for _, row in hetero_df.iterrows():
                f.write(f"| {row['Workers']} | {row['Latency p50 (s)']:.3f} | {row['Latency p95 (s)']:.3f} | {row['Latency Mean (s)']:.3f} |\n")
            f.write("\n")
        
        f.write("### Homogeneous Swarms\n\n")
        if not homo_df.empty:
            f.write("| Workers | p50 (s) | p95 (s) | Mean (s) |\n")
            f.write("|---------|---------|---------|----------|\n")
            for _, row in homo_df.iterrows():
                f.write(f"| {row['Workers']} | {row['Latency p50 (s)']:.3f} | {row['Latency p95 (s)']:.3f} | {row['Latency Mean (s)']:.3f} |\n")
            f.write("\n")
        
        # Consensus analysis
        f.write("## Consensus Analysis\n\n")
        f.write("### Heterogeneous Swarms\n\n")
        if not hetero_df.empty:
            f.write("| Workers | Avg Consensus Similarity | Outlier Detection Rate |\n")
            f.write("|---------|--------------------------|------------------------|\n")
            for _, row in hetero_df.iterrows():
                f.write(f"| {row['Workers']} | {row['Avg Consensus Similarity']:.4f} | {row['Outlier Detection Rate']:.2%} |\n")
            f.write("\n")
        
        f.write("### Homogeneous Swarms\n\n")
        if not homo_df.empty:
            f.write("| Workers | Avg Consensus Similarity | Outlier Detection Rate |\n")
            f.write("|---------|--------------------------|------------------------|\n")
            for _, row in homo_df.iterrows():
                f.write(f"| {row['Workers']} | {row['Avg Consensus Similarity']:.4f} | {row['Outlier Detection Rate']:.2%} |\n")
            f.write("\n")
        
        # Key findings
        f.write("## Key Findings\n\n")
        
        # Best performing configuration
        best_bertscore = df.loc[df['BERTScore'].idxmax()]
        f.write(f"### Highest Quality (BERTScore)\n")
        f.write(f"- **Configuration:** {best_bertscore['Configuration']}\n")
        f.write(f"- **BERTScore:** {best_bertscore['BERTScore']:.4f}\n")
        f.write(f"- **ROUGE-L:** {best_bertscore['ROUGE-L']:.4f}\n\n")
        
        # Fastest configuration
        fastest = df.loc[df['Latency Mean (s)'].idxmin()]
        f.write(f"### Fastest Configuration\n")
        f.write(f"- **Configuration:** {fastest['Configuration']}\n")
        f.write(f"- **Mean Latency:** {fastest['Latency Mean (s)']:.3f} seconds\n")
        f.write(f"- **Quality (BERTScore):** {fastest['BERTScore']:.4f}\n\n")
        
        # Quality scaling analysis
        if len(hetero_df) > 1:
            quality_improvement = ((hetero_df['BERTScore'].iloc[-1] - hetero_df['BERTScore'].iloc[0]) / 
                                   hetero_df['BERTScore'].iloc[0] * 100)
            f.write(f"### Heterogeneous Swarm Scaling\n")
            f.write(f"- **Quality improvement** ({hetero_df['Workers'].iloc[0]} → {hetero_df['Workers'].iloc[-1]} workers): "
                   f"{quality_improvement:+.2f}%\n")
            
            latency_change = ((hetero_df['Latency Mean (s)'].iloc[-1] - hetero_df['Latency Mean (s)'].iloc[0]) / 
                             hetero_df['Latency Mean (s)'].iloc[0] * 100)
            f.write(f"- **Latency change** ({hetero_df['Workers'].iloc[0]} → {hetero_df['Workers'].iloc[-1]} workers): "
                   f"{latency_change:+.2f}%\n\n")
        
        if len(homo_df) > 1:
            quality_improvement = ((homo_df['BERTScore'].iloc[-1] - homo_df['BERTScore'].iloc[0]) / 
                                   homo_df['BERTScore'].iloc[0] * 100)
            f.write(f"### Homogeneous Swarm Scaling\n")
            f.write(f"- **Quality improvement** ({homo_df['Workers'].iloc[0]} → {homo_df['Workers'].iloc[-1]} workers): "
                   f"{quality_improvement:+.2f}%\n")
            
            latency_change = ((homo_df['Latency Mean (s)'].iloc[-1] - homo_df['Latency Mean (s)'].iloc[0]) / 
                             homo_df['Latency Mean (s)'].iloc[0] * 100)
            f.write(f"- **Latency change** ({homo_df['Workers'].iloc[0]} → {homo_df['Workers'].iloc[-1]} workers): "
                   f"{latency_change:+.2f}%\n\n")
        
        # Recommendations
        f.write("## Recommendations\n\n")
        
        if not hetero_df.empty and not homo_df.empty:
            avg_hetero_quality = hetero_df['BERTScore'].mean()
            avg_homo_quality = homo_df['BERTScore'].mean()
            
            if avg_hetero_quality > avg_homo_quality * 1.01:
                f.write("- **Heterogeneous swarms** show better overall quality metrics\n")
            elif avg_homo_quality > avg_hetero_quality * 1.01:
                f.write("- **Homogeneous swarms** show better overall quality metrics\n")
            else:
                f.write("- Quality metrics are similar between heterogeneous and homogeneous swarms\n")
            
            avg_hetero_latency = hetero_df['Latency Mean (s)'].mean()
            avg_homo_latency = homo_df['Latency Mean (s)'].mean()
            
            if avg_hetero_latency < avg_homo_latency * 0.95:
                f.write("- **Heterogeneous swarms** are faster on average\n")
            elif avg_homo_latency < avg_hetero_latency * 0.95:
                f.write("- **Homogeneous swarms** are faster on average\n")
            else:
                f.write("- Latency is similar between heterogeneous and homogeneous swarms\n")
        
        # Quality saturation
        if len(df) >= 4:
            quality_changes = df.groupby('Swarm Type')['BERTScore'].apply(lambda x: x.diff().iloc[-1] if len(x) > 1 else 0)
            if any(abs(quality_changes) < 0.001):
                f.write("- Quality appears to **saturate** at higher worker counts\n")
        
        f.write("\n")
        
        # Experiment details
        f.write("## Experiment Details\n\n")
        f.write("- **Dataset:** SAMSum (test split)\n")
        f.write("- **Consensus Method:** Cosine similarity\n")
        f.write("- **Evaluation Metrics:** ROUGE-1/2/L, BERTScore, consensus similarity\n\n")
        
        # Per-configuration details
        f.write("## Configuration Details\n\n")
        for _, row in df.iterrows():
            f.write(f"### {row['Configuration']}\n")
            f.write(f"- **Workers:** {row['Workers']}\n")
            f.write(f"- **Type:** {row['Swarm Type']}\n")
            f.write(f"- **Duration:** {row['Duration (s)']} seconds\n")
            f.write(f"- **ROUGE-1/2/L:** {row['ROUGE-1']:.4f} / {row['ROUGE-2']:.4f} / {row['ROUGE-L']:.4f}\n")
            f.write(f"- **BERTScore:** {row['BERTScore']:.4f}\n")
            f.write(f"- **Latency:** {row['Latency Mean (s)']:.3f}s (mean), {row['Latency p50 (s)']:.3f}s (p50), {row['Latency p95 (s)']:.3f}s (p95)\n")
            f.write(f"- **Consensus:** {row['Avg Consensus Similarity']:.4f} (similarity), {row['Outlier Detection Rate']:.2%} (outlier rate)\n\n")
    
    print(f"\nReport generated: {report_file}")
    return report_file


def main():
    parser = argparse.ArgumentParser(description='Analyze multi-scale swarm experiment results')
    parser.add_argument('--results-dir', required=True, help='Results directory containing experiment outputs')
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    
    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        return 1
    
    print(f"Analyzing results in: {results_dir}")
    
    # Analyze results
    df = analyze_results(results_dir)
    
    if df is None or df.empty:
        print("Error: No valid results found")
        return 1
    
    # Save CSV summary
    csv_file = results_dir / "scaling_analysis.csv"
    df.to_csv(csv_file, index=False, float_format='%.4f')
    print(f"Summary saved: {csv_file}")
    
    # Generate report
    report_file = generate_report(df, results_dir)
    
    print("\n" + "="*50)
    print("Analysis complete!")
    print(f"View the report at: {report_file}")
    print("="*50)
    
    return 0


if __name__ == "__main__":
    exit(main())


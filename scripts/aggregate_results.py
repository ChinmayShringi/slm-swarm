#!/usr/bin/env python3
"""
Script to aggregate results from multiple evaluation runs

Usage:
    python aggregate_results.py [--results-dir ../results]
"""

import json
import argparse
from pathlib import Path
import pandas as pd


def aggregate_results(results_dir: str):
    """
    Aggregate metrics from all runs in results directory
    
    Args:
        results_dir: Path to results directory containing run_XXX folders
    """
    results_path = Path(results_dir)
    
    if not results_path.exists():
        print(f"Error: Results directory not found: {results_path}")
        return
    
    # Find all run directories
    run_dirs = sorted([d for d in results_path.iterdir() if d.is_dir() and d.name.startswith("run_")])
    
    if not run_dirs:
        print(f"No evaluation runs found in {results_path}")
        return
    
    print(f"Found {len(run_dirs)} evaluation runs\n")
    
    # Collect all metrics
    all_metrics = []
    
    for run_dir in run_dirs:
        metrics_file = run_dir / "metrics.csv"
        if not metrics_file.exists():
            print(f"Warning: No metrics.csv in {run_dir.name}")
            continue
        
        try:
            df = pd.read_csv(metrics_file)
            df['run_id'] = run_dir.name
            all_metrics.append(df)
        except Exception as e:
            print(f"Error reading {metrics_file}: {e}")
    
    if not all_metrics:
        print("No valid metrics found")
        return
    
    # Combine all metrics
    combined_df = pd.concat(all_metrics, ignore_index=True)
    
    # Display summary table
    print("="*100)
    print("AGGREGATE RESULTS SUMMARY")
    print("="*100)
    
    # Key metrics to display
    key_metrics = [
        'swarm_rouge1_f1', 'big_rouge1_f1',
        'swarm_rougeL_f1', 'big_rougeL_f1',
        'swarm_bertscore_f1', 'big_bertscore_f1',
        'swarm_latency_p50', 'big_latency_p50',
        'swarm_hallucination_rate', 'big_hallucination_rate',
        'swarm_estimated_cost', 'big_estimated_cost'
    ]
    
    # Filter to metrics that exist
    available_metrics = [m for m in key_metrics if m in combined_df.columns]
    
    if available_metrics:
        display_df = combined_df[['run_id'] + available_metrics]
        print("\n", display_df.to_string(index=False))
    
    # Compute averages across runs
    print("\n" + "="*100)
    print("AVERAGE ACROSS ALL RUNS")
    print("="*100)
    
    numeric_cols = combined_df.select_dtypes(include='number').columns
    avg_metrics = combined_df[numeric_cols].mean()
    
    print("\n## Swarm Performance")
    print(f"  ROUGE-1 F1: {avg_metrics.get('swarm_rouge1_f1', 0):.4f}")
    print(f"  ROUGE-L F1: {avg_metrics.get('swarm_rougeL_f1', 0):.4f}")
    print(f"  BERTScore F1: {avg_metrics.get('swarm_bertscore_f1', 0):.4f}")
    print(f"  Latency p50: {avg_metrics.get('swarm_latency_p50', 0):.2f}s")
    print(f"  Hallucination Rate: {avg_metrics.get('swarm_hallucination_rate', 0):.2%}")
    print(f"  Estimated Cost: ${avg_metrics.get('swarm_estimated_cost', 0):.4f}")
    
    print("\n## Big Model Performance")
    print(f"  ROUGE-1 F1: {avg_metrics.get('big_rouge1_f1', 0):.4f}")
    print(f"  ROUGE-L F1: {avg_metrics.get('big_rougeL_f1', 0):.4f}")
    print(f"  BERTScore F1: {avg_metrics.get('big_bertscore_f1', 0):.4f}")
    print(f"  Latency p50: {avg_metrics.get('big_latency_p50', 0):.2f}s")
    print(f"  Hallucination Rate: {avg_metrics.get('big_hallucination_rate', 0):.2%}")
    print(f"  Estimated Cost: ${avg_metrics.get('big_estimated_cost', 0):.4f}")
    
    print("\n## Comparison (Swarm vs Big Model)")
    swarm_rouge = avg_metrics.get('swarm_rougeL_f1', 0)
    big_rouge = avg_metrics.get('big_rougeL_f1', 0)
    rouge_diff = ((swarm_rouge - big_rouge) / big_rouge * 100) if big_rouge > 0 else 0
    print(f"  Quality (ROUGE-L): {rouge_diff:+.1f}% {'(Swarm wins)' if rouge_diff > 0 else '(Big wins)'}")
    
    swarm_latency = avg_metrics.get('swarm_latency_p50', 0)
    big_latency = avg_metrics.get('big_latency_p50', 0)
    latency_diff = ((swarm_latency - big_latency) / big_latency * 100) if big_latency > 0 else 0
    print(f"  Speed (Latency): {latency_diff:+.1f}% {'(Swarm slower)' if latency_diff > 0 else '(Swarm faster)'}")
    
    swarm_cost = avg_metrics.get('swarm_estimated_cost', 0)
    big_cost = avg_metrics.get('big_estimated_cost', 0)
    cost_diff = ((swarm_cost - big_cost) / big_cost * 100) if big_cost > 0 else 0
    print(f"  Cost: {cost_diff:+.1f}% {'(Swarm more expensive)' if cost_diff > 0 else '(Swarm cheaper)'}")
    
    print("\n" + "="*100)
    
    # Save combined metrics
    combined_file = results_path / "combined_metrics.csv"
    combined_df.to_csv(combined_file, index=False)
    print(f"\nCombined metrics saved to: {combined_file}")


def main():
    parser = argparse.ArgumentParser(description="Aggregate results from multiple evaluation runs")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="../results",
        help="Path to results directory"
    )
    
    args = parser.parse_args()
    aggregate_results(args.results_dir)


if __name__ == "__main__":
    main()


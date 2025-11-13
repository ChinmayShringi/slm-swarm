#!/usr/bin/env python3
"""
Compare results from multiple consensus experiments

Analyzes:
1. Quality: ROUGE/BERTScore across different swarm types
2. Robustness: Outlier detection rates
3. Consensus strength: Average similarity scores
4. Cost-quality tradeoff

Usage:
    python compare_consensus_experiments.py [--results-dir ../results]
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List
import pandas as pd
import numpy as np


def load_experiment_results(results_dir: Path) -> List[Dict]:
    """Load all experiment results from results directory"""
    
    experiments = []
    
    # Find all run directories
    run_dirs = sorted([d for d in results_dir.iterdir() if d.is_dir() and d.name.startswith("run_")])
    
    if not run_dirs:
        print(f"No evaluation runs found in {results_dir}")
        return []
    
    print(f"Found {len(run_dirs)} runs")
    
    for run_dir in run_dirs:
        # Load experiment name if available
        experiment_name_file = run_dir / "experiment_name.txt"
        if experiment_name_file.exists():
            experiment_name = experiment_name_file.read_text().strip()
        else:
            experiment_name = run_dir.name
        
        # Load metrics
        metrics_file = run_dir / "metrics.csv"
        if not metrics_file.exists():
            continue
        
        try:
            df = pd.read_csv(metrics_file)
            metrics = df.iloc[0].to_dict()
            
            experiments.append({
                "name": experiment_name,
                "run_id": run_dir.name,
                "metrics": metrics
            })
        except Exception as e:
            print(f"Error loading {metrics_file}: {e}")
    
    return experiments


def categorize_experiments(experiments: List[Dict]) -> Dict[str, List[Dict]]:
    """Categorize experiments by type"""
    
    categorized = {
        "heterogeneous": [],
        "homogeneous": [],
        "consensus_methods": {}
    }
    
    for exp in experiments:
        name = exp["name"].lower()
        
        # Categorize by swarm type
        if "heterogeneous" in name:
            categorized["heterogeneous"].append(exp)
        elif "homogeneous" in name:
            categorized["homogeneous"].append(exp)
        
        # Categorize by consensus method
        if "cosine" in name:
            if "cosine" not in categorized["consensus_methods"]:
                categorized["consensus_methods"]["cosine"] = []
            categorized["consensus_methods"]["cosine"].append(exp)
        elif "judge" in name:
            if "judge" not in categorized["consensus_methods"]:
                categorized["consensus_methods"]["judge"] = []
            categorized["consensus_methods"]["judge"].append(exp)
        elif "voting" in name:
            if "voting" not in categorized["consensus_methods"]:
                categorized["consensus_methods"]["voting"] = []
            categorized["consensus_methods"]["voting"].append(exp)
    
    return categorized


def generate_comparison_table(experiments: List[Dict]) -> str:
    """Generate comparison table for paper"""
    
    rows = []
    
    for exp in experiments:
        name = exp["name"]
        m = exp["metrics"]
        
        # Extract model info from name
        if "Qwen" in name:
            models = "4x Qwen-7B"
        elif "Llama" in name:
            models = "4x Llama-8B"
        elif "Mistral" in name:
            models = "4x Mistral-7B"
        elif "Phi" in name:
            models = "4x Phi-3.8B"
        elif "Heterogeneous" in name:
            models = "Qwen+Llama+Mistral+Phi"
        else:
            models = "Mixed"
        
        # Determine swarm type
        swarm_type = "Heterogeneous" if "Heterogeneous" in name else "Homogeneous"
        
        row = {
            "Experiment": name[:40],  # Truncate for display
            "Swarm Type": swarm_type,
            "Models": models,
            "ROUGE-L": m.get("swarm_rougeL_f1", 0),
            "BERTScore": m.get("swarm_bertscore_f1", 0),
            "Outlier Rate": m.get("swarm_outlier_detected_rate", 0),
            "Consensus Sim": m.get("swarm_consensus_avg_similarity", 0),
            "Latency p50": m.get("swarm_latency_p50", 0),
            "Halluc Rate": m.get("swarm_hallucination_rate", 0)
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Format for display
    df["ROUGE-L"] = df["ROUGE-L"].map("{:.4f}".format)
    df["BERTScore"] = df["BERTScore"].map("{:.4f}".format)
    df["Outlier Rate"] = df["Outlier Rate"].map("{:.2%}".format)
    df["Consensus Sim"] = df["Consensus Sim"].map("{:.4f}".format)
    df["Latency p50"] = df["Latency p50"].map("{:.2f}s".format)
    df["Halluc Rate"] = df["Halluc Rate"].map("{:.2%}".format)
    
    return df.to_string(index=False)


def analyze_homogeneous_vs_heterogeneous(categorized: Dict) -> str:
    """Compare homogeneous vs heterogeneous swarms"""
    
    homo = categorized["homogeneous"]
    hetero = categorized["heterogeneous"]
    
    if not homo or not hetero:
        return "Insufficient data for homogeneous vs heterogeneous comparison"
    
    # Average metrics across homogeneous experiments
    homo_metrics = {
        "rouge_l": np.mean([e["metrics"].get("swarm_rougeL_f1", 0) for e in homo]),
        "bert": np.mean([e["metrics"].get("swarm_bertscore_f1", 0) for e in homo]),
        "outlier_rate": np.mean([e["metrics"].get("swarm_outlier_detected_rate", 0) for e in homo]),
        "consensus_sim": np.mean([e["metrics"].get("swarm_consensus_avg_similarity", 0) for e in homo]),
        "latency": np.mean([e["metrics"].get("swarm_latency_p50", 0) for e in homo]),
        "halluc": np.mean([e["metrics"].get("swarm_hallucination_rate", 0) for e in homo])
    }
    
    # Average metrics across heterogeneous experiments
    hetero_metrics = {
        "rouge_l": np.mean([e["metrics"].get("swarm_rougeL_f1", 0) for e in hetero]),
        "bert": np.mean([e["metrics"].get("swarm_bertscore_f1", 0) for e in hetero]),
        "outlier_rate": np.mean([e["metrics"].get("swarm_outlier_detected_rate", 0) for e in hetero]),
        "consensus_sim": np.mean([e["metrics"].get("swarm_consensus_avg_similarity", 0) for e in hetero]),
        "latency": np.mean([e["metrics"].get("swarm_latency_p50", 0) for e in hetero]),
        "halluc": np.mean([e["metrics"].get("swarm_hallucination_rate", 0) for e in hetero])
    }
    
    report = f"""
## Homogeneous vs Heterogeneous Swarms

### Homogeneous Swarms (n={len(homo)})
- **Quality**: ROUGE-L = {homo_metrics['rouge_l']:.4f}, BERTScore = {homo_metrics['bert']:.4f}
- **Robustness**: Outlier rate = {homo_metrics['outlier_rate']:.2%}
- **Consensus**: Avg similarity = {homo_metrics['consensus_sim']:.4f}
- **Speed**: Latency p50 = {homo_metrics['latency']:.2f}s
- **Reliability**: Hallucination rate = {homo_metrics['halluc']:.2%}

### Heterogeneous Swarms (n={len(hetero)})
- **Quality**: ROUGE-L = {hetero_metrics['rouge_l']:.4f}, BERTScore = {hetero_metrics['bert']:.4f}
- **Robustness**: Outlier rate = {hetero_metrics['outlier_rate']:.2%}
- **Consensus**: Avg similarity = {hetero_metrics['consensus_sim']:.4f}
- **Speed**: Latency p50 = {hetero_metrics['latency']:.2f}s
- **Reliability**: Hallucination rate = {hetero_metrics['halluc']:.2%}

### Differences (Hetero - Homo)
- Quality (ROUGE-L): {(hetero_metrics['rouge_l'] - homo_metrics['rouge_l']):.4f} ({((hetero_metrics['rouge_l'] - homo_metrics['rouge_l']) / homo_metrics['rouge_l'] * 100):+.1f}%)
- Outlier detection: {(hetero_metrics['outlier_rate'] - homo_metrics['outlier_rate']):.2%} ({((hetero_metrics['outlier_rate'] - homo_metrics['outlier_rate']) / (homo_metrics['outlier_rate'] + 0.001) * 100):+.1f}%)
- Consensus strength: {(hetero_metrics['consensus_sim'] - homo_metrics['consensus_sim']):.4f}

### Key Findings:
1. {'Heterogeneous' if hetero_metrics['rouge_l'] > homo_metrics['rouge_l'] else 'Homogeneous'} swarms achieve better quality
2. {'Heterogeneous' if hetero_metrics['outlier_rate'] > homo_metrics['outlier_rate'] else 'Homogeneous'} swarms detect more outliers (higher diversity)
3. {'Homogeneous' if homo_metrics['consensus_sim'] > hetero_metrics['consensus_sim'] else 'Heterogeneous'} swarms have stronger consensus
4. Cost-benefit: {'Heterogeneous diversity worth the complexity' if hetero_metrics['rouge_l'] > homo_metrics['rouge_l'] else 'Homogeneous simplicity preferred'}
"""
    
    return report


def analyze_consensus_methods(categorized: Dict) -> str:
    """Compare different consensus methods"""
    
    methods = categorized["consensus_methods"]
    
    if not methods:
        return "No consensus method comparisons available"
    
    report = "\n## Consensus Method Comparison\n\n"
    
    for method_name, exps in methods.items():
        if not exps:
            continue
        
        avg_rouge = np.mean([e["metrics"].get("swarm_rougeL_f1", 0) for e in exps])
        avg_latency = np.mean([e["metrics"].get("swarm_latency_p50", 0) for e in exps])
        
        report += f"### {method_name.capitalize()} Method (n={len(exps)})\n"
        report += f"- Avg ROUGE-L: {avg_rouge:.4f}\n"
        report += f"- Avg Latency: {avg_latency:.2f}s\n\n"
    
    return report


def main():
    parser = argparse.ArgumentParser(description="Compare consensus experiment results")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="../results",
        help="Path to results directory"
    )
    
    args = parser.parse_args()
    results_dir = Path(args.results_dir)
    
    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        return
    
    print("="*80)
    print("CONSENSUS EXPERIMENTS COMPARISON")
    print("="*80)
    print()
    
    # Load all experiments
    experiments = load_experiment_results(results_dir)
    
    if not experiments:
        print("No experiments found")
        return
    
    print(f"Loaded {len(experiments)} experiments\n")
    
    # Categorize experiments
    categorized = categorize_experiments(experiments)
    
    # Generate overall comparison table
    print("## Overall Comparison Table")
    print("="*80)
    print(generate_comparison_table(experiments))
    print()
    
    # Analyze homogeneous vs heterogeneous
    print("="*80)
    print(analyze_homogeneous_vs_heterogeneous(categorized))
    print("="*80)
    
    # Analyze consensus methods
    print(analyze_consensus_methods(categorized))
    print("="*80)
    
    # Save to file
    report_file = results_dir / "consensus_comparison.md"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("# Consensus Experiments Comparison\n\n")
        f.write("## Overall Results\n\n")
        f.write("```\n")
        f.write(generate_comparison_table(experiments))
        f.write("\n```\n\n")
        f.write(analyze_homogeneous_vs_heterogeneous(categorized))
        f.write("\n")
        f.write(analyze_consensus_methods(categorized))
    
    print(f"\n✓ Comparison report saved to: {report_file}")
    print()


if __name__ == "__main__":
    main()


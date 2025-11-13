import os
import json
import time
import uuid
from typing import Dict, List, Any
from pathlib import Path

import requests
import pandas as pd
import numpy as np
from tqdm import tqdm
from rouge_score import rouge_scorer
from bert_score import score as bert_score
import spacy

# Load spaCy for entity recognition
def load_spacy_model():
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        print("spaCy model not found. Downloading...")
        try:
            import subprocess
            subprocess.check_call(["python", "-m", "spacy", "download", "en_core_web_sm"])
            return spacy.load("en_core_web_sm")
        except Exception as e:
            print(f"Warning: Could not download spaCy model: {e}")
            print("Factuality checks will be limited.")
            return None

nlp = load_spacy_model()

# Configuration
COORDINATOR_URL = os.getenv("COORDINATOR_URL", "http://coordinator:8000")
DATASET_PATH = os.getenv("DATASET_PATH", "/app/dataset/messages.jsonl")
RESULTS_DIR = os.getenv("RESULTS_DIR", "/app/results")
ENABLE_BIG_MODEL = os.getenv("ENABLE_BIG_MODEL", "true").lower() == "true"


def load_dataset(path: str) -> List[Dict[str, Any]]:
    """Load dataset from JSONL file"""
    dataset = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                dataset.append(json.loads(line))
    return dataset


def call_endpoint(endpoint: str, payload: Dict[str, Any], timeout: int = 150) -> tuple[Dict[str, Any], float]:
    """Call coordinator endpoint and return (response, latency)"""
    url = f"{COORDINATOR_URL}{endpoint}"
    t0 = time.time()
    
    try:
        response = requests.post(url, json=payload, timeout=timeout)
        response.raise_for_status()
        latency = time.time() - t0
        return response.json(), latency
    except requests.exceptions.Timeout:
        latency = time.time() - t0
        return {"error": "timeout"}, latency
    except requests.exceptions.RequestException as e:
        latency = time.time() - t0
        return {"error": str(e)}, latency


def extract_entities(text: str) -> set:
    """Extract named entities from text using spaCy"""
    if nlp is None:
        return set()  # Return empty set if spaCy is not available
    
    try:
        doc = nlp(text)
        entities = set()
        for ent in doc.ents:
            if ent.label_ in ['PERSON', 'ORG', 'GPE', 'DATE', 'MONEY', 'PRODUCT']:
                entities.add(ent.text.lower())
        return entities
    except Exception as e:
        print(f"Warning: Entity extraction failed: {e}")
        return set()


def check_factuality(source: str, summary: str) -> Dict[str, Any]:
    """
    Simple factuality checks without LLM judge (reproducible):
    1. Entity consistency - summary should not introduce entities not in source
    2. Content word overlap - low overlap suggests hallucination
    """
    # Entity consistency check
    source_entities = extract_entities(source)
    summary_entities = extract_entities(summary)
    
    # Find entities in summary but not in source (potential hallucinations)
    hallucinated_entities = summary_entities - source_entities
    
    # Content word overlap (simple proxy)
    source_words = set(word.lower() for word in source.split() if len(word) > 3)
    summary_words = set(word.lower() for word in summary.split() if len(word) > 3)
    
    if len(summary_words) > 0:
        overlap_ratio = len(source_words & summary_words) / len(summary_words)
    else:
        overlap_ratio = 0.0
    
    # Flag as potential hallucination if:
    # - Introduces entities not in source, OR
    # - Very low content word overlap (< 30%)
    is_hallucination = len(hallucinated_entities) > 0 or overlap_ratio < 0.3
    
    return {
        "hallucinated_entities": list(hallucinated_entities),
        "content_overlap_ratio": overlap_ratio,
        "flagged_as_hallucination": is_hallucination
    }


def compute_rouge_scores(references: List[str], hypotheses: List[str]) -> Dict[str, float]:
    """Compute ROUGE-1, ROUGE-2, ROUGE-L scores"""
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
    
    for ref, hyp in zip(references, hypotheses):
        score = scorer.score(ref, hyp)
        for key in scores:
            scores[key].append(score[key].fmeasure)
    
    # Return average scores
    return {f"{key}_f1": sum(vals) / len(vals) for key, vals in scores.items()}


def compute_bertscore(references: List[str], hypotheses: List[str]) -> Dict[str, float]:
    """Compute BERTScore F1"""
    try:
        P, R, F1 = bert_score(hypotheses, references, lang="en", rescale_with_baseline=True, verbose=False)
        return {
            "bertscore_precision": float(P.mean()),
            "bertscore_recall": float(R.mean()),
            "bertscore_f1": float(F1.mean())
        }
    except Exception as e:
        print(f"BERTScore computation failed: {e}")
        return {
            "bertscore_precision": 0.0,
            "bertscore_recall": 0.0,
            "bertscore_f1": 0.0
        }


def estimate_cost(text: str, model_type: str) -> float:
    """
    Estimate cost based on token count and model pricing
    Rough estimates (as of 2024):
    - Small models (7-8B): ~$0.0001 per 1K tokens
    - Large models (70B): ~$0.0008 per 1K tokens
    """
    # Simple token estimation: ~0.75 tokens per word
    token_count = len(text.split()) * 0.75
    
    if model_type == "swarm":
        # 3 small models + 1 judge
        cost_per_1k = 0.0001
        return (token_count / 1000) * cost_per_1k * 4
    elif model_type == "big":
        cost_per_1k = 0.0008
        return (token_count / 1000) * cost_per_1k
    else:
        return 0.0


def run_evaluation():
    """Main evaluation loop"""
    print(f"Loading dataset from {DATASET_PATH}...")
    dataset = load_dataset(DATASET_PATH)
    print(f"Loaded {len(dataset)} examples")
    
    # Wait for coordinator to be ready
    print("\nWaiting for coordinator to be ready...")
    max_retries = 30
    for attempt in range(max_retries):
        try:
            health = requests.get(f"{COORDINATOR_URL}/health", timeout=5)
            health.raise_for_status()
            print(f"✓ Coordinator is ready: {health.json()}")
            break
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"  Attempt {attempt + 1}/{max_retries}: Coordinator not ready yet, waiting 5s...")
                time.sleep(5)
            else:
                print(f"ERROR: Coordinator did not become ready after {max_retries} attempts")
                print(f"Last error: {e}")
                return
    
    # Create results directory
    run_id = uuid.uuid4().hex[:8]
    run_dir = Path(RESULTS_DIR) / f"run_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results will be saved to: {run_dir}")
    
    # Collect results
    results = []
    swarm_summaries = []
    big_summaries = []
    references = []
    
    print("\nRunning evaluation...")
    for item in tqdm(dataset, desc="Processing"):
        payload = {
            "id": item["id"],
            "message": item["message"],
            "mode": "tldr"
        }
        
        # Get reference
        ref_tldr = item.get("reference", {}).get("tldr", "")
        references.append(ref_tldr)
        
        # Call swarm endpoint
        swarm_response, swarm_latency = call_endpoint("/summarize", payload)
        
        if "error" in swarm_response:
            swarm_summary = f"ERROR: {swarm_response['error']}"
            swarm_candidates = []
            consensus_metadata = {}
        else:
            swarm_summary = swarm_response.get("swarm_summary", "")
            swarm_candidates = swarm_response.get("candidates", [])
            consensus_metadata = swarm_response.get("consensus_metadata", {})
        
        swarm_summaries.append(swarm_summary)
        
        # Call big model endpoint (if enabled)
        if ENABLE_BIG_MODEL:
            big_response, big_latency = call_endpoint("/summarize_big", payload)
            
            if "error" in big_response:
                big_summary = f"ERROR: {big_response['error']}"
            else:
                big_summary = big_response.get("summary", "")
        else:
            big_summary = "DISABLED"
            big_latency = 0.0
        
        big_summaries.append(big_summary)
        
        # Compute factuality for both
        swarm_factuality = check_factuality(item["message"], swarm_summary) if not swarm_summary.startswith("ERROR") else {}
        big_factuality = check_factuality(item["message"], big_summary) if not big_summary.startswith("ERROR") else {}
        
        # Store result
        result = {
            "id": item["id"],
            "message": item["message"],
            "reference": ref_tldr,
            "swarm_summary": swarm_summary,
            "swarm_latency": swarm_latency,
            "swarm_candidates": swarm_candidates,
            "swarm_factuality": swarm_factuality,
            "consensus_metadata": consensus_metadata,
            "big_summary": big_summary,
            "big_latency": big_latency,
            "big_factuality": big_factuality
        }
        results.append(result)
    
    # Save detailed outputs
    outputs_file = run_dir / "outputs.jsonl"
    with open(outputs_file, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")
    print(f"\nDetailed outputs saved to: {outputs_file}")
    
    # Filter out errors for metric computation
    valid_swarm = [(s, r) for s, r in zip(swarm_summaries, references) if not s.startswith("ERROR")]
    valid_big = [(s, r) for s, r in zip(big_summaries, references) if not s.startswith("ERROR") and s != "DISABLED"]
    
    # Compute aggregate metrics
    print("\nComputing aggregate metrics...")
    
    metrics = {}
    
    # ROUGE scores
    if valid_swarm:
        swarm_refs = [r for _, r in valid_swarm]
        swarm_hyps = [s for s, _ in valid_swarm]
        rouge_swarm = compute_rouge_scores(swarm_refs, swarm_hyps)
        metrics.update({f"swarm_{k}": v for k, v in rouge_swarm.items()})
    
    if valid_big:
        big_refs = [r for _, r in valid_big]
        big_hyps = [s for s, _ in valid_big]
        rouge_big = compute_rouge_scores(big_refs, big_hyps)
        metrics.update({f"big_{k}": v for k, v in rouge_big.items()})
    
    # BERTScore
    if valid_swarm:
        bert_swarm = compute_bertscore([r for _, r in valid_swarm], [s for s, _ in valid_swarm])
        metrics.update({f"swarm_{k}": v for k, v in bert_swarm.items()})
    
    if valid_big:
        bert_big = compute_bertscore([r for _, r in valid_big], [s for s, _ in valid_big])
        metrics.update({f"big_{k}": v for k, v in bert_big.items()})
    
    # Latency percentiles
    swarm_latencies = [r["swarm_latency"] for r in results if r["swarm_latency"] > 0]
    big_latencies = [r["big_latency"] for r in results if r["big_latency"] > 0]
    
    if swarm_latencies:
        metrics["swarm_latency_p50"] = float(pd.Series(swarm_latencies).quantile(0.5))
        metrics["swarm_latency_p95"] = float(pd.Series(swarm_latencies).quantile(0.95))
        metrics["swarm_latency_mean"] = float(pd.Series(swarm_latencies).mean())
    
    if big_latencies:
        metrics["big_latency_p50"] = float(pd.Series(big_latencies).quantile(0.5))
        metrics["big_latency_p95"] = float(pd.Series(big_latencies).quantile(0.95))
        metrics["big_latency_mean"] = float(pd.Series(big_latencies).mean())
    
    # Factuality metrics
    swarm_hallucinations = [r["swarm_factuality"].get("flagged_as_hallucination", False) 
                           for r in results if r.get("swarm_factuality")]
    big_hallucinations = [r["big_factuality"].get("flagged_as_hallucination", False) 
                         for r in results if r.get("big_factuality")]
    
    if swarm_hallucinations:
        metrics["swarm_hallucination_rate"] = sum(swarm_hallucinations) / len(swarm_hallucinations)
    
    if big_hallucinations:
        metrics["big_hallucination_rate"] = sum(big_hallucinations) / len(big_hallucinations)
    
    # Consensus metrics (from consensus_metadata)
    consensus_scores = []
    outlier_detected_count = 0
    consensus_confidences = []
    consensus_similarities = []
    
    for r in results:
        meta = r.get("consensus_metadata", {})
        if meta and meta.get("method") == "cosine":
            if "consensus_scores" in meta:
                consensus_scores.append(meta["consensus_scores"])
            if meta.get("outlier_detected"):
                outlier_detected_count += 1
            if "consensus_confidence" in meta:
                consensus_confidences.append(meta["consensus_confidence"])
            if "avg_consensus_similarity" in meta:
                consensus_similarities.append(meta["avg_consensus_similarity"])
    
    if consensus_similarities:
        metrics["swarm_consensus_avg_similarity"] = float(np.mean(consensus_similarities))
    
    if results:
        metrics["swarm_outlier_detected_rate"] = outlier_detected_count / len(results)
    
    if consensus_confidences:
        metrics["swarm_consensus_confidence"] = float(np.mean(consensus_confidences))
    
    # Cost estimates
    total_tokens_swarm = sum(len((r["message"] + r["swarm_summary"]).split()) for r in results) * 0.75
    total_tokens_big = sum(len((r["message"] + r["big_summary"]).split()) for r in results if r["big_summary"] != "DISABLED") * 0.75
    
    metrics["swarm_estimated_cost"] = (total_tokens_swarm / 1000) * 0.0001 * 4  # 3 workers + judge
    metrics["big_estimated_cost"] = (total_tokens_big / 1000) * 0.0008
    
    # Sample counts
    metrics["total_samples"] = len(dataset)
    metrics["swarm_valid_samples"] = len(valid_swarm)
    metrics["big_valid_samples"] = len(valid_big)
    
    # Save metrics CSV
    metrics_file = run_dir / "metrics.csv"
    pd.DataFrame([metrics]).to_csv(metrics_file, index=False)
    print(f"Aggregate metrics saved to: {metrics_file}")
    
    # Generate summary markdown
    summary_md = generate_summary_markdown(metrics, run_id)
    summary_file = run_dir / "summary.md"
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write(summary_md)
    print(f"Summary report saved to: {summary_file}")
    
    # Print summary to console
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    print(summary_md)
    print("="*80)


def generate_summary_markdown(metrics: Dict[str, float], run_id: str) -> str:
    """Generate human-readable markdown summary"""
    md = f"""# Evaluation Summary - Run {run_id}

## Dataset
- Total samples: {metrics.get('total_samples', 0)}
- Swarm valid samples: {metrics.get('swarm_valid_samples', 0)}
- Big model valid samples: {metrics.get('big_valid_samples', 0)}

## ROUGE Scores (F1)

| Metric | Swarm | Big Model |
|--------|-------|-----------|
| ROUGE-1 | {metrics.get('swarm_rouge1_f1', 0):.4f} | {metrics.get('big_rouge1_f1', 0):.4f} |
| ROUGE-2 | {metrics.get('swarm_rouge2_f1', 0):.4f} | {metrics.get('big_rouge2_f1', 0):.4f} |
| ROUGE-L | {metrics.get('swarm_rougeL_f1', 0):.4f} | {metrics.get('big_rougeL_f1', 0):.4f} |

## BERTScore

| Metric | Swarm | Big Model |
|--------|-------|-----------|
| Precision | {metrics.get('swarm_bertscore_precision', 0):.4f} | {metrics.get('big_bertscore_precision', 0):.4f} |
| Recall | {metrics.get('swarm_bertscore_recall', 0):.4f} | {metrics.get('big_bertscore_recall', 0):.4f} |
| F1 | {metrics.get('swarm_bertscore_f1', 0):.4f} | {metrics.get('big_bertscore_f1', 0):.4f} |

## Latency (seconds)

| Metric | Swarm | Big Model |
|--------|-------|-----------|
| Mean | {metrics.get('swarm_latency_mean', 0):.2f} | {metrics.get('big_latency_mean', 0):.2f} |
| Median (p50) | {metrics.get('swarm_latency_p50', 0):.2f} | {metrics.get('big_latency_p50', 0):.2f} |
| p95 | {metrics.get('swarm_latency_p95', 0):.2f} | {metrics.get('big_latency_p95', 0):.2f} |

## Factuality

| Metric | Swarm | Big Model |
|--------|-------|-----------|
| Hallucination Rate | {metrics.get('swarm_hallucination_rate', 0):.2%} | {metrics.get('big_hallucination_rate', 0):.2%} |

## Consensus Metrics (Swarm Only)

| Metric | Value |
|--------|-------|
| Avg Consensus Similarity | {metrics.get('swarm_consensus_avg_similarity', 0):.4f} |
| Outlier Detection Rate | {metrics.get('swarm_outlier_detected_rate', 0):.2%} |
| Consensus Confidence | {metrics.get('swarm_consensus_confidence', 0):.4f} |

## Cost Estimate (USD)

| System | Estimated Cost |
|--------|---------------|
| Swarm | ${metrics.get('swarm_estimated_cost', 0):.4f} |
| Big Model | ${metrics.get('big_estimated_cost', 0):.4f} |

## Key Findings

1. **Quality**: {'Swarm' if metrics.get('swarm_rougeL_f1', 0) > metrics.get('big_rougeL_f1', 0) else 'Big Model'} achieves higher ROUGE-L F1 score
2. **Speed**: {'Swarm' if metrics.get('swarm_latency_p50', 999) < metrics.get('big_latency_p50', 999) else 'Big Model'} is faster (median latency)
3. **Cost**: {'Swarm' if metrics.get('swarm_estimated_cost', 999) < metrics.get('big_estimated_cost', 999) else 'Big Model'} is more cost-effective
4. **Reliability**: {'Swarm' if metrics.get('swarm_hallucination_rate', 1) < metrics.get('big_hallucination_rate', 1) else 'Big Model'} has lower hallucination rate

---

*Generated: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}*
"""
    return md


if __name__ == "__main__":
    print("="*80)
    print("SLM Swarm Evaluator")
    print("="*80)
    print(f"Coordinator URL: {COORDINATOR_URL}")
    print(f"Dataset: {DATASET_PATH}")
    print(f"Results directory: {RESULTS_DIR}")
    print(f"Big model enabled: {ENABLE_BIG_MODEL}")
    print("="*80 + "\n")
    
    run_evaluation()


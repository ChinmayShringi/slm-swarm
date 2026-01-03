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
RUN_ID = os.getenv("RUN_ID", "")  # Optional: specify run ID to resume
CHECKPOINT_INTERVAL = int(os.getenv("CHECKPOINT_INTERVAL", "10"))  # Save checkpoint every N examples


def load_dataset(path: str) -> List[Dict[str, Any]]:
    """Load dataset from JSONL file"""
    dataset = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                dataset.append(json.loads(line))
    return dataset


def call_endpoint(endpoint: str, payload: Dict[str, Any], timeout: int = None) -> tuple[Dict[str, Any], float]:
    """Call coordinator endpoint and return (response, latency). No timeout - let it work naturally."""
    url = f"{COORDINATOR_URL}{endpoint}"
    t0 = time.time()
    
    try:
        # No timeout parameter - let requests work naturally
        response = requests.post(url, json=payload)
        response.raise_for_status()
        latency = time.time() - t0
        return response.json(), latency
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


def load_checkpoint(run_dir: Path) -> set:
    """Load already processed IDs from checkpoint"""
    outputs_file = run_dir / "outputs.jsonl"
    processed_ids = set()
    
    if outputs_file.exists():
        try:
            with open(outputs_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        processed_ids.add(data["id"])
            print(f"✓ Loaded checkpoint: {len(processed_ids)} examples already processed")
        except Exception as e:
            print(f"Warning: Could not load checkpoint: {e}")
    
    return processed_ids


def save_checkpoint(run_dir: Path, result: Dict[str, Any]):
    """Append single result to outputs file (checkpoint)"""
    outputs_file = run_dir / "outputs.jsonl"
    try:
        with open(outputs_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(result) + "\n")
    except Exception as e:
        print(f"Warning: Could not save checkpoint: {e}")


def run_evaluation():
    """Main evaluation loop with checkpoint/resume support"""
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
    
    # Get swarm size from coordinator health
    try:
        health = requests.get(f"{COORDINATOR_URL}/health", timeout=5)
        health_data = health.json()
        num_workers = health_data.get("workers", 0)
    except Exception as e:
        print(f"Warning: Could not get worker count from coordinator: {e}")
        num_workers = 0
    
    # Extract dataset name from path (e.g., "messages_small" from "dataset/messages_small.jsonl")
    dataset_name = Path(DATASET_PATH).stem  # Gets filename without extension
    
    # Create or resume run directory with new structure: results/<dataset_name>/<timestamp>_<workers>/
    if RUN_ID:
        run_id = RUN_ID
        print(f"Resuming run: {run_id}")
    else:
        timestamp = int(time.time())
        run_id = f"{timestamp}_{num_workers}"
        print(f"Starting new run: {run_id} (workers: {num_workers})")
    
    run_dir = Path(RESULTS_DIR) / dataset_name / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Load checkpoint (already processed IDs)
    processed_ids = load_checkpoint(run_dir)
    
    # Filter dataset to only unprocessed examples
    remaining_dataset = [item for item in dataset if item["id"] not in processed_ids]
    
    if len(remaining_dataset) == 0:
        print(f"✓ All examples already processed! Loading existing results for metrics...")
        dataset_to_process = []
    else:
        print(f"Processing {len(remaining_dataset)} remaining examples (out of {len(dataset)} total)")
        dataset_to_process = remaining_dataset
    
    print(f"Results will be saved to: {run_dir}")
    
    # Process remaining examples
    print("\nRunning evaluation...")
    examples_processed = 0
    
    for item in tqdm(dataset_to_process, desc="Processing"):
        payload = {
            "id": item["id"],
            "message": item["message"],
            "mode": "tldr"
        }
        
        # Get reference
        ref_tldr = item.get("reference", {}).get("tldr", "")
        
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
        
        # Save checkpoint incrementally
        save_checkpoint(run_dir, result)
        examples_processed += 1
        
        # Periodic status update
        if examples_processed % CHECKPOINT_INTERVAL == 0:
            print(f"\n[CHECKPOINT] Processed {examples_processed}/{len(dataset_to_process)} examples")
    
    print(f"\n✓ Processing complete: {examples_processed} new examples")
    
    # Load ALL results (including previously processed) for final metrics
    print("\nLoading all results for metrics calculation...")
    all_results = []
    outputs_file = run_dir / "outputs.jsonl"
    
    if outputs_file.exists():
        with open(outputs_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    all_results.append(json.loads(line))
    
    print(f"Total results loaded: {len(all_results)}")
    
    if len(all_results) == 0:
        print("ERROR: No results to process!")
        return
    
    # Extract data for metrics (use all_results, not just new ones)
    swarm_summaries = [r["swarm_summary"] for r in all_results]
    big_summaries = [r["big_summary"] for r in all_results]
    references = [r["reference"] for r in all_results]
    
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
    
    # Latency percentiles (use all_results)
    swarm_latencies = [r["swarm_latency"] for r in all_results if r["swarm_latency"] > 0]
    big_latencies = [r["big_latency"] for r in all_results if r["big_latency"] > 0]
    
    if swarm_latencies:
        metrics["swarm_latency_p50"] = float(pd.Series(swarm_latencies).quantile(0.5))
        metrics["swarm_latency_p95"] = float(pd.Series(swarm_latencies).quantile(0.95))
        metrics["swarm_latency_mean"] = float(pd.Series(swarm_latencies).mean())
    
    if big_latencies:
        metrics["big_latency_p50"] = float(pd.Series(big_latencies).quantile(0.5))
        metrics["big_latency_p95"] = float(pd.Series(big_latencies).quantile(0.95))
        metrics["big_latency_mean"] = float(pd.Series(big_latencies).mean())
    
    # Factuality metrics (use all_results)
    swarm_hallucinations = [r["swarm_factuality"].get("flagged_as_hallucination", False) 
                           for r in all_results if r.get("swarm_factuality")]
    big_hallucinations = [r["big_factuality"].get("flagged_as_hallucination", False) 
                         for r in all_results if r.get("big_factuality")]
    
    if swarm_hallucinations:
        metrics["swarm_hallucination_rate"] = sum(swarm_hallucinations) / len(swarm_hallucinations)
    
    if big_hallucinations:
        metrics["big_hallucination_rate"] = sum(big_hallucinations) / len(big_hallucinations)
    
    # Consensus metrics (from consensus_metadata) (use all_results)
    consensus_scores = []
    outlier_detected_count = 0
    consensus_confidences = []
    consensus_similarities = []
    
    for r in all_results:
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
    
    if all_results:
        metrics["swarm_outlier_detected_rate"] = outlier_detected_count / len(all_results)
    
    if consensus_confidences:
        metrics["swarm_consensus_confidence"] = float(np.mean(consensus_confidences))
    
    # Token statistics (measured, not cost estimates)
    total_words_input = sum(len(r["message"].split()) for r in all_results)
    total_words_output_swarm = sum(len(r["swarm_summary"].split()) for r in all_results if not r["swarm_summary"].startswith("ERROR"))
    total_words_output_big = sum(len(r["big_summary"].split()) for r in all_results if r["big_summary"] != "DISABLED" and not r["big_summary"].startswith("ERROR"))
    
    metrics["avg_input_words"] = total_words_input / len(all_results) if all_results else 0
    metrics["avg_output_words_swarm"] = total_words_output_swarm / len(all_results) if all_results else 0
    metrics["avg_output_words_big"] = total_words_output_big / len([r for r in all_results if r["big_summary"] != "DISABLED"]) if any(r["big_summary"] != "DISABLED" for r in all_results) else 0
    
    # Sample counts
    metrics["total_samples"] = len(all_results)
    metrics["swarm_valid_samples"] = len(valid_swarm)
    metrics["big_valid_samples"] = len(valid_big)
    metrics["examples_processed_this_run"] = examples_processed
    
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

## Computational Statistics

| Metric | Value |
|--------|-------|
| Avg Input Length (words) | {metrics.get('avg_input_words', 0):.1f} |
| Avg Output Length (words) | {metrics.get('avg_output_words_swarm', 0):.1f} |

## Key Findings

1. **Quality (ROUGE-L)**: {metrics.get('swarm_rougeL_f1', 0):.4f}
2. **Semantic Similarity (BERTScore)**: {metrics.get('swarm_bertscore_f1', 0):.4f}
3. **Consensus Strength**: {metrics.get('swarm_consensus_avg_similarity', 0):.4f}
4. **Throughput**: {metrics.get('total_samples', 0) / (metrics.get('swarm_latency_mean', 1) * metrics.get('total_samples', 1) / 60):.2f} examples/minute

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


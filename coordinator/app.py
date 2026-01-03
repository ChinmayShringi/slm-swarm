import os
import json
import time
import asyncio
from typing import Dict, List, Optional, Any, Tuple

import httpx
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel

app = FastAPI(title="SLM Swarm Coordinator")

# Load sentence transformer model for consensus (lightweight, fast)
print("Loading sentence transformer model for consensus...")
EMBEDDER = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
print("Model loaded successfully")

# Load configuration from environment
WORKERS = json.loads(os.getenv("WORKERS", '[]'))
MODELS = json.loads(os.getenv("MODELS", '[]'))
JUDGE = os.getenv("JUDGE", "")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "phi3:3.8b")

# Consensus configuration
CONSENSUS_METHOD = os.getenv("CONSENSUS_METHOD", "cosine")  # cosine, judge, or voting
CONSENSUS_THRESHOLD = float(os.getenv("CONSENSUS_THRESHOLD", "0.3"))
SWARM_TYPE = os.getenv("SWARM_TYPE", "heterogeneous")  # heterogeneous or homogeneous

# Temperature variation for homogeneous swarms (introduces diversity)
WORKER_TEMPERATURES_STR = os.getenv("WORKER_TEMPERATURES", "")
if WORKER_TEMPERATURES_STR:
    try:
        WORKER_TEMPERATURES = json.loads(WORKER_TEMPERATURES_STR)
    except:
        WORKER_TEMPERATURES = []
else:
    WORKER_TEMPERATURES = []

# If no temperatures specified, use default 0.2 for all workers
if not WORKER_TEMPERATURES:
    WORKER_TEMPERATURES = [0.2] * len(WORKERS)

# Adversarial testing configuration (for demonstrating outlier detection)
ADVERSARIAL_MODE = os.getenv("ADVERSARIAL_MODE", "false").lower() == "true"
ADVERSARIAL_WORKER_IDX = int(os.getenv("ADVERSARIAL_WORKER_IDX", "2"))
ADVERSARIAL_SUMMARY = os.getenv("ADVERSARIAL_SUMMARY", "This is completely incorrect information about unrelated topics.")

BIGMODEL_URL = os.getenv("BIGMODEL_URL", "")
BIGMODEL_MODEL = os.getenv("BIGMODEL_MODEL", "llama-3.1-70b-instruct")
BIGMODEL_TYPE = os.getenv("BIGMODEL_TYPE", "openai")
BIGMODEL_API_KEY = os.getenv("BIGMODEL_API_KEY", "")

# Unified prompt template for all models (critical for fair comparison)
# Simplified for small models
PROMPT_TEMPLATE = """Summarize this message in one short sentence:

{message}

Summary:"""

# Judge prompt to select best summary
JUDGE_PROMPT = """You are an evaluator. You will be given a MESSAGE and THREE candidate summaries.
Your task: Pick the BEST summary based on:
1. Faithfulness to source (no hallucinations)
2. Completeness (captures key points)
3. Brevity (concise)

Simply output the NUMBER (1, 2, or 3) of the best summary, nothing else.

MESSAGE:
{message}

CANDIDATE 1:
{candidate1}

CANDIDATE 2:
{candidate2}

CANDIDATE 3:
{candidate3}

Best summary number:"""


class SummarizeRequest(BaseModel):
    id: str
    message: str
    mode: str = "tldr"


class SummarizeResponse(BaseModel):
    id: str
    swarm_summary: str
    candidates: List[Dict[str, Any]]
    latency: float
    consensus_metadata: Optional[Dict[str, Any]] = None


class BigModelResponse(BaseModel):
    id: str
    summary: str
    latency: float


class CompareResponse(BaseModel):
    id: str
    swarm: Dict[str, Any]
    bigmodel: Dict[str, Any]


async def call_ollama(url: str, model: str, prompt: str, temperature: float = 0.2, max_tokens: int = 200) -> tuple[str, float]:
    """Call Ollama API and return (response_text, latency_seconds)"""
    t0 = time.time()

    # Use 600 second timeout for CPU inference (can take 3+ minutes per sample)
    async with httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=30.0)) as client:
        try:
            response = await client.post(
                f"{url}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens
                    }
                }
            )
            response.raise_for_status()
            data = response.json()
            text = data.get("response", "").strip()
            latency = time.time() - t0
            return text, latency
        except Exception as e:
            latency = time.time() - t0
            error_msg = str(e) if str(e) else type(e).__name__
            print(f"Error calling {url} with model {model}: {error_msg}")
            return f"ERROR: {error_msg}", latency


async def judge_best_summary(message: str, candidates: List[str]) -> int:
    """Use Phi-3 judge to select best summary. Returns index (0-2) of best candidate."""
    if len(candidates) != 3:
        return 0  # fallback to first if not exactly 3
    
    prompt = JUDGE_PROMPT.format(
        message=message,
        candidate1=candidates[0],
        candidate2=candidates[1],
        candidate3=candidates[2]
    )
    
    try:
        response, _ = await call_ollama(JUDGE, JUDGE_MODEL, prompt, temperature=0.1, max_tokens=10)
        # Parse judge response - looking for a number 1, 2, or 3
        for char in response:
            if char in ['1', '2', '3']:
                return int(char) - 1  # convert to 0-indexed
        return 0  # fallback
    except Exception:
        return 0  # fallback to first candidate


def select_consensus_cosine(summaries: List[str]) -> Tuple[int, Dict[str, Any]]:
    """
    Select best summary using cosine similarity consensus.
    
    Algorithm:
    1. Embed all summaries
    2. Compute pairwise cosine similarity matrix
    3. Calculate average similarity for each candidate
    4. Detect outlier (lowest avg similarity, below threshold)
    5. Return candidate with highest avg similarity (consensus centroid)
    
    Returns:
        (best_index, consensus_metadata)
    """
    if not summaries or len(summaries) < 2:
        return 0, {"method": "cosine", "error": "insufficient_candidates"}
    
    try:
        # Embed all summaries
        embeddings = EMBEDDER.encode(summaries, convert_to_numpy=True)
        
        # Compute pairwise cosine similarity matrix
        sim_matrix = cosine_similarity(embeddings)
        
        # Calculate average similarity for each candidate (excluding self-similarity)
        avg_similarities = []
        for i in range(len(summaries)):
            # Average of similarities with other candidates (exclude diagonal)
            other_sims = [sim_matrix[i][j] for j in range(len(summaries)) if i != j]
            avg_sim = np.mean(other_sims) if other_sims else 0.0
            avg_similarities.append(float(avg_sim))
        
        # Detect outlier
        min_sim = min(avg_similarities)
        mean_sim = np.mean(avg_similarities)
        
        outlier_detected = (mean_sim - min_sim) > CONSENSUS_THRESHOLD
        
        # Find best candidate (highest average similarity = consensus centroid)
        best_idx = int(np.argmax(avg_similarities))
        
        metadata = {
            "method": "cosine",
            "consensus_scores": avg_similarities,
            "similarity_matrix": sim_matrix.tolist(),
            "outlier_detected": bool(outlier_detected),
            "outlier_idx": int(np.argmin(avg_similarities)) if outlier_detected else None,
            "consensus_confidence": float(mean_sim - min_sim),
            "avg_consensus_similarity": float(mean_sim)
        }
        
        return best_idx, metadata
        
    except Exception as e:
        return 0, {"method": "cosine", "error": str(e)}


def select_consensus_voting(summaries: List[str]) -> Tuple[int, Dict[str, Any]]:
    """
    Simple majority voting based on exact string match.
    Returns most common summary.
    """
    if not summaries:
        return 0, {"method": "voting", "error": "no_candidates"}
    
    from collections import Counter
    counts = Counter(summaries)
    most_common_summary = counts.most_common(1)[0][0]
    best_idx = summaries.index(most_common_summary)
    
    metadata = {
        "method": "voting",
        "vote_counts": dict(counts),
        "winner_votes": counts[most_common_summary],
        "total_candidates": len(summaries)
    }
    
    return best_idx, metadata


async def select_best_summary(message: str, candidates: List[Dict[str, Any]]) -> Tuple[int, Dict[str, Any]]:
    """
    Select best summary based on configured consensus method.
    
    Returns:
        (best_index, consensus_metadata)
    """
    # Extract summary texts
    summaries = [c.get("summary", "") for c in candidates if not c.get("error", False)]
    
    if not summaries:
        return 0, {"method": CONSENSUS_METHOD, "error": "no_valid_candidates"}
    
    # Map back to original indices
    valid_indices = [i for i, c in enumerate(candidates) if not c.get("error", False)]
    
    if CONSENSUS_METHOD == "cosine":
        local_best_idx, metadata = select_consensus_cosine(summaries)
        best_idx = valid_indices[local_best_idx] if local_best_idx < len(valid_indices) else 0
    elif CONSENSUS_METHOD == "voting":
        local_best_idx, metadata = select_consensus_voting(summaries)
        best_idx = valid_indices[local_best_idx] if local_best_idx < len(valid_indices) else 0
    elif CONSENSUS_METHOD == "judge":
        # Pad summaries to 3 if needed for judge
        while len(summaries) < 3:
            summaries.append(summaries[0] if summaries else "")
        local_best_idx = await judge_best_summary(message, summaries[:3])
        best_idx = valid_indices[local_best_idx] if local_best_idx < len(valid_indices) else 0
        metadata = {"method": "judge", "selected_idx": local_best_idx}
    else:
        # Fallback: first candidate
        best_idx = valid_indices[0] if valid_indices else 0
        metadata = {"method": "fallback", "error": f"unknown_method_{CONSENSUS_METHOD}"}
    
    return best_idx, metadata


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "workers": len(WORKERS),
        "models": MODELS,
        "consensus_method": CONSENSUS_METHOD,
        "swarm_type": SWARM_TYPE,
        "judge_available": bool(JUDGE)
    }


@app.post("/summarize", response_model=SummarizeResponse)
async def summarize(request: SummarizeRequest):
    """
    Swarm approach: Fan out to 3 workers, collect candidates, judge picks best
    """
    t0 = time.time()
    
    if not WORKERS or not MODELS:
        raise HTTPException(status_code=500, detail="Workers not configured")
    
    # Build prompt
    prompt = PROMPT_TEMPLATE.format(message=request.message)
    
    # Fan out to all workers in parallel (with temperature variation if configured)
    tasks = []
    for i, (worker_url, model_name) in enumerate(zip(WORKERS, MODELS)):
        temp = WORKER_TEMPERATURES[i] if i < len(WORKER_TEMPERATURES) else 0.2
        tasks.append(call_ollama(worker_url, model_name, prompt, temperature=temp, max_tokens=150))
    
    # Collect responses
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    candidates = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            candidates.append({
                "model": MODELS[i] if i < len(MODELS) else "unknown",
                "summary": f"ERROR: {str(result)}",
                "latency": 0.0,
                "error": True
            })
        else:
            text, latency = result
            candidates.append({
                "model": MODELS[i] if i < len(MODELS) else "unknown",
                "summary": text,
                "latency": latency,
                "error": text.startswith("ERROR:")
            })
    
    # ADVERSARIAL MODE: Inject incorrect summary to test outlier detection
    if ADVERSARIAL_MODE and ADVERSARIAL_WORKER_IDX < len(candidates):
        candidates[ADVERSARIAL_WORKER_IDX]["summary"] = ADVERSARIAL_SUMMARY
        candidates[ADVERSARIAL_WORKER_IDX]["adversarial"] = True
        candidates[ADVERSARIAL_WORKER_IDX]["original_summary"] = candidates[ADVERSARIAL_WORKER_IDX].get("summary", "")
        print(f"[ADVERSARIAL MODE] Injected bad summary at worker {ADVERSARIAL_WORKER_IDX}")
    
    # Use consensus mechanism to select best summary
    if len(candidates) >= 2:
        best_idx, consensus_metadata = await select_best_summary(request.message, candidates)
        best_summary = candidates[best_idx]["summary"] if best_idx < len(candidates) else candidates[0]["summary"]
        
        # Add adversarial info to metadata if enabled
        if ADVERSARIAL_MODE:
            consensus_metadata["adversarial_enabled"] = True
            consensus_metadata["adversarial_worker_idx"] = ADVERSARIAL_WORKER_IDX
            consensus_metadata["adversarial_was_selected"] = (best_idx == ADVERSARIAL_WORKER_IDX)
    else:
        # Fallback: use first candidate
        best_summary = candidates[0]["summary"] if candidates else "ERROR: No responses"
        consensus_metadata = {"method": "fallback", "reason": "insufficient_candidates"}
    
    # Add swarm configuration metadata
    consensus_metadata["num_workers"] = len(WORKERS)
    consensus_metadata["swarm_type"] = SWARM_TYPE
    
    total_latency = time.time() - t0
    
    return SummarizeResponse(
        id=request.id,
        swarm_summary=best_summary,
        candidates=candidates,
        latency=total_latency,
        consensus_metadata=consensus_metadata
    )


async def bigmodel_summarize(prompt: str) -> tuple[str, float]:
    """Call big model via OpenAI-compatible API"""
    if not BIGMODEL_URL:
        raise HTTPException(status_code=500, detail="Big model not configured")
    
    t0 = time.time()
    
    async with httpx.AsyncClient(timeout=180.0) as client:
        try:
            headers = {}
            if BIGMODEL_API_KEY:
                headers["Authorization"] = f"Bearer {BIGMODEL_API_KEY}"
            
            if BIGMODEL_TYPE == "openai":
                # OpenAI-compatible API (vLLM, OpenAI, etc.)
                response = await client.post(
                    f"{BIGMODEL_URL}/chat/completions" if not BIGMODEL_URL.endswith("/chat/completions") else BIGMODEL_URL,
                    headers=headers,
                    json={
                        "model": BIGMODEL_MODEL,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.2,
                        "max_tokens": 200
                    }
                )
                response.raise_for_status()
                data = response.json()
                text = data["choices"][0]["message"]["content"].strip()
            elif BIGMODEL_TYPE == "anthropic":
                # Anthropic Claude API
                response = await client.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={
                        **headers,
                        "x-api-key": BIGMODEL_API_KEY,
                        "anthropic-version": "2023-06-01"
                    },
                    json={
                        "model": BIGMODEL_MODEL,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": 200,
                        "temperature": 0.2
                    }
                )
                response.raise_for_status()
                data = response.json()
                text = data["content"][0]["text"].strip()
            else:
                raise ValueError(f"Unknown BIGMODEL_TYPE: {BIGMODEL_TYPE}")
            
            latency = time.time() - t0
            return text, latency
            
        except Exception as e:
            latency = time.time() - t0
            return f"ERROR: {str(e)}", latency


@app.post("/summarize_big", response_model=BigModelResponse)
async def summarize_big(request: SummarizeRequest):
    """
    Big model baseline approach
    """
    prompt = PROMPT_TEMPLATE.format(message=request.message)
    summary, latency = await bigmodel_summarize(prompt)
    
    return BigModelResponse(
        id=request.id,
        summary=summary,
        latency=latency
    )


@app.post("/summarize_compare", response_model=CompareResponse)
async def summarize_compare(request: SummarizeRequest):
    """
    Compare swarm and big model side-by-side
    """
    # Run both in parallel
    swarm_task = summarize(request)
    big_task = summarize_big(request)
    
    swarm_result, big_result = await asyncio.gather(swarm_task, big_task, return_exceptions=True)
    
    # Handle errors
    if isinstance(swarm_result, Exception):
        swarm_data = {"error": str(swarm_result)}
    else:
        swarm_data = swarm_result.dict()
    
    if isinstance(big_result, Exception):
        big_data = {"error": str(big_result)}
    else:
        big_data = big_result.dict()
    
    return CompareResponse(
        id=request.id,
        swarm=swarm_data,
        bigmodel=big_data
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


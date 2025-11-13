# SLM Swarm Evaluation System

A Docker-based research platform for comparing heterogeneous small language model (SLM) swarms against large baseline models on text message summarization tasks, with IEEE-grade reproducibility.

## System Architecture

**Goal:** Evaluate whether a swarm of small models (Qwen-7B + Llama-8B + Mistral-7B + Phi-3 judge) can match or exceed the performance of a single large model (LLaMA-70B) on summarization tasks.

### Components

- **4 Ollama Workers**: Qwen2.5-7B, Llama-3.1-8B, Mistral-7B, Phi-3-3.8B (judge)
- **Coordinator**: FastAPI service that orchestrates model calls and aggregation
- **Evaluator**: Batch evaluation service with reproducible metrics
- **Datasets**: Synthetic dev set + SAMSum benchmark for paper results

### Key Features

- **Cosine Similarity Consensus**: Automatic outlier detection using embedding-based consensus (no LLM judge needed)
- **Multiple Consensus Methods**: Support for cosine similarity, LLM judge, and voting
- **Homogeneous & Heterogeneous Swarms**: Test same-model vs different-model configurations
- **Reproducible Metrics**: ROUGE, BERTScore, entity-based factuality, consensus confidence
- **Fair Comparison**: Unified prompt template across all models
- **Comprehensive Analysis**: Quality, latency, cost, hallucination rates, outlier detection
- **IEEE-Ready Outputs**: Structured CSV + Markdown reports for publication

## Quick Start

### Prerequisites

- Docker Desktop (with at least 24GB RAM allocated)
- Python 3.11+ (for local scripts)
- 50GB free disk space (for model downloads)

### 1. Clone and Setup

```bash
cd slm-swarm
```

### 2. Configure Big Model Endpoint (Optional)

Create a `.env` file:

```bash
# For remote vLLM endpoint
BIGMODEL_URL=https://your-vllm-api.com/v1
BIGMODEL_MODEL=llama-3.1-70b-instruct
BIGMODEL_TYPE=openai
BIGMODEL_API_KEY=your-api-key

# OR for OpenAI GPT-4 (optional reference)
# BIGMODEL_URL=https://api.openai.com/v1
# BIGMODEL_MODEL=gpt-4
# BIGMODEL_TYPE=openai
# BIGMODEL_API_KEY=sk-...

# OR for Anthropic Claude (optional reference)
# BIGMODEL_URL=https://api.anthropic.com/v1/messages
# BIGMODEL_MODEL=claude-3-5-sonnet-20241022
# BIGMODEL_TYPE=anthropic
# BIGMODEL_API_KEY=sk-ant-...
```

**Note:** If you skip big model configuration, the evaluator will only test the swarm. Set `ENABLE_BIG_MODEL=false` in that case.

### 3. Start the System

```bash
# Start all services (first run downloads models - takes 15-30 minutes)
docker compose up --build -d

# Watch model download progress
docker compose logs -f worker_qwen

# Check coordinator health
curl http://localhost:8000/health
```

### 4. Test with Synthetic Dataset

```bash
# Run evaluation on 50 synthetic messages
docker compose run --rm evaluator

# Check results
ls results/run_*
cat results/run_*/summary.md
```

## Usage

### API Endpoints

The coordinator exposes three endpoints on `http://localhost:8000`:

#### 1. `/summarize` - Swarm Approach

Fan out to 3 workers, judge picks best summary.

```bash
curl -X POST http://localhost:8000/summarize \
  -H "Content-Type: application/json" \
  -d '{
    "id": "test_001",
    "message": "Your message text here...",
    "mode": "tldr"
  }'
```

Response:
```json
{
  "id": "test_001",
  "swarm_summary": "Best summary selected by judge",
  "candidates": [
    {"model": "qwen2.5:7b-instruct", "summary": "...", "latency": 2.3},
    {"model": "llama3.1:8b-instruct", "summary": "...", "latency": 2.5},
    {"model": "mistral:7b-instruct", "summary": "...", "latency": 2.1}
  ],
  "latency": 3.2
}
```

#### 2. `/summarize_big` - Big Model Baseline

Single large model approach.

```bash
curl -X POST http://localhost:8000/summarize_big \
  -H "Content-Type: application/json" \
  -d '{
    "id": "test_001",
    "message": "Your message text here...",
    "mode": "tldr"
  }'
```

#### 3. `/summarize_compare` - Side-by-Side Comparison

Runs both approaches in parallel.

```bash
curl -X POST http://localhost:8000/summarize_compare \
  -H "Content-Type: application/json" \
  -d '{
    "id": "test_001",
    "message": "Your message text here...",
    "mode": "tldr"
  }'
```

### Batch Evaluation

#### Option 1: Use Synthetic Dev Set (Default)

```bash
docker compose run --rm evaluator
```

#### Option 2: Use SAMSum Benchmark (For Paper)

First, prepare SAMSum dataset:

```bash
# Install dependencies
pip install datasets

# Download and convert SAMSum
cd scripts
python prepare_samsum.py --output ../dataset/samsum.jsonl --split test
```

Then run evaluation:

```bash
docker compose run --rm -e DATASET_PATH=/app/dataset/samsum.jsonl evaluator
```

#### Option 3: Without Big Model

```bash
docker compose run --rm -e ENABLE_BIG_MODEL=false evaluator
```

### Results Analysis

Results are saved to `results/run_XXXXXXXX/`:

```
results/run_abc12345/
├── outputs.jsonl          # Per-example detailed results
├── metrics.csv            # Aggregate metrics
└── summary.md             # Human-readable report
```

View the summary:

```bash
cat results/run_*/summary.md
```

Aggregate multiple runs:

```bash
cd scripts
python aggregate_results.py --results-dir ../results
```

## Evaluation Metrics

### 1. ROUGE-1/2/L
Standard n-gram overlap with reference summaries.

### 2. BERTScore
Semantic similarity using contextualized embeddings (rescaled with baseline).

### 3. Factuality (Reproducible, No LLM Judge)
- **Entity Consistency**: Flags summaries introducing entities not in source
- **Content Overlap**: Measures overlap between source and summary content words
- **Hallucination Rate**: Percentage of summaries flagged as potential hallucinations

### 4. Latency
- p50 (median), p95 (95th percentile), and mean response times

### 5. Cost Estimate
Token-based cost estimation:
- Small models (7-8B): ~$0.0001 per 1K tokens
- Large models (70B): ~$0.0008 per 1K tokens

### 6. Consensus Metrics (Swarm Only)
- **Average Consensus Similarity**: Mean cosine similarity among candidates
- **Outlier Detection Rate**: Percentage of examples where an outlier was identified
- **Consensus Confidence**: Separation between consensus cluster and outlier

## Consensus Mechanisms

The system supports multiple consensus methods for selecting the best summary from swarm candidates:

### Cosine Similarity Consensus (Default, Recommended)

**How it works:**
1. Embed all candidate summaries using `sentence-transformers/all-MiniLM-L6-v2`
2. Compute pairwise cosine similarity matrix (NxN)
3. Calculate average similarity for each candidate
4. Detect outlier: candidate with lowest avg similarity below threshold
5. Return candidate with highest avg similarity (consensus centroid)

**Advantages:**
- ✅ **No extra inference cost** - just vector computation (~10ms for 4 summaries)
- ✅ **Fully reproducible** - pure mathematics, no model variability
- ✅ **Outlier robust** - automatically detects and filters bad outputs
- ✅ **Interpretable** - provides similarity scores and confidence metrics

**Configuration:**
```bash
CONSENSUS_METHOD=cosine
CONSENSUS_THRESHOLD=0.3  # Outlier detection threshold
```

**Example response:**
```json
{
  "swarm_summary": "Meeting rescheduled to Friday 2 PM in Room B.",
  "consensus_metadata": {
    "method": "cosine",
    "consensus_scores": [0.89, 0.91, 0.88, 0.42],
    "outlier_detected": true,
    "outlier_idx": 3,
    "consensus_confidence": 0.47,
    "avg_consensus_similarity": 0.89
  }
}
```

### LLM Judge Consensus (Legacy)

Uses Phi-3 model to evaluate and select best summary.

**Configuration:**
```bash
CONSENSUS_METHOD=judge
```

**Note:** Slower and less reproducible than cosine similarity.

### Voting Consensus

Simple majority voting based on exact string match.

**Configuration:**
```bash
CONSENSUS_METHOD=voting
```

**Note:** Only effective when multiple models produce identical outputs.

## Swarm Configurations

### Heterogeneous Swarm (Default)

Different models for diversity:
- Worker 1: Qwen2.5-7B-Instruct
- Worker 2: Llama-3.1-8B-Instruct
- Worker 3: Mistral-7B-Instruct
- Worker 4: Phi-3-3.8B

**Benefits:** Higher diversity, better outlier detection, catches model-specific errors

**Usage:**
```bash
docker compose up -d
```

### Homogeneous Swarm

Same model with temperature variation for diversity:
- All workers: Same model (e.g., Qwen-7B)
- Temperature variation: [0.1, 0.3, 0.5, 0.7]

**Benefits:** Simpler, faster, higher consensus similarity

**Usage:**
```bash
# For Qwen
HOMO_MODEL=qwen2.5:7b-instruct docker compose -f docker-compose.homogeneous.yml up -d

# For Llama
HOMO_MODEL=llama3.1:8b-instruct docker compose -f docker-compose.homogeneous.yml up -d

# For Mistral
HOMO_MODEL=mistral:7b-instruct docker compose -f docker-compose.homogeneous.yml up -d
```

## Consensus Experiments

### Run All Experiments

Automated script to test all configurations:

```bash
./scripts/run_consensus_experiments.sh
```

This runs:
1. Heterogeneous swarm (cosine consensus)
2. Heterogeneous swarm (judge consensus)
3. Heterogeneous swarm (voting consensus)
4. Homogeneous Qwen swarm
5. Homogeneous Llama swarm
6. Homogeneous Mistral swarm
7. Homogeneous Phi swarm

### Adversarial Outlier Testing (Scenario 3)

Test the consensus mechanism by intentionally injecting bad summaries:

```bash
./scripts/test_adversarial_outlier.sh
```

This demonstrates outlier detection by running 3 tests:
1. **Baseline:** Normal consensus (no injection)
2. **Adversarial 1:** Inject nonsense at worker 2
3. **Adversarial 2:** Inject different nonsense at worker 1

**Manual test:**
```bash
# Inject bad summary at worker 2
ADVERSARIAL_MODE=true \
ADVERSARIAL_WORKER_IDX=2 \
ADVERSARIAL_SUMMARY="The moon is made of cheese." \
docker compose up -d

# Run evaluation
docker compose run --rm -e ENABLE_BIG_MODEL=false evaluator

# Check if outlier was detected
cat results/run_*/outputs.jsonl | jq '.consensus_metadata | select(.outlier_detected == true)' | head -5
```

**Expected results:**
- **Outlier detected:** 100% (every example detects the bad summary)
- **Adversarial never selected:** `adversarial_was_selected: false`
- **Consensus confidence:** High (>0.5) indicating clear separation
- **Consensus similarity:** Lower than baseline (~0.65 vs 0.93)

### Compare Results

```bash
python scripts/compare_consensus_experiments.py --results-dir results/
```

Generates:
- Comparison table across all experiments
- Homogeneous vs heterogeneous analysis
- Consensus method comparison
- `results/consensus_comparison.md` report

### Expected Findings

Based on the consensus mechanism design:

**Quality:**
- Heterogeneous swarms: Slightly better ROUGE/BERTScore (model diversity)
- Homogeneous swarms: Competitive quality with simpler setup

**Robustness:**
- Heterogeneous: Higher outlier detection rate (28% vs 12-15%)
- Homogeneous: Lower but still effective outlier detection

**Consensus Strength:**
- Homogeneous: Higher avg similarity (0.89-0.92)
- Heterogeneous: Lower similarity (0.81) due to diversity

**Speed:**
- Homogeneous: Slightly faster (less model diversity overhead)
- Heterogeneous: Marginal slowdown (~0.3s)

**Research Questions:**
1. Does consensus improve quality vs single model? → Yes, removes outliers
2. Do heterogeneous swarms outperform homogeneous? → Marginal quality gain, more outliers detected
3. Is diversity worth the complexity? → Depends on application requirements

## Project Structure

```
slm-swarm/
├── docker-compose.yml              # Multi-container orchestration
├── coordinator/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── app.py                      # FastAPI coordinator service
├── evaluator/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── eval.py                     # Batch evaluation + metrics
├── dataset/
│   ├── messages.jsonl              # 50 synthetic messages (dev set)
│   ├── samsum.jsonl                # SAMSum benchmark (download separately)
│   └── qmsum.jsonl                 # QMSum benchmark (optional)
├── scripts/
│   ├── prepare_samsum.py           # Download and convert SAMSum
│   └── aggregate_results.py        # Combine results from multiple runs
├── results/                        # Auto-generated evaluation results
│   └── run_XXXXXXXX/
│       ├── outputs.jsonl
│       ├── metrics.csv
│       └── summary.md
└── README.md
```

## Configuration

### Environment Variables

All configurable via `.env` file or docker-compose environment:

**Swarm Configuration:**
- `WORKERS`: JSON array of worker URLs
- `MODELS`: JSON array of model names (must match Ollama tags)
- `WORKER_TEMPERATURES`: JSON array of temperatures (for homogeneous swarms)
- `JUDGE`: URL of judge worker (for judge consensus method)
- `JUDGE_MODEL`: Judge model name

**Consensus Configuration:**
- `CONSENSUS_METHOD`: `cosine` (default), `judge`, or `voting`
- `CONSENSUS_THRESHOLD`: Outlier detection threshold (default: 0.3)
- `SWARM_TYPE`: `heterogeneous` (default) or `homogeneous`

**Big Model Configuration:**
- `BIGMODEL_URL`: API endpoint URL
- `BIGMODEL_MODEL`: Model identifier
- `BIGMODEL_TYPE`: `openai`, `vllm`, or `anthropic`
- `BIGMODEL_API_KEY`: Authentication key

**Evaluator Configuration:**
- `COORDINATOR_URL`: Coordinator service URL (default: `http://coordinator:8000`)
- `DATASET_PATH`: Path to evaluation dataset JSONL
- `RESULTS_DIR`: Output directory for results
- `ENABLE_BIG_MODEL`: `true` or `false`

### Prompt Normalization

**Critical for fair comparison:** All models use the same prompt template:

```python
PROMPT_TEMPLATE = """You are a concise summarization model.
Task: Summarize the message into one sentence TL;DR.
Constraints:
- One sentence, max 20 words
- Preserve facts (names, dates, amounts)
- No invented information

Message:
{message}
"""
```

**Decoding settings** (identical for all models):
- `temperature=0.2`
- `max_tokens=200`
- `timeout=180s`

## Troubleshooting

### Out of Memory (OOM)

Docker Desktop → Settings → Resources → Increase RAM to 24GB+

Or reduce to 2 workers:
```yaml
# In docker-compose.yml, comment out worker_mistral
```

### Models Not Downloading

Check logs:
```bash
docker compose logs worker_qwen
```

Manually pull:
```bash
docker exec -it worker_qwen ollama pull qwen2.5:7b-instruct
```

### Coordinator Not Responding

Check health:
```bash
docker compose ps
curl http://localhost:8000/health
```

Restart:
```bash
docker compose restart coordinator
```

### Slow Evaluation

- Use quantized models (add to docker-compose.yml):
  ```yaml
  command: [..., "ollama pull qwen2.5:7b-instruct-q4_0", ...]
  ```
- Reduce dataset size for testing
- Check if big model endpoint has rate limits

### BERTScore Warnings

First run downloads BERT model (~400MB). Subsequent runs are faster.

## Experimental Workflow

### Typical Research Workflow

1. **Dev Testing** (synthetic dataset):
   ```bash
   docker compose up -d
   docker compose run --rm evaluator
   ```

2. **Paper Experiments** (SAMSum benchmark):
   ```bash
   # Prepare SAMSum
   python scripts/prepare_samsum.py
   
   # Run evaluation
   docker compose run --rm -e DATASET_PATH=/app/dataset/samsum.jsonl evaluator
   ```

3. **Ablation Studies**:
   - 2-worker swarm: Comment out one worker in docker-compose.yml
   - Different aggregation: Modify judge logic in coordinator/app.py
   - Different models: Change model tags in docker-compose.yml

4. **Aggregate Results**:
   ```bash
   python scripts/aggregate_results.py
   ```

### For IEEE Paper

The system outputs are structured for direct inclusion in research papers:

- `metrics.csv`: Import into tables (Excel/LaTeX)
- `summary.md`: Copy-paste into paper draft
- `outputs.jsonl`: Sample outputs for qualitative analysis
- Reproducible methodology clearly documented

## Hardware Requirements

### Minimum (Swarm Only)
- 16GB RAM
- 40GB disk space
- 4-core CPU

### Recommended (Swarm + Remote Big Model)
- 32GB RAM
- 50GB disk space
- 8-core CPU

### For Local Big Model (70B)
- 64GB RAM + 40GB VRAM (A100/H100)
- 100GB disk space
- Use remote vLLM endpoint instead if unavailable

## Citation

If you use this system in your research, please cite:

```bibtex
@article{your_paper_2025,
  title={Evaluating Small Language Model Swarms for Text Summarization},
  author={Your Name},
  journal={IEEE Conference},
  year={2025}
}
```

## License

MIT License - See LICENSE file for details

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request with clear description

## Support

For issues or questions:
- Open a GitHub issue
- Check troubleshooting section above
- Review Docker logs: `docker compose logs`

---

**Last Updated:** January 2025  
**System Version:** 1.0.0


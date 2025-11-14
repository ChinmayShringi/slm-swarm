# Evaluation Summary - Run samsum_main

## Dataset
- Total samples: 819
- Swarm valid samples: 819
- Big model valid samples: 0

## ROUGE Scores (F1)

| Metric | Swarm | Big Model |
|--------|-------|-----------|
| ROUGE-1 | 0.4211 | 0.0000 |
| ROUGE-2 | 0.1612 | 0.0000 |
| ROUGE-L | 0.3332 | 0.0000 |

## BERTScore

| Metric | Swarm | Big Model |
|--------|-------|-----------|
| Precision | 0.4098 | 0.0000 |
| Recall | 0.4331 | 0.0000 |
| F1 | 0.4212 | 0.0000 |

## Latency (seconds)

| Metric | Swarm | Big Model |
|--------|-------|-----------|
| Mean | 16.89 | 0.00 |
| Median (p50) | 15.19 | 0.00 |
| p95 | 33.38 | 0.00 |

## Factuality

| Metric | Swarm | Big Model |
|--------|-------|-----------|
| Hallucination Rate | 66.18% | 100.00% |

## Consensus Metrics (Swarm Only)

| Metric | Value |
|--------|-------|
| Avg Consensus Similarity | 0.9069 |
| Outlier Detection Rate | 0.00% |
| Consensus Confidence | 0.0272 |

## Cost Estimate (USD)

| System | Estimated Cost |
|--------|---------------|
| Swarm | $0.0275 |
| Big Model | $0.0000 |

## Key Findings

1. **Quality**: Swarm achieves higher ROUGE-L F1 score
2. **Speed**: Swarm is faster (median latency)
3. **Cost**: Big Model is more cost-effective
4. **Reliability**: Swarm has lower hallucination rate

---

*Generated: 2025-11-14 10:20:38 UTC*

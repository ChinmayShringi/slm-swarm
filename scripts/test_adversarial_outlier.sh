#!/bin/bash

# Test adversarial outlier detection
# Demonstrates consensus mechanism by injecting intentionally bad summaries

set -e

echo "=============================================="
echo "Adversarial Outlier Detection Test"
echo "=============================================="
echo ""

DATASET="${DATASET:-dataset/messages_tiny.jsonl}"

# Test 1: Normal baseline (no adversarial)
echo "[1/3] Test 1: Normal consensus (no adversarial injection)"
echo "----------------------------------------------"
docker compose down 2>/dev/null || true
docker compose up -d
sleep 15

docker compose run --rm \
  -e ENABLE_BIG_MODEL=false \
  -e DATASET_PATH=/app/${DATASET} \
  evaluator

BASELINE_RUN=$(ls -td results/run_* | head -1)
echo "✓ Baseline results: $BASELINE_RUN"
echo ""
sleep 5

# Test 2: Adversarial injection at worker 2
echo "[2/3] Test 2: Adversarial injection at worker index 2"
echo "----------------------------------------------"
echo "Injecting: 'The moon is made of cheese and unicorns live on Mars.'"
docker compose down 2>/dev/null || true

ADVERSARIAL_MODE=true \
ADVERSARIAL_WORKER_IDX=2 \
ADVERSARIAL_SUMMARY="The moon is made of cheese and unicorns live on Mars." \
docker compose up -d

sleep 15

ADVERSARIAL_MODE=true \
ADVERSARIAL_WORKER_IDX=2 \
ADVERSARIAL_SUMMARY="The moon is made of cheese and unicorns live on Mars." \
docker compose run --rm \
  -e ENABLE_BIG_MODEL=false \
  -e DATASET_PATH=/app/${DATASET} \
  evaluator

ADVERSARIAL_RUN=$(ls -td results/run_* | head -1)
echo "✓ Adversarial results: $ADVERSARIAL_RUN"
echo ""
sleep 5

# Test 3: Different adversarial at worker 1
echo "[3/3] Test 3: Adversarial injection at worker index 1"
echo "----------------------------------------------"
echo "Injecting: 'Random irrelevant information about cooking recipes.'"
docker compose down 2>/dev/null || true

ADVERSARIAL_MODE=true \
ADVERSARIAL_WORKER_IDX=1 \
ADVERSARIAL_SUMMARY="Random irrelevant information about cooking recipes." \
docker compose up -d

sleep 15

ADVERSARIAL_MODE=true \
ADVERSARIAL_WORKER_IDX=1 \
ADVERSARIAL_SUMMARY="Random irrelevant information about cooking recipes." \
docker compose run --rm \
  -e ENABLE_BIG_MODEL=false \
  -e DATASET_PATH=/app/${DATASET} \
  evaluator

ADVERSARIAL2_RUN=$(ls -td results/run_* | head -1)
echo "✓ Second adversarial results: $ADVERSARIAL2_RUN"
echo ""

# Compare results
echo "=============================================="
echo "RESULTS COMPARISON"
echo "=============================================="
echo ""

echo "Baseline (No Adversarial):"
echo "  Run: $BASELINE_RUN"
cat "$BASELINE_RUN/metrics.csv" | grep -E "(outlier_detected_rate|consensus_avg_similarity|consensus_confidence)" || echo "  Consensus metrics available in metrics.csv"

echo ""
echo "Adversarial Test 1 (Worker 2 injected):"
echo "  Run: $ADVERSARIAL_RUN"
cat "$ADVERSARIAL_RUN/metrics.csv" | grep -E "(outlier_detected_rate|consensus_avg_similarity|consensus_confidence)" || echo "  Consensus metrics available in metrics.csv"

echo ""
echo "Adversarial Test 2 (Worker 1 injected):"
echo "  Run: $ADVERSARIAL2_RUN"
cat "$ADVERSARIAL2_RUN/metrics.csv" | grep -E "(outlier_detected_rate|consensus_avg_similarity|consensus_confidence)" || echo "  Consensus metrics available in metrics.csv"

echo ""
echo "=============================================="
echo "Expected Outcomes:"
echo "=============================================="
echo ""
echo "1. Baseline: Outlier rate ~0%, consensus sim ~0.93"
echo "2. Adversarial: Outlier rate ~100%, consensus sim lower"
echo "3. Adversarial: Should NOT select the injected bad summary"
echo ""
echo "To verify outlier detection worked:"
echo "  cat $ADVERSARIAL_RUN/outputs.jsonl | jq '.consensus_metadata.outlier_detected' | head -5"
echo "  cat $ADVERSARIAL_RUN/outputs.jsonl | jq '.consensus_metadata.adversarial_was_selected' | head -5"
echo ""


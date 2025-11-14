#!/bin/bash

# Run SAMSum evaluation with checkpoint support
# Can be stopped and resumed anytime

set -e

echo "=============================================="
echo "SAMSum Evaluation Runner (with checkpoints)"
echo "=============================================="
echo ""

# Configuration
RUN_ID="${RUN_ID:-samsum_main}"
DATASET="/app/dataset/samsum.jsonl"
RESULTS="/app/results/samsum"

echo "Configuration:"
echo "  Run ID: $RUN_ID"
echo "  Dataset: $DATASET"
echo "  Results: $RESULTS"
echo "  Checkpoint interval: Every 10 examples"
echo ""

# Check if resuming
if [ -f "results/samsum/run_${RUN_ID}/outputs.jsonl" ]; then
    PROCESSED=$(wc -l < "results/samsum/run_${RUN_ID}/outputs.jsonl")
    echo "⚠️  Existing run found with $PROCESSED examples processed"
    echo "This will RESUME from where it left off"
    echo ""
fi

echo "Starting evaluation..."
echo "Started at: $(date)"
echo ""
echo "Monitor progress with:"
echo "  ./scripts/check_progress.sh"
echo ""
echo "Press Ctrl+C to stop (progress will be saved)"
echo "=============================================="
echo ""

# Run evaluation
docker compose run --rm \
  -e ENABLE_BIG_MODEL=false \
  -e DATASET_PATH="$DATASET" \
  -e RESULTS_DIR="$RESULTS" \
  -e RUN_ID="$RUN_ID" \
  -e CHECKPOINT_INTERVAL=10 \
  evaluator

echo ""
echo "=============================================="
echo "Evaluation complete!"
echo "Results saved to: results/samsum/run_${RUN_ID}/"
echo ""
echo "View results:"
echo "  cat results/samsum/run_${RUN_ID}/summary.md"
echo "=============================================="


#!/bin/bash
# Run evaluations for messages_small.jsonl on all swarm sizes (3, 6, 12, 18)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

# Configuration
DATASET="dataset/messages_small.jsonl"
RESULTS_BASE="results"
ENABLE_BIG_MODEL="false"

# Coordinator ports for each swarm size
declare -A COORDINATOR_PORTS=(
    [3]="8103"
    [6]="8106"
    [12]="8112"
    [18]="8118"
)

# Function to run evaluation for a specific swarm size
run_evaluation() {
    local workers=$1
    local port=${COORDINATOR_PORTS[$workers]}
    
    if [ -z "$port" ]; then
        echo "ERROR: No coordinator port configured for $workers workers"
        return 1
    fi
    
    echo "=========================================="
    echo "Running evaluation for $workers workers"
    echo "Coordinator: http://localhost:$port"
    echo "Dataset: $DATASET"
    echo "=========================================="
    
    # Check if coordinator is healthy
    if ! curl -s "http://localhost:$port/health" > /dev/null 2>&1; then
        echo "ERROR: Coordinator on port $port is not responding"
        echo "Please ensure the $workers-worker configuration is running"
        return 1
    fi
    
    # Activate virtual environment
    if [ -d "venv" ]; then
        source venv/bin/activate
    fi
    
    # Run evaluation
    COORDINATOR_URL="http://localhost:$port" \
    DATASET_PATH="$DATASET" \
    RESULTS_DIR="$RESULTS_BASE" \
    ENABLE_BIG_MODEL="$ENABLE_BIG_MODEL" \
    nohup python3 evaluator/eval.py > "eval_messages_small_${workers}w.log" 2>&1 &
    
    local pid=$!
    echo "Started evaluation (PID: $pid)"
    echo "Log file: eval_messages_small_${workers}w.log"
    echo ""
    
    return 0
}

# Run evaluations for all swarm sizes
echo "Starting evaluations for messages_small.jsonl dataset"
echo "Swarm sizes: 3, 6, 12, 18 workers"
echo ""

for workers in 3 6 12 18; do
    run_evaluation $workers
    sleep 2  # Small delay between starts
done

echo "All evaluations started!"
echo ""
echo "Monitor progress with:"
echo "  tail -f eval_messages_small_*w.log"
echo ""
echo "Check running processes:"
echo "  ps aux | grep eval.py"
echo ""
echo "Results will be in: $RESULTS_BASE/messages_small/<timestamp>_<workers>/"


#!/bin/bash

# Automated script to run consensus experiments comparing homogeneous vs heterogeneous swarms
# Tests different model configurations and consensus mechanisms

set -e

echo "=============================================="
echo "Consensus Mechanism Experiments"
echo "=============================================="
echo ""

# Configuration
DATASET="${DATASET_PATH:-dataset/messages.jsonl}"
RESULTS_DIR="results"

# Experiment counters
TOTAL_EXPERIMENTS=7
CURRENT=0

# Function to run experiment
run_experiment() {
    local name=$1
    local compose_file=$2
    local env_vars=$3
    
    CURRENT=$((CURRENT + 1))
    echo ""
    echo "[$CURRENT/$TOTAL_EXPERIMENTS] Running: $name"
    echo "=============================================="
    
    # Stop any running services
    docker compose -f $compose_file down 2>/dev/null || true
    
    # Start services
    echo "Starting services..."
    $env_vars docker compose -f $compose_file up -d
    
    # Wait for services to be ready
    echo "Waiting for coordinator to be ready..."
    for i in {1..60}; do
        if curl -s http://localhost:8000/health > /dev/null 2>&1; then
            echo "✓ Coordinator ready"
            break
        fi
        if [ $i -eq 60 ]; then
            echo "✗ Coordinator did not become ready"
            docker compose -f $compose_file logs coordinator
            return 1
        fi
        sleep 5
    done
    
    # Run evaluation
    echo "Running evaluation..."
    $env_vars docker compose -f $compose_file run --rm evaluator
    
    # Tag results with experiment name
    latest_run=$(ls -td ${RESULTS_DIR}/run_* | head -1)
    if [ -d "$latest_run" ]; then
        echo "$name" > "$latest_run/experiment_name.txt"
        echo "✓ Results saved to: $latest_run"
    fi
    
    # Stop services
    docker compose -f $compose_file down
    
    echo "✓ Experiment complete: $name"
    sleep 5
}

# Experiment 1: Heterogeneous swarm with cosine consensus (baseline)
run_experiment \
    "Heterogeneous Swarm (Cosine Consensus)" \
    "docker-compose.yml" \
    "CONSENSUS_METHOD=cosine"

# Experiment 2: Heterogeneous swarm with judge consensus
run_experiment \
    "Heterogeneous Swarm (Judge Consensus)" \
    "docker-compose.yml" \
    "CONSENSUS_METHOD=judge"

# Experiment 3: Heterogeneous swarm with voting consensus
run_experiment \
    "Heterogeneous Swarm (Voting Consensus)" \
    "docker-compose.yml" \
    "CONSENSUS_METHOD=voting"

# Experiment 4: Homogeneous Qwen swarm
run_experiment \
    "Homogeneous Qwen-7B Swarm" \
    "docker-compose.homogeneous.yml" \
    "HOMO_MODEL=qwen2.5:7b-instruct CONSENSUS_METHOD=cosine"

# Experiment 5: Homogeneous Llama swarm
run_experiment \
    "Homogeneous Llama-8B Swarm" \
    "docker-compose.homogeneous.yml" \
    "HOMO_MODEL=llama3.1:8b-instruct CONSENSUS_METHOD=cosine"

# Experiment 6: Homogeneous Mistral swarm
run_experiment \
    "Homogeneous Mistral-7B Swarm" \
    "docker-compose.homogeneous.yml" \
    "HOMO_MODEL=mistral:7b-instruct CONSENSUS_METHOD=cosine"

# Experiment 7: Homogeneous Phi swarm
run_experiment \
    "Homogeneous Phi-3.8B Swarm" \
    "docker-compose.homogeneous.yml" \
    "HOMO_MODEL=phi3:3.8b CONSENSUS_METHOD=cosine"

echo ""
echo "=============================================="
echo "All Experiments Complete!"
echo "=============================================="
echo ""
echo "Total runs completed: $TOTAL_EXPERIMENTS"
echo "Results directory: $RESULTS_DIR"
echo ""
echo "Next steps:"
echo "  1. Analyze results: python scripts/compare_consensus_experiments.py"
echo "  2. View individual reports: cat results/run_*/summary.md"
echo ""


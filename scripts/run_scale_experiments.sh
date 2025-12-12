#!/bin/bash

# Multi-Scale SLM Swarm Experiments
# Runs evaluations for 3, 6, 12, and 18 workers in both heterogeneous and homogeneous modes

set -e

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BASE_RESULTS_DIR="results/scale_experiment_${TIMESTAMP}"
mkdir -p "${BASE_RESULTS_DIR}"

echo "=========================================="
echo "Starting Multi-Scale Swarm Experiments"
echo "Timestamp: ${TIMESTAMP}"
echo "Results will be saved to: ${BASE_RESULTS_DIR}"
echo "=========================================="

# Function to run a single experiment
run_experiment() {
    local config_name=$1
    local compose_file=$2
    local num_workers=$3
    local swarm_type=$4
    
    echo ""
    echo "=========================================="
    echo "Running: ${config_name}"
    echo "Workers: ${num_workers} (${swarm_type})"
    echo "Compose File: ${compose_file}"
    echo "=========================================="
    
    # Stop any existing containers
    echo "Stopping existing containers..."
    sudo docker-compose -f docker-compose.hetero-3.yml down 2>/dev/null || true
    sudo docker-compose -f docker-compose.hetero-6.yml down 2>/dev/null || true
    sudo docker-compose -f docker-compose.hetero-12.yml down 2>/dev/null || true
    sudo docker-compose -f docker-compose.hetero-18.yml down 2>/dev/null || true
    sudo docker-compose -f docker-compose.homo-3.yml down 2>/dev/null || true
    sudo docker-compose -f docker-compose.homo-6.yml down 2>/dev/null || true
    sudo docker-compose -f docker-compose.homo-12.yml down 2>/dev/null || true
    sudo docker-compose -f docker-compose.homo-18.yml down 2>/dev/null || true
    sleep 5
    
    # Start the configuration
    echo "Starting ${config_name}..."
    sudo docker-compose -f "${compose_file}" up -d
    
    # Wait for services to be ready
    echo "Waiting for services to start..."
    sleep 30
    
    # Check coordinator health
    local max_retries=10
    local retry=0
    while [ $retry -lt $max_retries ]; do
        if sudo docker-compose -f "${compose_file}" exec -T coordinator curl -f http://localhost:8000/health > /dev/null 2>&1; then
            echo "Coordinator is healthy"
            break
        fi
        retry=$((retry + 1))
        echo "Waiting for coordinator... ($retry/$max_retries)"
        sleep 10
    done
    
    if [ $retry -eq $max_retries ]; then
        echo "ERROR: Coordinator failed to become healthy"
        sudo docker-compose -f "${compose_file}" logs coordinator
        exit 1
    fi
    
    # Run evaluation
    echo "Running evaluation on SAMSum dataset..."
    START_TIME=$(date +%s)
    sudo docker-compose -f "${compose_file}" run --rm evaluator
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    
    echo "Evaluation completed in ${DURATION} seconds"
    
    # Find the latest run directory
    LATEST_RUN=$(ls -td results/run_* 2>/dev/null | head -1)
    
    if [ -n "${LATEST_RUN}" ]; then
        # Move results to organized location
        DEST_DIR="${BASE_RESULTS_DIR}/${config_name}"
        mkdir -p "${DEST_DIR}"
        cp -r "${LATEST_RUN}"/* "${DEST_DIR}/"
        
        # Add metadata
        echo "Configuration: ${config_name}" > "${DEST_DIR}/metadata.txt"
        echo "Workers: ${num_workers}" >> "${DEST_DIR}/metadata.txt"
        echo "Swarm Type: ${swarm_type}" >> "${DEST_DIR}/metadata.txt"
        echo "Compose File: ${compose_file}" >> "${DEST_DIR}/metadata.txt"
        echo "Duration: ${DURATION} seconds" >> "${DEST_DIR}/metadata.txt"
        echo "Timestamp: $(date)" >> "${DEST_DIR}/metadata.txt"
        
        echo "Results saved to: ${DEST_DIR}"
    else
        echo "WARNING: No results found for ${config_name}"
    fi
    
    # Stop configuration
    echo "Stopping ${config_name}..."
    sudo docker-compose -f "${compose_file}" down
    sleep 5
    
    echo "✓ ${config_name} completed"
}

# Run all experiments
echo ""
echo "Starting experiments..."
echo ""

# Heterogeneous swarms
run_experiment "hetero-3" "docker-compose.hetero-3.yml" 3 "heterogeneous"
run_experiment "hetero-6" "docker-compose.hetero-6.yml" 6 "heterogeneous"
run_experiment "hetero-12" "docker-compose.hetero-12.yml" 12 "heterogeneous"
run_experiment "hetero-18" "docker-compose.hetero-18.yml" 18 "heterogeneous"

# Homogeneous swarms
run_experiment "homo-3" "docker-compose.homo-3.yml" 3 "homogeneous"
run_experiment "homo-6" "docker-compose.homo-6.yml" 6 "homogeneous"
run_experiment "homo-12" "docker-compose.homo-12.yml" 12 "homogeneous"
run_experiment "homo-18" "docker-compose.homo-18.yml" 18 "homogeneous"

echo ""
echo "=========================================="
echo "All experiments completed!"
echo "=========================================="
echo ""
echo "Results location: ${BASE_RESULTS_DIR}"
echo ""
echo "Running analysis script..."
python3 scripts/analyze_scale_results.py --results-dir "${BASE_RESULTS_DIR}"

echo ""
echo "=========================================="
echo "Experiments complete!"
echo "View the scaling report at:"
echo "  ${BASE_RESULTS_DIR}/SCALING_REPORT.md"
echo "=========================================="


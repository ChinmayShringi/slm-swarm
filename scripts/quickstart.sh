#!/bin/bash

# Quick start script for SLM Swarm Evaluation System

set -e

echo "=========================================="
echo "SLM Swarm Evaluation System - Quick Start"
echo "=========================================="
echo ""

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "Error: Docker is not running. Please start Docker Desktop."
    exit 1
fi

echo "✓ Docker is running"
echo ""

# Build containers
echo "Building Docker containers..."
docker compose build
echo "✓ Containers built"
echo ""

# Start services
echo "Starting services (this will download models - may take 15-30 minutes)..."
docker compose up -d

echo ""
echo "Waiting for models to download and services to be healthy..."
echo "You can monitor progress with: docker compose logs -f worker_qwen"
echo ""

# Wait for coordinator
echo "Waiting for coordinator to be ready..."
for i in {1..60}; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "✓ Coordinator is ready!"
        break
    fi
    if [ $i -eq 60 ]; then
        echo "Warning: Coordinator did not become ready in time."
        echo "Check logs with: docker compose logs coordinator"
    fi
    sleep 5
done

echo ""
echo "=========================================="
echo "System is ready!"
echo "=========================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Test the API:"
echo "   curl http://localhost:8000/health"
echo ""
echo "2. Run evaluation on synthetic dataset:"
echo "   docker compose run --rm evaluator"
echo ""
echo "3. (Optional) Prepare SAMSum benchmark:"
echo "   pip install datasets"
echo "   cd scripts && python prepare_samsum.py"
echo ""
echo "4. View logs:"
echo "   docker compose logs -f coordinator"
echo ""
echo "5. Stop services:"
echo "   docker compose down"
echo ""
echo "For more information, see README.md"
echo ""


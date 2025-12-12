#!/bin/bash

# Download all required models for scale experiments

set -e

echo "=========================================="
echo "Downloading Models for Scale Experiments"
echo "=========================================="

# Function to download a model
download_model() {
    local model_name=$1
    local model_path=$2
    
    echo ""
    echo "Downloading: ${model_name} to ${model_path}"
    echo "==========================================

"
    
    # Start temporary container
    sudo docker run -d --name temp_download_${model_name//[:\/.]/_} \
        -v "${model_path}:/root/.ollama" \
        ollama/ollama:latest
    
    # Wait for container to be ready
    sleep 10
    
    # Download model
    sudo docker exec temp_download_${model_name//[:\/.]/_} ollama pull "${model_name}"
    
    # Stop and remove container
    sudo docker rm -f temp_download_${model_name//[:\/.]/_}
    
    echo "✓ ${model_name} downloaded successfully"
}

# Download models for heterogeneous experiments
cd /home/chinmay/slm-swarm

echo "Downloading models..."

# Already have: qwen, mistral
# Need: llama, gemma, tinyllama, phi

download_model "llama3.1:8b" "/home/chinmay/slm-swarm/models/llama"
download_model "gemma2:2b" "/home/chinmay/slm-swarm/models/gemma"
download_model "tinyllama:latest" "/home/chinmay/slm-swarm/models/tinyllama"
download_model "phi3:3.8b" "/home/chinmay/slm-swarm/models/phi"

echo ""
echo "=========================================="
echo "All models downloaded successfully!"
echo "=========================================="
echo ""
echo "Verifying models..."
ls -lh models/*/models/blobs | head -30


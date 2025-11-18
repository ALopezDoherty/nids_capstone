#!/bin/bash
echo " Starting Ollama Docker container..."

cd "$(dirname "$0")"

# Pull the latest Ollama image
docker pull ollama/ollama:latest

# Start the container
docker-compose up -d

# Wait for service to start
echo "Waiting for Ollama to start..."
sleep 15

# Check if container is running
if docker ps | grep -q nids-ollama; then
    echo "Ollama container is running"
else
    echo "Failed to start Ollama container"
    exit 1
fi

# Pull model
echo "Pulling Llama 3.1 8B model..."
docker exec nids-ollama ollama pull llama3.1:8b

echo "Ollama setup complete!"
echo "API available at: http://localhost:11434"
#!/bin/bash
echo "Stopping Ollama Docker container..."

cd "$(dirname "$0")"
docker-compose down

echo "Ollama stopped"
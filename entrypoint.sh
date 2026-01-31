#!/bin/bash
# entrypoint.sh

set -e

echo "Starting Ollama server in background..."
OLLAMA_HOST=0.0.0.0:11434 /app/ollama serve &
OLLAMA_PID=$!

# Wait until Ollama API is responsive
echo "Waiting for Ollama API..."
until curl -f http://localhost:11434/ > /dev/null 2>&1; do
  echo "Ollama not ready... retrying in 3s"
  sleep 3
done
echo "Ollama is live!"

# Pull your lightweight model (GGUF format)
# Note: Update this when switching checkpoints
MODEL_NAME="hf.co/kshitijthakkar/loggenix-moe-0.4B-0.2A-sft-s3.1:Q8_0"
echo "Pulling model: $MODEL_NAME"
/app/ollama pull "$MODEL_NAME" || {
  echo "Failed to pull model. Check name and internet."
  exit 1
}

# Start your app
echo "Launching enhanced_app.py"
exec python /app/enhanced_app.py

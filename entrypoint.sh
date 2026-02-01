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

# Authenticate with HuggingFace for private models
if [ -n "$HF_TOKEN" ]; then
  echo "Logging into HuggingFace..."
  export HF_TOKEN="$HF_TOKEN"
  export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
  python -m huggingface_hub.commands.huggingface_cli login --token "$HF_TOKEN"
  echo "HF authentication configured"
else
  echo "WARNING: HF_TOKEN not set - private model access may fail"
fi

# Pull your lightweight model (GGUF format)
# Note: Update this when switching checkpoints
MODEL_NAME="hf.co/kshitijthakkar/loggenix-moe-0.4B-0.2A-sft-s3.1:Q8_0"
echo "Pulling model: $MODEL_NAME"

/app/ollama pull "$MODEL_NAME" || {
  echo "Failed to pull model. Check name, auth token, and internet."
  echo "Model: $MODEL_NAME"
  echo "HF_TOKEN set: $([ -n "$HF_TOKEN" ] && echo 'yes' || echo 'no')"
  exit 1
}

# Start your app
echo "Launching enhanced_app.py"
exec python /app/enhanced_app.py

# Dockerfile - Hugging Face Space with Ollama (small model)
FROM python:3.11-slim

# Set Ollama environment
ENV OLLAMA_HOST=0.0.0.0:11434
ENV OLLAMA_ORIGINS=http://*,https://*
# Optional: change model storage to /data for better caching
# ENV OLLAMA_MODELS=/data/ollama

# Install dependencies
RUN apt-get update && \
    apt-get install -y curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Create non-root user and app directory
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app && \
    chown -R appuser:appuser /app

USER appuser
WORKDIR /app

# Install Ollama CLI from GitHub releases (more reliable than ollama.com)
RUN curl -fL --retry 5 --retry-delay 5 \
    -o /tmp/ollama https://github.com/ollama/ollama/releases/download/v0.5.7/ollama-linux-amd64 && \
    cp /tmp/ollama /app/ollama && \
    chmod +x /app/ollama && \
    rm -f /tmp/ollama

ENV PATH="/home/appuser/.local/bin:$PATH"

# Copy app
COPY --chown=appuser:appuser . /app

# Install Python dependencies
RUN ls -ltr && pwd
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Expose Gradio port (required)
EXPOSE 7860

# Entrypoint
COPY --chown=appuser:appuser entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

CMD ["/app/entrypoint.sh"]

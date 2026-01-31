# Dockerfile - Hugging Face Space with Ollama (small model)
FROM python:3.11-slim

# Set Ollama environment
ENV OLLAMA_HOST=0.0.0.0:11434
ENV OLLAMA_ORIGINS=http://*,https://*

# Install dependencies (including zstd for Ollama extraction)
RUN apt-get update && \
    apt-get install -y curl ca-certificates zstd && \
    rm -rf /var/lib/apt/lists/*

# Create non-root user and app directory
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app && \
    chown -R appuser:appuser /app

# Install Ollama CLI from GitHub releases (as root, then copy)
RUN curl -fL --retry 5 --retry-delay 5 \
    -o /tmp/ollama.tar.zst https://github.com/ollama/ollama/releases/download/v0.15.2/ollama-linux-amd64.tar.zst && \
    cd /tmp && \
    zstd -d ollama.tar.zst && \
    tar -xf ollama.tar && \
    cp /tmp/bin/ollama /app/ollama && \
    chmod +x /app/ollama && \
    chown appuser:appuser /app/ollama && \
    rm -rf /tmp/ollama*

USER appuser
WORKDIR /app

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

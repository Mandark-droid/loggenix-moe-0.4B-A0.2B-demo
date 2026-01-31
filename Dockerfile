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

# Install Ollama CLI
RUN curl -fL --retry 5 --retry-delay 5 \
    -H "User-Agent: Mozilla/5.0 (X11; Linux x86_64)" \
    -o /tmp/ollama.tgz https://ollama.com/download/ollama-linux-amd64.tgz && \
    tar -xzf /tmp/ollama.tgz --no-same-owner --no-same-permissions -C /tmp && \
    cp /tmp/bin/ollama /app/ollama && \
    chmod +x /app/ollama && \
    rm -rf /tmp/ollama.tgz /tmp/bin

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

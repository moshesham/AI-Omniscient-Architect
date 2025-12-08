# Multi-stage build for minimal final image
FROM python:3.11-slim AS builder

# Install build dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /install

# Copy dependency files first (better layer caching)
COPY requirements.txt pyproject.toml ./
COPY packages/ ./packages/

# Install Python dependencies
RUN python -m pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir --target=/install -r requirements.txt \
    && pip install --no-cache-dir --target=/install -e ./packages/core \
    && pip install --no-cache-dir --target=/install -e ./packages/llm \
    && pip install --no-cache-dir --target=/install -e ./packages/agents \
    && pip install --no-cache-dir --target=/install -e ./packages/tools \
    && pip install --no-cache-dir --target=/install -e ./packages/github \
    && pip install --no-cache-dir --target=/install -e ./packages/api \
    && pip install --no-cache-dir --target=/install -e ./packages/rag


# Final runtime stage
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app:/install \
    # Streamlit configuration
    STREAMLIT_GLOBAL_DEVELOPMENT_MODE=false \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_FILE_WATCHER_TYPE=none

WORKDIR /app

# Install runtime dependencies only
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy installed Python packages from builder
COPY --from=builder /install /install

# Copy application code
COPY web_app.py ./
COPY config.yaml ./
COPY packages/ ./packages/
COPY src/ ./src/

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash --uid 1000 app \
    && mkdir -p /app/knowledge /app/.streamlit \
    && chown -R app:app /app /install

# Copy Streamlit config
COPY --chown=app:app <<EOF /app/.streamlit/config.toml
[server]
port = 8501
address = "0.0.0.0"
enableCORS = false
enableXsrfProtection = true

[browser]
gatherUsageStats = false

[theme]
base = "light"
primaryColor = "#1f77b4"
EOF

USER app

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Run the application
CMD ["python", "-m", "streamlit", "run", "web_app.py", "--server.headless", "true"]
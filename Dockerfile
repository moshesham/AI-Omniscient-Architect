FROM python:3.11-slim AS builder

# Install build dependencies and curl in builder
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /install

# Copy and install Python dependencies into a target directory
COPY requirements.txt ./
RUN python -m pip install --upgrade pip \
    && pip install --no-cache-dir --target=/install -r requirements.txt


FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app:/install

WORKDIR /app

# Install runtime dependencies (curl for healthchecks and any runtime tools)
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder
COPY --from=builder /install /install

# Copy application code
COPY . /app

# Create a non-root user for security and fix permissions
RUN useradd --create-home --shell /bin/bash app \
    && chown -R app:app /app /install
USER app

# Expose Streamlit port
EXPOSE 8501

# Health check for Streamlit (attempt to connect to exposed port)
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import socket,sys; s=socket.socket(); s.settimeout(2);\
        sys.exit(0) if s.connect_ex(('127.0.0.1', 8501))==0 else sys.exit(1)"

# Run the application via streamlit and bind to 0.0.0.0
CMD ["streamlit", "run", "web_app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
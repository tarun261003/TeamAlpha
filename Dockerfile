FROM python:3.11-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (better Docker layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Setup the environment
ENV PYTHONPATH=/app
ENV ENABLE_WEB_INTERFACE=true
COPY . .

# Install the package itself (registers the truth-seeker-server entry point)
RUN pip install --no-cache-dir -e .

# Expose server port
EXPOSE 8000

# Health check — evaluator pings /health after docker run
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start the server
CMD ["server"]

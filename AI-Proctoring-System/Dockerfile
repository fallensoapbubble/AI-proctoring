# AI Proctoring System - Ultra-Optimized Docker Build
FROM python:3.11-slim

# Set environment variables for optimal performance
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    FLASK_APP=app.py \
    FLASK_ENV=production \
    DEPLOYMENT_MODE=docker \
    PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive \
    OPENCV_LOG_LEVEL=ERROR \
    TORCH_HOME=/app/models \
    MPLCONFIGDIR=/tmp/matplotlib \
    YOLO_CONFIG_DIR=/tmp/yolo

# Install minimal system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Essential utilities
    wget curl ca-certificates \
    # OpenCV dependencies
    libglib2.0-0 libsm6 libxext6 libxrender1 libgomp1 \
    libgl1-mesa-dev libglu1-mesa-dev \
    # Audio dependencies (minimal)
    libasound2-dev portaudio19-dev \
    # Build tools (will be removed after pip install)
    gcc g++ python3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Create application user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Copy requirements for better caching
COPY requirements.docker.txt ./

# Install Python packages efficiently
RUN pip install --upgrade pip setuptools wheel && \
    # Install CPU-only PyTorch first for smaller size
    pip install --no-cache-dir torch==2.1.0+cpu torchvision==0.16.0+cpu --index-url https://download.pytorch.org/whl/cpu && \
    # Install remaining packages
    pip install --no-cache-dir -r requirements.docker.txt && \
    # Clean up build dependencies and caches
    apt-get remove -y gcc g++ python3-dev && \
    apt-get autoremove -y && apt-get clean && \
    rm -rf /var/lib/apt/lists/* /root/.cache/pip/* /tmp/* ~/.cache

# Create directory structure with proper permissions
RUN mkdir -p evidence/screenshots evidence/metadata \
             cheating_evidence/screenshots \
             config logs/system logs/cheating logs/evidence models \
             sessions templates static \
             /tmp/matplotlib /tmp/yolo /home/appuser/.config && \
    chown -R appuser:appuser /app /tmp/matplotlib /tmp/yolo /home/appuser && \
    chmod -R 755 /app

# Copy application code (optimized with .dockerignore)
COPY --chown=appuser:appuser . .

# Set up entrypoint
RUN chmod +x docker-entrypoint.sh

# Switch to non-root user for security
USER appuser

# Expose port
EXPOSE 5000

# Health check endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Use entrypoint script
ENTRYPOINT ["./docker-entrypoint.sh"]
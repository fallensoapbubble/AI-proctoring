#!/bin/bash
# Docker entrypoint script for AI Proctoring System

set -e

# Function to log messages with timestamp
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1"
}

log "🚀 Starting AI Proctoring System in Docker..."

# Ensure we're in the correct directory
cd /app

# Create necessary directories with proper structure
log "📁 Creating directory structure..."
mkdir -p evidence/screenshots evidence/audio evidence/metadata \
         cheating_evidence/screenshots cheating_evidence/audio \
         sessions config logs/system logs/cheating logs/evidence models \
         2>/dev/null || log "⚠️  Some directories already exist or permission denied - continuing..."

# Ensure sessions directory is writable (critical for app functionality)
log "🔐 Setting up sessions directory permissions..."
if [ -d "/app/sessions" ]; then
    chmod -R 755 /app/sessions 2>/dev/null || log "⚠️  Could not set sessions permissions"
    touch /app/sessions/.test_write 2>/dev/null && rm -f /app/sessions/.test_write && log "✓ Sessions directory is writable" || log "❌ Sessions directory is not writable"
else
    mkdir -p /app/sessions && chmod 755 /app/sessions && log "✓ Created sessions directory"
fi

# Set proper permissions (ignore errors for mounted volumes)
chmod -R 755 evidence config logs cheating_evidence models 2>/dev/null || log "⚠️  Permission setting skipped for mounted volumes"

# Verify and download YOLO models if missing
log "🔍 Checking YOLO models..."
if [ ! -f "models/yolov8n.pt" ]; then
    log "⬇️  Downloading YOLOv8n model..."
    wget -q --timeout=60 --tries=2 -O "models/yolov8n.pt" \
        "https://github.com/ultralytics/assets/releases/download/v8.0.0/yolov8n.pt" || \
        log "⚠️  Warning: Could not download yolov8n.pt - will use fallback detection"
else
    log "✓ YOLOv8n model available ($(du -h models/yolov8n.pt | cut -f1))"
fi

# Set environment variables for Docker
export DEPLOYMENT_MODE=docker
export FLASK_ENV=${FLASK_ENV:-production}
export PYTHONPATH=/app
export OPENCV_LOG_LEVEL=ERROR
export TORCH_HOME=/app/models

# Quick environment check
log "🔧 Checking Python environment..."
python -c "
import sys
print(f'Python version: {sys.version.split()[0]}')

# Quick package availability check
packages = ['cv2', 'torch', 'flask', 'numpy']
for pkg in packages:
    try:
        __import__(pkg)
        print(f'✓ {pkg} available')
    except ImportError:
        print(f'✗ {pkg} not available')

# Test basic OpenCV
import cv2
import numpy as np
img = np.zeros((10, 10, 3), dtype=np.uint8)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print('✓ OpenCV basic operations working')

# Test PyTorch
import torch
print(f'✓ PyTorch device: {torch.device(\"cpu\")}')
"

log "✅ Environment setup complete. Starting application..."

# Start the application with proper error handling
log "🎯 Starting AI Proctoring Service..."

# Use gunicorn for production or python for development
if [ "$FLASK_ENV" = "production" ]; then
    log "🚀 Starting with Gunicorn (Production mode)..."
    exec gunicorn --bind 0.0.0.0:5000 --workers 2 --timeout 120 --keep-alive 2 \
         --max-requests 1000 --max-requests-jitter 50 \
         --preload --log-level info \
         --access-logfile logs/access.log \
         --error-logfile logs/error.log \
         app:app
else
    log "🔧 Starting with Flask dev server (Development mode)..."
    exec python app.py
fi
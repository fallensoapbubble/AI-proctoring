#!/bin/bash

# Ultra-Optimized Build script for AI Proctoring System Docker image
# This script builds the Docker image with maximum performance optimizations

set -e

echo "🚀 Building AI Proctoring System Docker Image (Optimized)..."

# Set build arguments
BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ')
GIT_COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")

# Enable BuildKit and multi-platform builds
export DOCKER_BUILDKIT=1
export COMPOSE_DOCKER_CLI_BUILD=1

# Clean up previous builds to save space
echo "🧹 Cleaning up previous builds..."
docker system prune -f --volumes || true

# Pre-download models to avoid download during build
echo "📥 Pre-downloading ML models..."
mkdir -p models
if [ ! -f "models/yolov5s.pt" ]; then
    echo "⬇️  Downloading YOLOv5s..."
    wget -q -O models/yolov5s.pt https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt || echo "Warning: YOLOv5s download failed"
fi
if [ ! -f "models/yolov8n.pt" ]; then
    echo "⬇️  Downloading YOLOv8n..."
    wget -q -O models/yolov8n.pt https://github.com/ultralytics/assets/releases/download/v8.0.0/yolov8n.pt || echo "Warning: YOLOv8n download failed"
fi

# Build with optimizations
echo "🔨 Building Docker image with optimizations..."
docker build \
    --build-arg BUILD_DATE="$BUILD_DATE" \
    --build-arg GIT_COMMIT="$GIT_COMMIT" \
    --build-arg BUILDKIT_INLINE_CACHE=1 \
    --cache-from ai-proctoring:latest \
    --tag ai-proctoring:latest \
    --tag ai-proctoring:$GIT_COMMIT \
    --progress=plain \
    .

echo "✅ Build completed successfully!"
echo "📦 Image tagged as: ai-proctoring:latest and ai-proctoring:$GIT_COMMIT"

# Show image size
echo "📏 Image size:"
docker images ai-proctoring:latest --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}"

# Optional: Run the container
read -p "🤔 Do you want to run the container now? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🏃 Starting container with docker-compose..."
    docker-compose up -d
    echo "⏳ Waiting for application to start..."
    sleep 10
    echo "🌐 Application should be available at http://localhost:5000"
    echo "📊 Check logs with: docker-compose logs -f"
fi
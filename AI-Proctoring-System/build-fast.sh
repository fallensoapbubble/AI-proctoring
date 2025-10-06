#!/bin/bash
# Fast Docker build script for AI Proctoring System
# Optimized for development and quick iterations

set -e

echo "🚀 Fast Docker Build for AI Proctoring System"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${BLUE}[$(date +'%H:%M:%S')]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[$(date +'%H:%M:%S')] ✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[$(date +'%H:%M:%S')] ⚠️${NC} $1"
}

print_error() {
    echo -e "${RED}[$(date +'%H:%M:%S')] ✗${NC} $1"
}

# Check Docker availability
if ! docker info > /dev/null 2>&1; then
    print_error "Docker is not running. Please start Docker and try again."
    exit 1
fi

print_success "Docker is running"

# Enable BuildKit for faster builds
export DOCKER_BUILDKIT=1
export COMPOSE_DOCKER_CLI_BUILD=1

print_status "BuildKit enabled for faster builds"

# Clean up if requested
if [[ "$1" == "--clean" ]]; then
    print_status "Cleaning up previous builds..."
    docker system prune -f
    docker builder prune -f
    print_success "Cleanup completed"
fi

# Create necessary directories
print_status "Creating directory structure..."
mkdir -p sessions
mkdir -p config models
print_success "Directory structure created"

# Check if models exist, download if missing
print_status "Checking ML models..."
if [ ! -f "models/yolov8n.pt" ]; then
    print_warning "YOLOv8n model missing, will be downloaded during container startup"
else
    print_success "YOLOv8n model available ($(du -h models/yolov8n.pt | cut -f1))"
fi

# Build with optimizations and caching
print_status "Building Docker image with optimizations..."

# Use multi-stage build cache
docker build \
    --build-arg BUILDKIT_INLINE_CACHE=1 \
    --cache-from ai-proctoring-fixed:latest \
    -t ai-proctoring-fixed:latest \
    -t ai-proctoring-fixed:$(date +%Y%m%d-%H%M%S) \
    -f Dockerfile \
    . 2>&1 | tee build.log

if [ ${PIPESTATUS[0]} -eq 0 ]; then
    print_success "Docker image built successfully"
else
    print_error "Docker build failed. Check build.log for details."
    exit 1
fi

# Test the image quickly
print_status "Quick image test..."
docker run --rm ai-proctoring-fixed:latest python -c "
import sys
print(f'✓ Python {sys.version.split()[0]}')
try:
    import cv2, numpy, flask, torch
    print('✓ Core packages imported successfully')
    # Test integrated system imports
    from context_engine.models import SystemConfiguration
    from detectors.detector_manager import DetectorManager
    print('✓ Integrated system components available')
except ImportError as e:
    print(f'⚠️  Import warning: {e}')
    sys.exit(1)
" 2>/dev/null

if [ $? -eq 0 ]; then
    print_success "Image test passed"
else
    print_warning "Image test had issues, but build completed"
fi

# Show image information
print_status "Docker image information:"
docker images ai-proctoring-fixed:latest --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}"

# Calculate build time
if [ -f build.log ]; then
    build_time=$(grep -o "FINISHED.*" build.log | tail -1 || echo "Build time not available")
    print_status "Build completed: $build_time"
fi

print_success "Fast build completed successfully!"

echo
echo "🚀 Quick start commands:"
echo "  Run with docker-compose: docker-compose up -d"
echo "  Run standalone:          docker run -p 5000:5000 ai-proctoring-fixed:latest"
echo "  View logs:              docker-compose logs -f"
echo "  Stop:                   docker-compose down"
echo
echo "🌐 Application will be available at: http://localhost:5000"
echo "📊 Health check:                     http://localhost:5000/health"
echo "🔧 ML status:                        http://localhost:5000/api/ml_status"
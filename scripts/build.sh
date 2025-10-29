#!/bin/bash
# build.sh - Automated build script

set -e

echo "==================================="
echo "CUDA Parallel Image Pipeline Build"
echo "==================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check for CUDA
echo -n "Checking for CUDA... "
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release //' | sed 's/,.*//')
    echo -e "${GREEN}✓${NC} Found CUDA $CUDA_VERSION"
else
    echo -e "${RED}✗${NC} CUDA not found!"
    echo "Please install CUDA toolkit from https://developer.nvidia.com/cuda-downloads"
    exit 1
fi

# Check for OpenCV
echo -n "Checking for OpenCV... "
if pkg-config --exists opencv4; then
    OPENCV_VERSION=$(pkg-config --modversion opencv4)
    echo -e "${GREEN}✓${NC} Found OpenCV $OPENCV_VERSION"
else
    echo -e "${RED}✗${NC} OpenCV 4 not found!"
    echo "Install with: sudo apt install libopencv-dev"
    exit 1
fi

# Check for GPU
echo -n "Checking for NVIDIA GPU... "
if command -v nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
    echo -e "${GREEN}✓${NC} Found: $GPU_NAME"
else
    echo -e "${YELLOW}⚠${NC} nvidia-smi not available"
fi

echo ""
echo "Creating directories..."
mkdir -p bin obj data/input data/output data/benchmarks

echo "Building project..."
echo ""

# Detect CUDA architecture if not specified
if [ -z "$CUDA_ARCH" ]; then
    echo "Auto-detecting CUDA architecture..."
    if command -v nvidia-smi &> /dev/null; then
        # Simple heuristic based on GPU name
        if nvidia-smi | grep -q "RTX 30\|A100\|A40"; then
            export CUDA_ARCH=sm_86
            echo "Detected Ampere GPU, using sm_86"
        elif nvidia-smi | grep -q "RTX 20\|T4\|Turing"; then
            export CUDA_ARCH=sm_75
            echo "Detected Turing GPU, using sm_75"
        elif nvidia-smi | grep -q "GTX 10\|P100\|Pascal"; then
            export CUDA_ARCH=sm_60
            echo "Detected Pascal GPU, using sm_60"
        else
            export CUDA_ARCH=sm_75
            echo "Unknown GPU, defaulting to sm_75"
        fi
    else
        export CUDA_ARCH=sm_75
        echo "Cannot detect GPU, defaulting to sm_75"
    fi
fi

# Build using make
echo "Compiling with CUDA_ARCH=$CUDA_ARCH..."
make clean
make -j$(nproc)

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✓ Build successful!${NC}"
    echo ""
    echo "Executable: bin/cuda_image_pipeline"
    echo ""
    echo "Quick start:"
    echo "  1. Place images in data/input/"
    echo "  2. Run: ./bin/cuda_image_pipeline -i data/input -o data/output -f gaussian_blur"
    echo ""
    echo "For help: ./bin/cuda_image_pipeline --help"
    echo "Run tests: make test"
    echo "Benchmark: make benchmark"
else
    echo -e "${RED}✗ Build failed!${NC}"
    exit 1
fi
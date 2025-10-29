
set -e

EXECUTABLE="./bin/cuda_image_pipeline"
INPUT_DIR="data/input"
OUTPUT_BASE="data/output"
METRICS_DIR="data/benchmarks"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "==========================================="
echo "CUDA Image Pipeline - Benchmark Suite"
echo "==========================================="
echo ""

# Check executable
if [ ! -f "$EXECUTABLE" ]; then
    echo "Error: Executable not found at $EXECUTABLE"
    echo "Please run ./build.sh first"
    exit 1
fi

# Check for input images
IMAGE_COUNT=$(find "$INPUT_DIR" -type f \( -iname "*.jpg" -o -iname "*.png" -o -iname "*.bmp" \) 2>/dev/null | wc -l)
if [ "$IMAGE_COUNT" -eq 0 ]; then
    echo "Error: No images found in $INPUT_DIR"
    echo "Please add test images or run: ./scripts/download_dataset.sh"
    exit 1
fi

echo "Found $IMAGE_COUNT images in $INPUT_DIR"
echo ""

# Create output directories
mkdir -p "$METRICS_DIR"

# Filters to test
FILTERS=("gaussian_blur" "box_blur" "sobel" "sharpen" "emboss")

# Batch sizes to test
BATCH_SIZES=(8 16 32 64)

# Results file
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_FILE="$METRICS_DIR/benchmark_results_${TIMESTAMP}.txt"
METRICS_JSON="$METRICS_DIR/metrics_${TIMESTAMP}.json"

echo "Starting benchmark suite..." | tee "$RESULTS_FILE"
echo "Timestamp: $(date)" | tee -a "$RESULTS_FILE"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)" | tee -a "$RESULTS_FILE"
echo "Image count: $IMAGE_COUNT" | tee -a "$RESULTS_FILE"
echo "========================================" | tee -a "$RESULTS_FILE"
echo "" | tee -a "$RESULTS_FILE"

# Test each filter
for FILTER in "${FILTERS[@]}"; do
    echo -e "${BLUE}Testing filter: $FILTER${NC}" | tee -a "$RESULTS_FILE"
    
    for BATCH_SIZE in "${BATCH_SIZES[@]}"; do
        OUTPUT_DIR="${OUTPUT_BASE}/${FILTER}_batch${BATCH_SIZE}"
        mkdir -p "$OUTPUT_DIR"
        
        echo -n "  Batch size $BATCH_SIZE: " | tee -a "$RESULTS_FILE"
        
        # Run benchmark
        METRICS_FILE="$METRICS_DIR/${FILTER}_batch${BATCH_SIZE}.json"
        
        "$EXECUTABLE" \
            --input "$INPUT_DIR" \
            --output "$OUTPUT_DIR" \
            --filter "$FILTER" \
            --batch-size "$BATCH_SIZE" \
            --metrics-file "$METRICS_FILE" \
            > /tmp/cuda_bench_output.txt 2>&1
        
        # Extract throughput from metrics
        if [ -f "$METRICS_FILE" ]; then
            THROUGHPUT=$(grep "throughput_img_per_sec" "$METRICS_FILE" | sed 's/.*: //' | sed 's/,//')
            AVG_TIME=$(grep "avg_time_per_image_ms" "$METRICS_FILE" | sed 's/.*: //' | sed 's/,//')
            echo -e "${GREEN}${THROUGHPUT} img/s${NC} (avg: ${AVG_TIME} ms/img)" | tee -a "$RESULTS_FILE"
        else
            echo "Failed" | tee -a "$RESULTS_FILE"
        fi
    done
    
    echo "" | tee -a "$RESULTS_FILE"
done

echo "========================================" | tee -a "$RESULTS_FILE"
echo "Benchmark complete!" | tee -a "$RESULTS_FILE"
echo "Results saved to: $RESULTS_FILE"
echo "Detailed metrics in: $METRICS_DIR"
echo ""

# Generate summary report
echo "Generating summary report..."
cat > "$METRICS_DIR/summary_${TIMESTAMP}.md" << 'EOF'
# CUDA Image Pipeline - Benchmark Summary

## Test Configuration
- **Date**: $(date)
- **GPU**: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
- **CUDA Version**: $(nvcc --version | grep "release" | sed 's/.*release //' | sed 's/,.*//')
- **Images Tested**: $(find "$INPUT_DIR" -type f \( -iname "*.jpg" -o -iname "*.png" \) | wc -l)

## Performance Results

Filter performance across different batch sizes:

| Filter | Batch 8 | Batch 16 | Batch 32 | Batch 64 |
|--------|---------|----------|----------|----------|
EOF

# Add results to table
for FILTER in "${FILTERS[@]}"; do
    echo -n "| $FILTER " >> "$METRICS_DIR/summary_${TIMESTAMP}.md"
    for BATCH_SIZE in "${BATCH_SIZES[@]}"; do
        METRICS_FILE="$METRICS_DIR/${FILTER}_batch${BATCH_SIZE}.json"
        if [ -f "$METRICS_FILE" ]; then
            THROUGHPUT=$(grep "throughput_img_per_sec" "$METRICS_FILE" | sed 's/.*: //' | sed 's/,//')
            echo -n "| ${THROUGHPUT} " >> "$METRICS_DIR/summary_${TIMESTAMP}.md"
        else
            echo -n "| N/A " >> "$METRICS_DIR/summary_${TIMESTAMP}.md"
        fi
    done
    echo "|" >> "$METRICS_DIR/summary_${TIMESTAMP}.md"
done

echo ""
echo "Summary report: $METRICS_DIR/summary_${TIMESTAMP}.md"
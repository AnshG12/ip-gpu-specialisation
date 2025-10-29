# ===================================================================
# download_dataset.sh - Download test images
# ===================================================================

#!/bin/bash
# download_dataset.sh - Download sample images for testing

set -e

TARGET_DIR="data/input"
COUNT=100

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --count)
            COUNT="$2"
            shift 2
            ;;
        --dir)
            TARGET_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--count N] [--dir PATH]"
            exit 1
            ;;
    esac
done

echo "Downloading $COUNT sample images to $TARGET_DIR..."
mkdir -p "$TARGET_DIR"

# Download from public image databases
# Using Unsplash API for diverse high-quality images
# Note: For production use, obtain an API key from unsplash.com

SAMPLE_URLS=(
    "https://picsum.photos/1920/1080"
    "https://picsum.photos/1280/720"
    "https://picsum.photos/1024/768"
    "https://picsum.photos/800/600"
)

echo "Downloading images..."
for ((i=1; i<=COUNT; i++)); do
    # Rotate through different resolutions
    URL_INDEX=$((i % ${#SAMPLE_URLS[@]}))
    URL="${SAMPLE_URLS[$URL_INDEX]}"
    
    OUTPUT_FILE="$TARGET_DIR/sample_$(printf "%04d" $i).jpg"
    
    if [ ! -f "$OUTPUT_FILE" ]; then
        wget -q -O "$OUTPUT_FILE" "$URL?random=$i" || curl -s -o "$OUTPUT_FILE" "$URL?random=$i"
        echo "Downloaded: $OUTPUT_FILE ($i/$COUNT)"
    else
        echo "Skipped: $OUTPUT_FILE (already exists)"
    fi
    
    # Rate limiting
    sleep 0.5
done

echo ""
echo "Download complete!"
echo "Total images: $(ls -1 "$TARGET_DIR" | wc -l)"
echo "Directory size: $(du -sh "$TARGET_DIR" | cut -f1)"
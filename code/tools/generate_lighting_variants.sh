#!/bin/bash

# Script to generate lighting variants for input videos
# Usage: ./generate_lighting_variants.sh [input-video-name]

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default input
INPUT_NAME=${1:-"input-01"}
INPUT_FILE="data/input/${INPUT_NAME}.mp4"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Lighting Variants Generator${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check if input exists
if [ ! -f "$INPUT_FILE" ]; then
    echo -e "${YELLOW}Error: Input file not found: $INPUT_FILE${NC}"
    exit 1
fi

echo -e "${GREEN}Input video: $INPUT_FILE${NC}"
echo ""

# Generate low-light variant
echo -e "${BLUE}[1/2] Generating low-light variant...${NC}"
python tools/lighting_simulator.py \
    --input "$INPUT_FILE" \
    --mode low-light \
    --comparison

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Low-light variant created${NC}"
else
    echo -e "${YELLOW}✗ Failed to create low-light variant${NC}"
    exit 1
fi
echo ""

# Generate night variant
echo -e "${BLUE}[2/2] Generating night variant...${NC}"
python tools/lighting_simulator.py \
    --input "$INPUT_FILE" \
    --mode night \
    --comparison

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Night variant created${NC}"
else
    echo -e "${YELLOW}✗ Failed to create night variant${NC}"
    exit 1
fi
echo ""

# Summary
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}Generation Complete!${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "Generated files:"
echo "  - data/input/${INPUT_NAME}-low-light.mp4"
echo "  - data/input/${INPUT_NAME}-low-light_comparison.jpg"
echo "  - data/input/${INPUT_NAME}-night.mp4"
echo "  - data/input/${INPUT_NAME}-night_comparison.jpg"
echo ""
echo "Next steps:"
echo "  1. Review comparison images to verify quality"
echo "  2. Run system with appropriate config:"
echo ""
echo "     # Normal lighting"
echo "     python src/main.py --config config/config-normal.yaml"
echo ""
echo "     # Low-light"
echo "     python src/main.py --config config/config-lowlight.yaml"
echo ""
echo "     # Night"
echo "     python src/main.py --config config/config-night.yaml"
echo ""

#!/bin/bash

# Script to test intrusion detection system with all lighting conditions
# Usage: ./test_all_lighting.sh

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Lighting Conditions Test Suite${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Test counter
TOTAL_TESTS=3
PASSED_TESTS=0

# Test 1: Normal lighting
echo -e "${BLUE}[1/3] Testing NORMAL lighting conditions...${NC}"
echo -e "${YELLOW}Config: config/config-normal.yaml${NC}"
echo ""

python src/main.py --config config/config-normal.yaml

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Normal lighting test PASSED${NC}"
    ((PASSED_TESTS++))
else
    echo -e "${RED}✗ Normal lighting test FAILED${NC}"
fi
echo ""
echo "Press Enter to continue to next test..."
read
echo ""

# Test 2: Low-light
echo -e "${BLUE}[2/3] Testing LOW-LIGHT conditions...${NC}"
echo -e "${YELLOW}Config: config/config-lowlight.yaml${NC}"
echo ""

# Check if low-light video exists
if [ ! -f "data/input/input-01-low-light.mp4" ]; then
    echo -e "${YELLOW}Warning: Low-light video not found!${NC}"
    echo "Generating now..."
    python tools/lighting_simulator.py --input data/input/input-01.mp4 --mode low-light
    echo ""
fi

python src/main.py --config config/config-lowlight.yaml

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Low-light test PASSED${NC}"
    ((PASSED_TESTS++))
else
    echo -e "${RED}✗ Low-light test FAILED${NC}"
fi
echo ""
echo "Press Enter to continue to next test..."
read
echo ""

# Test 3: Night
echo -e "${BLUE}[3/3] Testing NIGHT conditions...${NC}"
echo -e "${YELLOW}Config: config/config-night.yaml${NC}"
echo ""

# Check if night video exists
if [ ! -f "data/input/input-01-night.mp4" ]; then
    echo -e "${YELLOW}Warning: Night video not found!${NC}"
    echo "Generating now..."
    python tools/lighting_simulator.py --input data/input/input-01.mp4 --mode night
    echo ""
fi

python src/main.py --config config/config-night.yaml

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Night test PASSED${NC}"
    ((PASSED_TESTS++))
else
    echo -e "${RED}✗ Night test FAILED${NC}"
fi
echo ""

# Summary
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Test Summary${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "Tests passed: ${GREEN}${PASSED_TESTS}/${TOTAL_TESTS}${NC}"
echo ""

if [ $PASSED_TESTS -eq $TOTAL_TESTS ]; then
    echo -e "${GREEN}All tests passed! ✓${NC}"
    echo ""
    echo "Output locations:"
    echo "  - data/output/input-01/result-normal.mp4"
    echo "  - data/output/input-01-lowlight/result-lowlight.mp4"
    echo "  - data/output/input-01-night/result-night.mp4"
    echo ""
    echo "Alert logs:"
    echo "  - data/output/input-01/alerts.log"
    echo "  - data/output/input-01-lowlight/alerts.log"
    echo "  - data/output/input-01-night/alerts.log"
else
    echo -e "${YELLOW}Some tests failed. Check the output above.${NC}"
fi
echo ""

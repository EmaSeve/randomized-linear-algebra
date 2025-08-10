#!/bin/bash

# Build script for StochasticLA library

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Building randomized-linear-algebra library...${NC}"

# Create build directory if it doesn't exist
if [ ! -d "build" ]; then
    mkdir build
fi

cd build

# Configure with CMake
echo -e "${YELLOW}Configuring with CMake...${NC}"
cmake .. -DCMAKE_BUILD_TYPE=Release

if [ $? -ne 0 ]; then
    echo -e "${RED}CMake configuration failed!${NC}"
    exit 1
fi

# Build
echo -e "${YELLOW}Building...${NC}"
make -j$(nproc)

if [ $? -ne 0 ]; then
    echo -e "${RED}Build failed!${NC}"
    exit 1
fi

# Run tests
echo -e "${YELLOW}Running tests...${NC}"
ctest --output-on-failure

if [ $? -eq 0 ]; then
    echo -e "${GREEN}Build and tests completed successfully!${NC}"
else
    echo -e "${RED}Some tests failed!${NC}"
    exit 1
fi

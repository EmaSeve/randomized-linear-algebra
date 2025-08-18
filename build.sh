#!/bin/bash

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Flags
RUN_BENCHMARK=false
ENABLE_OPENMP=ON

# Parse args
for arg in "$@"; do
    case "$arg" in
        --benchmark) RUN_BENCHMARK=true ;;
        --no-openmp) ENABLE_OPENMP=OFF ;;
    esac
done

echo -e "${YELLOW}Configuring build (OpenMP=${ENABLE_OPENMP})...${NC}"

mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DENABLE_OPENMP=${ENABLE_OPENMP} || exit 1

echo -e "${YELLOW}Compiling...${NC}"
make -j$(nproc) || exit 1

if [ "$RUN_BENCHMARK" = true ]; then
    echo -e "${YELLOW}Running benchmark...${NC}"
    ./benchmark_fixed_rank || exit 1
else
    echo -e "${YELLOW}Running tests...${NC}"
    ctest --output-on-failure || exit 1
fi

echo -e "${GREEN}Done!${NC}"

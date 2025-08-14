#!/bin/bash

# Build script for StochasticLA library

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Building randomized-linear-algebra library...${NC}"

# Parse args
THREADS=""
ENABLE_OPENMP=1
GENERATOR_THREADS=$(nproc)
while [[ $# -gt 0 ]]; do
    case "$1" in
        -t|--threads)
            THREADS="$2"; shift 2 ;;
        --no-openmp)
            ENABLE_OPENMP=0; shift ;;
        -j|--jobs)
            GENERATOR_THREADS="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 [-t|--threads N] [--no-openmp] [-j|--jobs N]";
            echo "  -t, --threads N   Imposta OMP_NUM_THREADS in modo uniforme per build ed esecuzione test";
            echo "      --no-openmp   Disabilita OpenMP in compilazione (CMake ENABLE_OPENMP=OFF)";
            echo "  -j, --jobs N      Numero di job per make (default: nproc)";
            exit 0 ;;
        *) echo -e "${YELLOW}Ignoring unknown arg: $1${NC}"; shift ;;
    esac
done

# Create build directory if it doesn't exist
if [ ! -d "build" ]; then
    mkdir build
fi

cd build

# Configure with CMake
echo -e "${YELLOW}Configuring with CMake...${NC}"
cmake .. -DCMAKE_BUILD_TYPE=Release -DENABLE_OPENMP=$ENABLE_OPENMP

if [ $? -ne 0 ]; then
    echo -e "${RED}CMake configuration failed!${NC}"
    exit 1
fi

# Build
echo -e "${YELLOW}Building...${NC}"
make -j"$GENERATOR_THREADS"

if [ $? -ne 0 ]; then
    echo -e "${RED}Build failed!${NC}"
    exit 1
fi

# Run tests (and optionally set uniform OMP threads for everything)
if [[ -n "$THREADS" ]]; then
    export OMP_NUM_THREADS="$THREADS"
    echo -e "${YELLOW}Using OMP_NUM_THREADS=$OMP_NUM_THREADS for build and tests...${NC}"
fi

echo -e "${YELLOW}Running tests...${NC}"
ctest --output-on-failure

if [ $? -eq 0 ]; then
    echo -e "${GREEN}Build and tests completed successfully!${NC}"
else
    echo -e "${RED}Some tests failed!${NC}"
    exit 1
fi

#!/bin/bash

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Flags
RUN_BENCHMARK=false
RUN_FIXED_RANK=false
RUN_FIXED_PRECISION=false
RUN_TESTS=true
ENABLE_OPENMP=ON

# Parse args
while [ $# -gt 0 ]; do
    case "$1" in
        --benchmark)
            RUN_BENCHMARK=true
            # Check if next parameter specifies benchmark type
            if [ $# -gt 1 ] && [[ ! "$2" =~ ^-- ]]; then
                case "$2" in
                    fr)
                        RUN_FIXED_RANK=true
                        shift
                        ;;
                    fp)
                        RUN_FIXED_PRECISION=true
                        shift
                        ;;
                    *)
                        # If not recognized, run all benchmarks
                        RUN_FIXED_RANK=true
                        RUN_FIXED_PRECISION=true
                        ;;
                esac
            else
                # If --benchmark is the last parameter, run all benchmarks
                RUN_FIXED_RANK=true
                RUN_FIXED_PRECISION=true
            fi
            ;;
        --no-openmp) ENABLE_OPENMP=OFF ;;
        --no-test) RUN_TESTS=false ;;
    esac
    shift
done

echo -e "${YELLOW}Configuring build (OpenMP=${ENABLE_OPENMP})...${NC}"

mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DENABLE_OPENMP=${ENABLE_OPENMP} || exit 1

echo -e "${YELLOW}Compiling...${NC}"
make -j$(nproc) || exit 1

if [ "$RUN_BENCHMARK" = true ]; then
    if [ "$RUN_FIXED_RANK" = true ]; then
        echo -e "${YELLOW}Running fixed rank benchmark...${NC}"
        ./benchmark_fixed_rank || exit 1
    fi
    
    if [ "$RUN_FIXED_PRECISION" = true ]; then
        echo -e "${YELLOW}Running fixed precision benchmark...${NC}"
        ./benchmark_fixed_precision || exit 1
    fi
elif [ "$RUN_TESTS" = true ]; then
    echo -e "${YELLOW}Running tests...${NC}"
    ctest --output-on-failure || exit 1
else
    echo -e "${GREEN}Skipping tests and benchmarks.${NC}"
fi

echo -e "${GREEN}Done!${NC}"

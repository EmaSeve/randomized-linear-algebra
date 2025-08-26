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
THREADING_MODE="openmp" # default


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
        --no-test)
            RUN_TESTS=false
            ;;
        --threading)
            if [ $# -gt 1 ] && [[ ! "$2" =~ ^-- ]]; then
                THREADING_MODE="$2"
                shift
            else
                echo -e "${RED}Missing argument for --threading. Use blas, openmp, or single.${NC}"
                exit 1
            fi
            ;;
    esac
    shift
done

echo -e "${YELLOW}Configuring build...${NC}"

mkdir -p build && cd build


if [ "$RUN_BENCHMARK" = true ]; then
    cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=OFF -DBUILD_BENCHMARKS=ON -DTHREADING_MODE=${THREADING_MODE} || exit 1
elif [ "$RUN_TESTS" = false ]; then
    cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=OFF -DBUILD_BENCHMARKS=OFF -DTHREADING_MODE=${THREADING_MODE} || exit 1
else
    cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=ON -DBUILD_BENCHMARKS=OFF -DTHREADING_MODE=${THREADING_MODE} || exit 1
fi

echo -e "${YELLOW}Compiling...${NC}"
make -j$(nproc) || exit 1

if [ "$RUN_BENCHMARK" = true ]; then
    if [ "$RUN_FIXED_RANK" = true ]; then
        echo -e "${YELLOW}Running fixed rank benchmark...${NC}"
        ./benchmark_fixed_rank_A || exit 1
    fi
    
    if [ "$RUN_FIXED_PRECISION" = true ]; then
        echo -e "${YELLOW}Running fixed precision benchmark...${NC}"
        ./benchmark_fixed_precision_A || exit 1
    fi
elif [ "$RUN_TESTS" = true ]; then
    echo -e "${YELLOW}Running tests...${NC}"
    ctest --output-on-failure || exit 1
else
    echo -e "${GREEN}Skipping tests and benchmarks.${NC}"
fi

echo -e "${GREEN}Done!${NC}"

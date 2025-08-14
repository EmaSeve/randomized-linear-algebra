#!/bin/bash

# Build script for randomized-linear-algebra

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Building randomized-linear-algebra library...${NC}"

# Parse args
THREADS=""
BENCHMARK_LIST=""
ENABLE_OPENMP=1
GENERATOR_THREADS=$(nproc)
while [[ $# -gt 0 ]]; do
    case "$1" in
        -t|--threads)
            THREADS="$2"; shift 2 ;;
        --benchmark)
            BENCHMARK_LIST="$2"; shift 2 ;;
        --no-openmp)
            ENABLE_OPENMP=0; shift ;;
        -j|--jobs)
            GENERATOR_THREADS="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 [-t|--threads N] [--no-openmp] [-j|--jobs N]";
            echo "  -t, --threads N   Imposta OMP_NUM_THREADS in modo uniforme per build ed esecuzione test";
            echo "      --benchmark L Lista di thread per il benchmark (es: 1,2,4,8). In questo modo NON vengono eseguiti i test.";
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

if [[ -n "$BENCHMARK_LIST" ]]; then
    # Benchmark mode: esegue benchmark con i thread specificati e salta i test
    echo -e "${YELLOW}Benchmark mode attivo. Thread list: $BENCHMARK_LIST${NC}"
    if [[ ! -x ./rla_benchmark ]]; then
        echo -e "${YELLOW}Benchmark target non trovato, provo a costruirlo...${NC}"
        make -j"$GENERATOR_THREADS" rla_benchmark || { echo -e "${RED}Build del benchmark fallita!${NC}"; exit 1; }
    fi

    # Split comma-separated list and run
    IFS=',' read -r -a _bench_threads <<< "$BENCHMARK_LIST"
    exit_code=0
    for t in "${_bench_threads[@]}"; do
        t_trimmed="${t//[[:space:]]/}"
        if [[ -z "$t_trimmed" ]]; then continue; fi
        if ! [[ "$t_trimmed" =~ ^[0-9]+$ ]]; then
            echo -e "${YELLOW}Valore thread non valido: '$t_trimmed' (ignoro)${NC}"
            continue
        fi
        export OMP_NUM_THREADS="$t_trimmed"
        echo -e "${YELLOW}Eseguo benchmark con OMP_NUM_THREADS=$OMP_NUM_THREADS...${NC}"
        ./rla_benchmark
        rc=$?
        if [[ $rc -ne 0 ]]; then
            echo -e "${RED}Benchmark fallito con OMP_NUM_THREADS=$OMP_NUM_THREADS (exit $rc)${NC}"
            exit_code=$rc
        fi
    done
    if [[ $exit_code -eq 0 ]]; then
        echo -e "${GREEN}Benchmark completato con successo per tutti i thread!${NC}"
    fi
    exit $exit_code
fi

echo -e "${YELLOW}Running tests...${NC}"
ctest --output-on-failure

if [ $? -eq 0 ]; then
    echo -e "${GREEN}Build and tests completed successfully!${NC}"
else
    echo -e "${RED}Some tests failed!${NC}"
    exit 1
fi

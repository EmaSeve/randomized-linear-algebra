# Building and testing randomized-linear-algebra

This document explains how to build the project, run the tests and (optionally) the benchmark, with or without OpenMP.

## Requirements

- CMake >= 3.15
- C++17 compiler (g++/clang++)
- Eigen3 (header-only)
- FFTW3 (double precision)
- OpenMP (optional)
- pkg-config (optional, helps find FFTW)

## Build script

The `build.sh` script configures in Release mode, builds, and then runs either the tests or the benchmark.

```bash
# from the repo root
chmod +x build.sh
./build.sh              # builds and runs the tests
./build.sh --benchmark  # builds and runs the benchmark instead of the tests
./build.sh --no-openmp  # disables OpenMP in configuration and build
```

Notes:
- By default the script enables OpenMP if available.
- Build output in `./build/`.

## OpenMP

- Enable/disable at configure time: `-DENABLE_OPENMP=ON|OFF`.
- If not found, the build proceeds without OpenMP (an informational message will be shown).
- To control threads at runtime: `OMP_NUM_THREADS=8 ./test_fixed_rank` (adjust the number to your CPUs).

## Dependency detection and useful variables

The project automatically tries to find Eigen3 and FFTW3:

- Eigen3: via `find_package(Eigen3)` or via include path.
- FFTW3: preferably via `pkg-config`; alternatively provide header and library via manual paths.

If auto-detection fails, you can help CMake by passing paths explicitly:

```bash
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DEIGEN3_INCLUDE_DIR=/path/to/eigen \
  -DFFTW3_INCLUDE_DIR=/path/to/fftw/include \
  -DFFTW3_LIB=/path/to/fftw/lib/libfftw3.so
```

Alternatively, you can use environment variables that CMake treats as hints:
- `CPATH` for header includes (e.g., Eigen/FFTW)
- `LIBRARY_PATH` or `LD_LIBRARY_PATH` for libraries (e.g., FFTW)

Example:

```bash
export CPATH=/usr/local/include:$CPATH
export LIBRARY_PATH=/usr/local/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
```

## Generated executables

After building, in `build/` you will find:
- `test_fixed_rank`
- `test_fixed_precision`
- `RRF_benchmark` (if `benchmark/benchmark.cpp` exists)

Examples:

```bash
# from the build/ folder
./test_fixed_rank
./test_fixed_precision
./RRF_benchmark
```

# Randomized Linear Algebra (RandLA)

Header-only C++17 library implementing **randomized algorithms for low-rank matrix approximation**.  
It provides methods for *range finding* (Stage A) and *matrix factorizations* (Stage B), 
integrating seamlessly with [Eigen](https://eigen.tuxfamily.org/), and optionally exploiting **OpenMP**, **OpenBLAS**, and **FFTW** for performance.

The implementation follows the algorithms described in:
- N. Halko, P.-G. Martinsson, J. A. Tropp. *Finding structure with randomness: Probabilistic algorithms for constructing approximate matrix decompositions*. SIAM Review, 2011.  
- E. Liberty, F. Woolfe, P.-G. Martinsson, V. Rokhlin, M. Tygert. *Randomized algorithms for the low-rank approximation of matrices*. PNAS, 2007.

---

## Features

- **Stage A (Range Approximation):**
  - Randomized Range Finder, Power Iteration, Subspace Iteration
  - Fast Randomized Range Finder (SRFT, via FFTW)
  - Adaptive randomized schemes (fixed-precision stopping criteria)

- **Stage B (Factorizations):**
  - Direct SVD, Eigenvalue decomposition
  - Nyström method for PSD matrices
  - Interpolative Decomposition (ID)
  - Single-pass methods for large/streaming data

- **Utilities:**
  - Synthetic matrix generators (low-rank, exponential decay, PSD, sparse, etc.)
  - Error estimators (posterior bounds, spectral norm approximation)
  - Random Gaussian/complex matrices and vectors
  - Threading wrapper for OpenMP / BLAS

---

## Project Structure

```
include/randla/      # Core headers
 ├── types.hpp                 # Common type aliases
 ├── aliases.hpp               # Ready-to-use typedefs (float, double, long double)
 ├── randla.hpp                # Umbrella header (include <randla/randla.hpp>)
 ├── algorithms/               # Stage A & Stage B algorithms
 ├── metrics/error_estimators.hpp
 ├── random/random_generator.hpp
 ├── utils/matrix_generators.hpp
 └── threading/threading.hpp

tests/               # GoogleTest unit tests
benchmark/           # Benchmark executables
build.sh             # Helper script (configure + build + test/benchmark)
CMakeLists.txt       # Build configuration
plot_benchmark.py    # Python script for plotting benchmark results
```

---

## Build & Installation

### Requirements
- C++17 compiler
- [Eigen3](https://eigen.tuxfamily.org/) (header-only)
- [CMake ≥ 3.15](https://cmake.org)
- [FFTW3](http://www.fftw.org/) (for SRFT algorithms)
- Optional: [OpenMP](https://www.openmp.org/) / [OpenBLAS](https://www.openblas.net/)

### Defaults
- **Threading mode:** `openmp`
- **Tests:** `BUILD_TESTS=ON`
- **Benchmarks:** `BUILD_BENCHMARKS=OFF`

So by default the library builds with OpenMP enabled and compiles the 
GoogleTest-based unit tests. Benchmarks are not built unless explicitly requested.

---

### Quick start with `build.sh`

```bash
# Default: OpenMP backend + run tests
./build.sh

# BLAS backend (OpenBLAS, via pkg-config), without tests
./build.sh --no-test --threading blas

# Run benchmarks (fixed-rank only, OpenMP backend)
./build.sh --benchmark fr --threading openmp
```

Flags:
- `--no-test` → disables building/running tests  
- `--benchmark [fr|fp]` → builds and runs benchmarks (fixed-rank, fixed-precision, or both)  
- `--threading {openmp|blas|single}` → selects parallel backend  

---

### Manual build (without script)

```bash
# OpenMP backend
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DTHREADING_MODE=openmp
cmake --build build -j

# BLAS backend (OpenBLAS via pkg-config)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DTHREADING_MODE=blas
cmake --build build -j

# Serial (single-threaded)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DTHREADING_MODE=single
cmake --build build -j
```

Additional options:
- `-DBUILD_TESTS=OFF` → disable tests  
- `-DBUILD_BENCHMARKS=ON` → enable benchmarks  

Examples:
```bash
# Disable tests, build only the library
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release       -DTHREADING_MODE=openmp -DBUILD_TESTS=OFF

# Enable benchmarks (in addition to library + tests)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release       -DTHREADING_MODE=blas -DBUILD_BENCHMARKS=ON
```

---

## Usage in your project (CMake)

The library is **header-only** and exposes a CMake target 
`randomized-linear-algebra`:

```cmake
# In your CMakeLists.txt
find_package(Eigen3 QUIET NO_MODULE)
add_subdirectory(path/to/randomized-linear-algebra)

add_executable(my_app src/main.cpp)
target_link_libraries(my_app PRIVATE randomized-linear-algebra)
```

The target automatically propagates include paths and links against its 
dependencies (Eigen3, FFTW, and optionally OpenMP / OpenBLAS depending 
on the chosen backend).

---

## Minimal Example

```cpp
#include <Eigen/Dense>
#include <randla/randla.hpp>
#include <iostream>

int main() {
    Eigen::MatrixXd A = Eigen::MatrixXd::Random(300, 150);
    int l = 20, seed = 42;

    auto Q = randla::RandRangeFinderD::randomizedRangeFinder(A, l, seed);

    std::cout << "Computed basis Q of size "
              << Q.rows() << " x " << Q.cols() << std::endl;
}
```

Compile and run:

```bash
g++ -std=c++17 -Iinclude my_app.cpp -o my_app
./my_app
```

---

## Tests

The project includes a GoogleTest suite:

```bash
ctest --test-dir build --output-on-failure
```

Covers:
- Fixed-rank algorithms (range finders, power/subspace iteration, SRFT)
- Adaptive algorithms (fixed-precision stopping)
- Matrix factorizations (SVD, EVD, ID)

---

## Benchmarks

Two benchmark drivers:
- `benchmark_fixed_rank_A`
- `benchmark_fixed_precision_A`

Results are logged in CSV under `build/` and can be plotted with:

```bash
python3 plot_benchmark.py
```

This produces tables and plots in `benchmark_plots/`.

---

## Authors

- Nicola Noventa  
- Emanuele Severino  

---

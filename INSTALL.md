# StochasticLA - Randomized Linear Algebra Library

A C++ template library implementing randomized linear algebra algorithms with support for different floating-point precisions.

## Features

- **Generic Template Design**: Supports `float`, `double`, and `long double` precision
- **Eigen Integration**: Built on top of the efficient Eigen linear algebra library
- **Header-Only**: Easy to integrate into existing projects
- **Modern C++**: Uses C++17 features for better performance and safety

## Dependencies

- **CMake** (≥ 3.15)
- **Eigen3** (≥ 3.3)
- **C++17** compatible compiler

### Installing Eigen on Ubuntu/Debian:
```bash
sudo apt-get install libeigen3-dev
```

### Installing Eigen on macOS (with Homebrew):
```bash
brew install eigen
```

## Building

1. Clone the repository:
```bash
git clone <repository-url>
cd StochasticLA
```

2. Build using the provided script:
```bash
./build.sh
```

Or manually:
```bash
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
ctest --output-on-failure
```

## Usage

```cpp
#include "RandomizedLinearAlgebra.hpp"
using namespace randla;

// Use double precision (default)
using RLA = RandomizedLinearAlgebraD;

// Create matrices
RLA::Matrix A(3, 3);
A << 1, 2, 3,
     4, 5, 6,
     7, 8, 9;

// Generate random matrix
RLA::Matrix R = RLA::randomMatrix(3, 3, 42); // seed = 42

// Basic operations
RLA::Matrix C = RLA::multiply(A, R);
double norm = RLA::frobeniusNorm(A);

// Use different precisions
using RLAF = RandomizedLinearAlgebraF;  // float
using RLALD = RandomizedLinearAlgebraLD; // long double
```

## Project Structure

```
├── include/
│   ├── RandomizedLinearAlgebra.hpp      # Main header file
│   └── RandomizedLinearAlgebra_impl.hpp # Implementation
├── src/
│   └── CMakeLists.txt        # Library build configuration
├── tests/
│   ├── test_basic.cpp        # Basic functionality tests
│   └── CMakeLists.txt        # Test build configuration
├── build.sh                  # Build script
└── CMakeLists.txt           # Main CMake configuration
```

## Current API

### Basic Operations
- `multiply(A, B)`: Matrix multiplication wrapper
- `frobeniusNorm(A)`: Frobenius norm computation
- `randomMatrix(rows, cols, seed)`: Random matrix generation

### Planned Features
- **Randomized SVD**: Fast approximate singular value decomposition
- **Matrix Sketching**: Random sampling techniques for dimension reduction
- **Low-Rank Approximation**: Efficient algorithms for matrix approximation
- **Random Projection**: Johnson-Lindenstrauss lemma implementations

## Testing

Run the test suite:
```bash
cd build
ctest --output-on-failure
```

Or run tests directly:
```bash
./tests/test_basic
```

## Contributing

This project is part of an academic assignment on randomized linear algebra. The goal is to implement algorithms from the literature mentioned in the references.

## References

- Halko, N., Martinsson, P. G., & Tropp, J. A. (2011). Finding structure with randomness: Probabilistic algorithms for constructing approximate matrix decompositions. SIAM review, 53(2), 217-288.
- Mahoney, M. W. (2011). Randomized algorithms for matrices and data. Foundations and Trends® in Machine Learning, 3(2), 123-224.
- R. Murray et al. (2023) Randomized Numerical Linear Algebra A Perspective on the Field With an Eye to Software, https://arxiv.org/abs/2302.11474

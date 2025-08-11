#include <iostream>
#include <iomanip>
#include <Eigen/SVD>
#include <randla/randla.hpp>

using namespace randla::algorithms;
using namespace randla::utils;
using RLA     = randla::RandomizedLinearAlgebraD;
using TestMat = randla::MatrixGeneratorsD;


void testAlgorithms() {
    std::cout << std::fixed << std::setprecision(6);

    const int m = 1000, n = 800;        // matrix shape
    const int l = 10;                   // target rank / samples
    const int q = 2;                    // power/subspace iterations
    const int seed = 123;               // base seed
    const double tol_adaptive = 5e-1;   // tolerance for adaptive range finder
    const int r_adaptive = 10;          // probes for adaptive posterior test

    auto A = TestMat::matrixWithExponentialDecay(m, n, 0.3, seed);

    std::cout << "\n--- Algorithm comparison (exp decay spectrum) ---\n";
    std::cout << "Shape: " << A.rows() << " x " << A.cols() << "\n";
    std::cout << "Norm(A) = " << A.norm() << "\n";

    auto Q_range    = RLA::randomizedRangeFinder(A, l, seed + 1);
    auto Q_power    = RLA::randomizedPowerIteration(A, l, q, seed + 2);
    auto Q_subspace = RLA::randomizedSubspaceIteration(A, l, q, seed + 3);
    auto Q_adaptive = RLA::adaptiveRangeFinder(A, tol_adaptive, r_adaptive, seed + 4);

    std::cout << "[RRF] cols=" << Q_range.cols() << " err=" << RLA::realError(A, Q_range) << "\n";
    std::cout << "[RPI q=" << q << "] cols=" << Q_power.cols() << " err=" << RLA::realError(A, Q_power) << "\n";
    std::cout << "[RSI q=" << q << "] cols=" << Q_subspace.cols() << " err=" << RLA::realError(A, Q_subspace) << "\n";
    std::cout << "[AdaptiveRF tol=" << tol_adaptive << "] cols=" << Q_adaptive.cols() << " err=" << RLA::realError(A, Q_adaptive) << "\n";

    // Posterior error estimates on one basis (power iteration)
    for (int r_probe : {1, 5, 10}) {
        std::cout << "\n[Posterior r=" << r_probe << "] est="
                  << RLA::posteriorErrorEstimation(A, Q_power, r_probe, seed) << "\n";
    }
}

int main() {
    try {
        testAlgorithms();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}

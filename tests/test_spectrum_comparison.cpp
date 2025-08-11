#include <iostream>
#include <iomanip>
#include <randla/randla.hpp>

using namespace randla::algorithms;
using namespace randla::utils;

using RLA     = randla::RandomizedLinearAlgebraD;
using TestMat = randla::MatrixGeneratorsD;

int main() {
    std::cout << std::fixed << std::setprecision(6);

    const int m = 1000;
    const int n = 800;
    const int l = 50;      // samples / target columns
    const int rank = 100;  // effective rank for constructed spectra
    const int seed = 42;
    const int q = 3;       // iterations for power/subspace

    // --- Exponential decay spectrum ---
    {
        auto A = TestMat::matrixWithExponentialDecay(m, n, 0.1, rank, seed);
        std::cout << "\n--- Spectrum test: exponential decay ---\n";
        std::cout << "Shape: " << A.rows() << " x " << A.cols() << "\n";
        std::cout << "Norm(A) = " << A.norm() << "\n";

        auto Q_range    = RLA::randomizedRangeFinder(A, l, seed + 10);
        auto Q_power    = RLA::randomizedPowerIteration(A, l, q, seed + 11);
        auto Q_subspace = RLA::randomizedSubspaceIteration(A, l, q, seed + 12);

        std::cout << "[Range] cols=" << Q_range.cols() << " err=" << RLA::realError(A, Q_range) << "\n";
        std::cout << "[Power q=" << q << "] cols=" << Q_power.cols() << " err=" << RLA::realError(A, Q_power) << "\n";
        std::cout << "[Subspace q=" << q << "] cols=" << Q_subspace.cols() << " err=" << RLA::realError(A, Q_subspace) << "\n";
    }

    // --- Flat(ter) spectrum (slow decay) ---
    {
        auto A = TestMat::matrixWithExponentialDecay(m, n, 0.01, rank, seed);
        std::cout << "\n--- Spectrum test: near-flat (rank " << rank << ") ---\n";
        std::cout << "Shape: " << A.rows() << " x " << A.cols() << "\n";
        std::cout << "Norm(A) = " << A.norm() << "\n";

        auto Q_range    = RLA::randomizedRangeFinder(A, l, seed + 13);
        auto Q_power    = RLA::randomizedPowerIteration(A, l, q, seed + 14);
        auto Q_subspace = RLA::randomizedSubspaceIteration(A, l, q, seed + 15);

        std::cout << "[Range] cols=" << Q_range.cols() << " err=" << RLA::realError(A, Q_range) << "\n";
        std::cout << "[Power q=" << q << "] cols=" << Q_power.cols() << " err=" << RLA::realError(A, Q_power) << "\n";
    std::cout << "[Subspace q=" << q << "] cols=" << Q_subspace.cols() << " err=" << RLA::realError(A, Q_subspace) << "\n";
    }

    std::cout << "\nExpected qualitative behavior: power/subspace iterations perform better on flatter spectra." << std::endl;
    return 0;
}

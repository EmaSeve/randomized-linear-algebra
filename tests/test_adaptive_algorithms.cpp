// tests/test_adaptive_algorithms.cpp (or whichever name you prefer)
#include <iostream>
#include <iomanip>
#include <randla/randla.hpp>
#include "load_matrix_market.hpp"

using namespace randla::algorithms;
using namespace randla::utils;

using RLA       = randla::RandomizedRangeFinderD;
using TestMat   = randla::MatrixGeneratorsD;
using Err       = randla::algorithms::ErrorEstimators<double>;


int main() {
    using FloatType = double;
    using Matrix = Eigen::MatrixXd;

    const int rows = 200;
    const int cols = 50;
    const int rank = 5;
    const double tol = 1e-1;   // absolute tolerance (consistent with rest)
    const int r = 10;          // probes for adaptive algorithms
    const int seed = 42;

    // SRFT parameters
    const int l_srft = 12;     // moderate oversampling for bulk SRFT
    const int l0_fp  = 32;     // initial samples for fixed-precision SRFT

    using GM = randla::utils::MatrixGenerators<FloatType>;
    std::cout << std::fixed << std::setprecision(6);

    auto run_test = [&](const std::string& name, const Matrix& A) {
        std::cout << "\n--- " << name << " ---\n";
        std::cout << "Shape: " << A.rows() << " x " << A.cols() << "\n";
        std::cout << "Norm(A) = " << A.norm() << "\n";

        // ====== Adaptive Range Finder (real error) ======
        Matrix Q1 = RLA::adaptiveRangeFinder(A, tol, r, seed);
        std::cout << "[ARF] cols=" << Q1.cols()
                  << " err=" << Err::realError(A, Q1) << "\n";

        // ====== Adaptive Power Iteration (real error) ======
        Matrix Q2 = RLA::adaptivePowerIteration(A, tol, r, 2, seed);
        std::cout << "[API] cols=" << Q2.cols()
                  << " err=" << Err::realError(A, Q2) << "\n";

        // ====== SRFT fixed-precision (Alg. 4.5) ======
        {
            auto Qc_fp = RLA::adaptiveFastRandomizedRangeFinder(A, tol, l0_fp, seed);
            double err_fp = Err::realError(A, Qc_fp); // real error (overload handles complex types)
            std::cout << "[SRFT] l=" << Qc_fp.cols()
                      << " err=" << err_fp << "\n";
        }
    };

    // Test 1: exact rank
    {
        GM::Vector sv = GM::Vector::Zero(std::min(rows, cols));
        for (int i = 0; i < rank; ++i) sv(i) = 1.0;
        Matrix A_exact = GM::matrixWithSingularValues(rows, cols, sv, seed);
        run_test("Exact rank " + std::to_string(rank), A_exact);
    }

    // Test 2: exponential decay
    {
        double decay_rate = 0.2;
        Matrix A_decay = GM::matrixWithExponentialDecay(rows, cols, decay_rate, rank, seed);
        run_test("Exponential decay (rate=" + std::to_string(decay_rate) + ")", A_decay);
    }

    // Test 3: low-rank + noise
    {
        double noise_level = 0.001;
        Matrix A_noise = GM::lowRankPlusNoise(rows, cols, rank, noise_level, seed);
        run_test("Low-rank + noise (sigma=" + std::to_string(noise_level) + ")", A_noise);
    }

    // Test 4: real matrix (same style as other tests)
    try {
        const std::string path = "data/well1850.mtx";
        Matrix A_real = loadMatrixMarket(path);
        run_test("Real matrix (" + path + ")", A_real);
    } catch (const std::exception& e) {
        std::cerr << "Skipping real matrix test: " << e.what() << "\n";
    }


    std::cout << "\n=== Tests completed ===\n";
    return 0;
}

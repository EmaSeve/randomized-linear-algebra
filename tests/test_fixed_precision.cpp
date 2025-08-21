// tests/test_adaptive_algorithms.cpp (or whichever name you prefer)
#include <iostream>
#include <iomanip>
#include <randla/randla.hpp>
#include "load_matrix_market.hpp"

using namespace randla::algorithms;
using namespace randla::utils;

using ARRF      = randla::AdaptiveRandRangeFinderD;
using TestMat   = randla::MatrixGeneratorsD;
using Err       = randla::metrics::ErrorEstimators<double>;

int main() {
    randla::threading::setThreads(1);

    using FloatType = double;
    using Matrix = Eigen::MatrixXd;

    std::cout << "Eigen nbThreads = " << Eigen::nbThreads() << "\n";

    const int rows = 800;
    const int cols = 400;
    const int rank = 20;
    const double tol = 0.1;   // absolute tolerance
    const int r = 10;          // probes for adaptive algorithms
    const int seed = 42;
    const double growth_factor = 2.0; // for adaptive growth

    // SRFT parameters
    const int l_srft = 1;     // moderate oversampling for bulk SRFT

    using GM = randla::utils::MatrixGenerators<FloatType>;
    std::cout << std::fixed << std::setprecision(6);

    auto run_test = [&](const std::string& name, const Matrix& A) {
        std::cout << "\n--- " << name << " ---\n";
        std::cout << "Shape: " << A.rows() << " x " << A.cols() << "\n";
        std::cout << "Norm(A) = " << A.norm() << "\n";
    // Timing helpers
    auto tic = [](){return std::chrono::high_resolution_clock::now();};
    auto ms  = [](auto s, auto e){return std::chrono::duration<double, std::milli>(e-s).count();};

    // ====== Adaptive Range Finder (real error) ======
    auto t0 = tic();
    Matrix Q1 = ARRF::adaptiveRangeFinder(A, tol, r, seed);
    auto t1 = tic();
    std::cout << "[ARF]  cols=" << Q1.cols()
          << " err=" << Err::realError(A, Q1)
          << " time_ms=" << ms(t0,t1) << "\n";

    // ====== Adaptive Power Iteration (real error) ======
    auto t2 = tic();
    Matrix Q2 = ARRF::adaptivePowerIteration(A, tol, r, 2, seed);
    auto t3 = tic();
    std::cout << "[API]  cols=" << Q2.cols()
          << " err=" << Err::realError(A, Q2)
          << " time_ms=" << ms(t2,t3) << "\n";

    // ====== SRFT fixed-precision (Alg. 4.5) ======
    auto t4 = tic();
    auto Qc_fp = ARRF::adaptiveFastRandRangeFinder(A, tol, l_srft, seed, growth_factor);
    auto t5 = tic();
    double err_fp = Err::realError(A, Qc_fp); // real error (overload handles complex types)
    std::cout << "[SRFT] l=" << Qc_fp.cols()
          << " err=" << err_fp
          << " time_ms=" << ms(t4,t5) << "\n";
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

    // Test 3: low-rank
    {
        Matrix A_noise = GM::lowRankPlusNoise(rows, cols, rank, 0, seed);
        run_test("Low-rank (rank=" + std::to_string(rank) + ")", A_noise);
    }

    // Test 4: low-rank + noise
    {
        double noise_level = 0.001;
        Matrix A_noise = GM::lowRankPlusNoise(rows, cols, rank, noise_level, seed);
        run_test("Low-rank + noise (rank=" + std::to_string(rank) +
                 ", sigma=" + std::to_string(noise_level) + ")", A_noise);
    }

    // Test 5: real matrix
    // try {
    //     const std::string path = "data/well1850.mtx";
    //     Matrix A_real = loadMatrixMarket(path);
    //     run_test("Real matrix (" + path + ")", A_real);
    // } catch (const std::exception& e) {
    //     std::cerr << "Skipping real matrix test: " << e.what() << "\n";
    // }


    std::cout << "\n=== Tests completed ===\n";
    return 0;
}

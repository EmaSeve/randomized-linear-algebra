#include <iostream>
#include <iomanip>
#include <randla/randla.hpp>
#include "load_matrix_market.hpp"

using namespace randla::algorithms;
using namespace randla::utils;

using RLA     = randla::RandomizedLinearAlgebraD;
using TestMat = randla::MatrixGeneratorsD;

int main() {
    using FloatType = double;
    using Matrix = Eigen::MatrixXd;

    const int rows = 200;
    const int cols = 50;
    const int rank = 5;
    const double tol = 1e-1;
    const int r = 10;
    const int seed = 42;

    using GM = randla::utils::MatrixGenerators<FloatType>;
    std::cout << std::fixed << std::setprecision(6);

    auto run_test = [&](const std::string& name, const Matrix& A) {
        std::cout << "\n--- " << name << " ---\n";
        std::cout << "Norm(A) = " << A.norm() << "\n";

        Matrix Q1 = RLA::adaptiveRangeFinder(A, tol, r, seed);
        std::cout << "[ARF] cols=" << Q1.cols() 
                  << " err=" << RLA::realError(A, Q1) << "\n";

        Matrix Q2 = RLA::adaptivePowerIteration(A, tol, r, 2, seed);
        std::cout << "[API] cols=" << Q2.cols() 
                  << " err=" << RLA::realError(A, Q2) << "\n";
    };

    // Test 1: rango esatto
    GM::Vector sv = GM::Vector::Zero(std::min(rows, cols));
    for (int i = 0; i < rank; ++i) sv(i) = 1.0;
    Matrix A_exact = GM::matrixWithSingularValues(rows, cols, sv, seed);
    run_test("Exact rank " + std::to_string(rank), A_exact);

    // Test 2: decadimento esponenziale
    double decay_rate = 0.2;
    Matrix A_decay = GM::matrixWithExponentialDecay(rows, cols, decay_rate, rank, seed);
    run_test("Exponential decay (rate=" + std::to_string(decay_rate) + ")", A_decay);

    // Test 3: low-rank + rumore
    double noise_level = 0.001;
    Matrix A_noise = GM::lowRankPlusNoise(rows, cols, rank, noise_level, seed);
    run_test("Low-rank + noise (sigma=" + std::to_string(noise_level) + ")", A_noise);

    // Test 4: real matrix
    try {
        const std::string path = "data/well1850.mtx"; 
        Matrix A_real = loadMatrixMarket(path);

        std::cout << "\n--- Real matrix (" << path << ") ---\n";
        std::cout << "Shape: " << A_real.rows() << "x" << A_real.cols() << "\n";
        std::cout << "Norm(A) = " << A_real.norm() << "\n";

        // Adaptive Range Finder
        Matrix Q1 = RLA::adaptiveRangeFinder(A_real, tol, r, seed);
        std::cout << "[ARF] cols=" << Q1.cols()
                << " err=" << RLA::realError(A_real, Q1) << "\n";

        // Adaptive Power Iteration (q=2)
        Matrix Q2 = RLA::adaptivePowerIteration(A_real, tol, r, 2, seed);
        std::cout << "[API q=2] cols=" << Q2.cols()
                << " err=" << RLA::realError(A_real, Q2) << "\n";
    }
    catch (const std::exception& e) {
        std::cerr << "Skipping real matrix test: " << e.what() << "\n";
    }



    std::cout << "\n=== Tests completed ===\n";
    return 0;
}

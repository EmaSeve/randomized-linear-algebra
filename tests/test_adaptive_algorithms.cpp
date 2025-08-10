#include <iostream>
#include <iomanip>
#include "../include/randla/algorithms/randomized_linear_algebra.hpp"
#include "../include/randla/utils/test_matrices.hpp"

using namespace randla::algorithms;
using namespace randla::utils;

int main() {
    using FloatType = double;
    using Matrix = Eigen::MatrixXd;

    std::cout << "=== Adaptive Algorithms Test (Simplified) ===" << std::endl;
    std::cout << std::fixed << std::setprecision(6);

    // Parameters
    const int rows = 1000;
    const int cols = 800;
    const int rank = 10;          // exact rank
    const double tol = 1;         // broad tolerance
    const int r = 10;             // number of probes

    std::cout << "Parameters: " << rows << "x" << cols
              << ", rank=" << rank
              << ", tol=" << tol << ", r=" << r << std::endl;

    using TM = randla::utils::TestMatrices<FloatType>;

    // Create vector of singular values: first k >0, rest = 0
    TM::Vector sv = TM::Vector::Zero(std::min(rows, cols));
    for (int i = 0; i < rank; ++i) {
        sv(i) = 1.0; // all equal to 1
    }

    // Matrix with exact rank k
    Matrix A = TM::matrixWithSingularValues(rows, cols, sv, /*seed=*/42);

    std::cout << "Norm of A: " << A.norm() << std::endl;

    // 1. Adaptive Range Finder
    std::cout << "\n1. Adaptive Range Finder Test" << std::endl;
    std::cout << "------------------------------" << std::endl;
    try {
        std::cout << "Starting adaptiveRangeFinder..." << std::endl;
        Matrix Q1 = RandomizedLinearAlgebra<FloatType>::adaptiveRangeFinder(A, tol, r);
        double error1 = RandomizedLinearAlgebra<FloatType>::realError(A, Q1);

        std::cout << "Success" << std::endl;
        std::cout << "  Q dimensions: " << Q1.rows() << "x" << Q1.cols() << std::endl;
        std::cout << "  Error: " << error1 << std::endl;
    } catch (const std::exception& e) {
        std::cout << "Error: " << e.what() << std::endl;
    }

    // 2. Adaptive Power Iteration
    std::cout << "\n2. Adaptive Power Iteration Test" << std::endl;
    std::cout << "---------------------------------" << std::endl;
    try {
        std::cout << "Starting adaptivePowerIteration..." << std::endl;
        const double threshold = tol / (10 * std::sqrt(2.0 / M_PI));
        std::cout << "Internal threshold: " << threshold << std::endl;

        Matrix Q2 = RandomizedLinearAlgebra<FloatType>::adaptivePowerIteration(A, tol, r, 2);
        double error2 = RandomizedLinearAlgebra<FloatType>::realError(A, Q2);

        std::cout << "Success" << std::endl;
        std::cout << "  Q dimensions: " << Q2.rows() << "x" << Q2.cols() << std::endl;
        std::cout << "  Actual error: " << error2 << std::endl;
    } catch (const std::exception& e) {
        std::cout << "Error: " << e.what() << std::endl;
    }

    std::cout << "\n=== Tests completed ===" << std::endl;
    return 0;
}

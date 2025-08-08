#include <iostream>
#include <iomanip>
#include <Eigen/SVD>
#include "../include/randla/algorithms/randomized_linear_algebra.hpp"
#include "../include/randla/utils/test_matrices.hpp"

using namespace randla::algorithms;
using namespace randla::utils;
using RLA = RandomizedLinearAlgebraD;
using TestMat = TestMatricesD;

void testAlgorithms() {
    std::cout << "\n=== ALGORITHM TESTS ===" << std::endl;
    
    const int m = 1000, n = 800, l = 10, q = 2;
    const int seed = 123;
    
    auto A = TestMat::matrixWithExponentialDecay(m, n, 0.3, seed);
    auto Q_power = RLA::randomizedPowerIteration(A, l, q);
    auto Q_subspace = RLA::randomizedSubspaceIteration(A, l, q);

    auto Q_range = RLA::randomizedRangeFinder(A, l);
    auto Q_adaptiveRange = RLA::adaptiveRangeFinder(A, 0.5, 10);

    std::cout<< "Dimension of Q_Adaptive range: "<<Q_range.rows()<< " x "<<Q_range.cols()<<std::endl;
    
    std::cout << "Power Iteration error: " << std::fixed << std::setprecision(4) 
              << RLA::realError(A, Q_power) << std::endl;
    std::cout << "Subspace Iteration error: " << std::fixed << std::setprecision(4) 
              << RLA::realError(A, Q_subspace) << std::endl;
    std::cout<< "Range Finder error: "<< std::fixed << std::setprecision(4) 
              << RLA::realError(A, Q_range) << std::endl;
    std::cout<< "Adaptive Range Finder error: "<< std::fixed << std::setprecision(4) 
              << RLA::realError(A, Q_adaptiveRange) << std::endl;
                       
    std::cout << "Posterior error estimate (r=1): " << std::fixed << std::setprecision(4) 
            << RLA::posteriorErrorEstimation(A, Q_power, 1, seed) << std::endl;
    std::cout << "Posterior error estimate (r=5): " << std::fixed << std::setprecision(4) 
              << RLA::posteriorErrorEstimation(A, Q_power, 5, seed) << std::endl;
    std::cout << "Posterior error estimate (r=10): " << std::fixed << std::setprecision(4) 
            << RLA::posteriorErrorEstimation(A, Q_power, 10, seed) << std::endl;
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

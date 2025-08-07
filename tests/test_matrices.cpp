#include <iostream>
#include <iomanip>
#include <Eigen/SVD>
#include "../include/StochasticLA.hpp"
#include "../include/TestMatrices.hpp"

using namespace StochasticLA;
using RLA = RandomizedLinearAlgebraD;
using TestMat = TestMatricesD;

void testAlgorithms() {
    std::cout << "\n=== ALGORITHM TESTS ===" << std::endl;
    
    const int m = 1000, n = 800, l = 10, q = 2;
    const int seed = 123;
    
    auto A = TestMat::matrixWithExponentialDecay(m, n, 0.3, seed);
    auto Q_power = RLA::randomizedPowerIteration(A, l, q);
    auto Q_subspace = RLA::randomizedSubspaceIteration(A, l, q);
    
    std::cout << "Power Iteration error: " << std::fixed << std::setprecision(4) 
              << RLA::realError(A, Q_power) << std::endl;
    std::cout << "Subspace Iteration error: " << std::fixed << std::setprecision(4) 
              << RLA::realError(A, Q_subspace) << std::endl;
    std::cout << "Posterior error estimate: " << std::fixed << std::setprecision(4) 
              << RLA::posteriorErrorEstimation(A, Q_power, 5, seed) << std::endl;
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

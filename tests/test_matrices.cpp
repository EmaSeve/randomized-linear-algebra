#include <iostream>
#include <iomanip>
#include <Eigen/SVD>
#include "../include/StochasticLA.hpp"
#include "../include/TestMatrices.hpp"

using namespace StochasticLA;
using RLA = RandomizedLinearAlgebraD;
using TestMat = TestMatricesD;

void printSingularValues(const RLA::Matrix& A, const std::string& name, int num_to_print = 5) {
    Eigen::JacobiSVD<RLA::Matrix> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
    auto singular_values = svd.singularValues();
    
    std::cout << name << ": ";
    int num_print = std::min(num_to_print, static_cast<int>(singular_values.size()));
    for (int i = 0; i < num_print; ++i) {
        std::cout << std::fixed << std::setprecision(3) << singular_values(i);
        if (i < num_print - 1) std::cout << ", ";
    }
    if (singular_values.size() > num_to_print) {
        std::cout << ", ...";
    }
    std::cout << std::endl;
}

void testMatrixTypes() {
    const int size = 50;
    const int seed = 42;
    
    std::cout << "=== MATRIX TESTS ===" << std::endl;
    
    // Basic matrices
    auto dense = RLA::randomMatrix(size, size, seed);
    printSingularValues(dense, "Dense");
    
    auto sparse = TestMat::randomSparseMatrix(size, size, 0.1, seed);
    printSingularValues(sparse, "Sparse (10%)");
    
    // Decay
    auto exp_fast = TestMat::matrixWithExponentialDecay(size, size, 0.5, seed);
    printSingularValues(exp_fast, "Exp decay (fast)");
    
    auto exp_slow = TestMat::matrixWithExponentialDecay(size, size, 0.1, seed);
    printSingularValues(exp_slow, "Exp decay (slow)");
    
    // Low-rank
    auto lowrank = TestMat::lowRankMatrixWithNoise(size, size, 5, 0.0, seed);
    printSingularValues(lowrank, "Low-rank (r=5)");
    
    auto lowrank_noisy = TestMat::lowRankMatrixWithNoise(size, size, 5, 0.1, seed);
    printSingularValues(lowrank_noisy, "Low-rank + noise");
    
    // Structured
    auto hankel = TestMat::hankelMatrix(size, seed);
    printSingularValues(hankel, "Hankel");
    
    auto tridiag = TestMat::tridiagonalMatrix(size, 2.0, -1.0, seed);
    printSingularValues(tridiag, "Tridiagonal");
}

void testAlgorithms() {
    std::cout << "\n=== ALGORITHM TESTS ===" << std::endl;
    
    const int m = 100, n = 80, l = 10, q = 2;
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
        testMatrixTypes();
        testAlgorithms();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}

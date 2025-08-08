#include <iostream>
#include <iomanip>
#include "../include/RandomizedLinearAlgebra.hpp"
#include "../include/TestMatrices.hpp"

using namespace randla;
using RLA = RandomizedLinearAlgebraD;
using TestMat = TestMatricesD;

int main() {
    std::cout << "Testing algorithms on different singular spectra" << std::endl;
    
    const int m = 300, n = 200, l = 15, rank = 25, seed = 42;
    
    // Matrix with exponential decay
    auto A_decay = TestMat::matrixWithExponentialDecay(m, n, 0.1, rank, seed);
    
    std::cout << "\nEXPONENTIAL DECAY:" << std::endl;
    auto Q1 = RLA::randomizedRangeFinder(A_decay, l);
    auto Q2 = RLA::randomizedPowerIteration(A_decay, l, 3);
    auto Q3 = RLA::randomizedSubspaceIteration(A_decay, l, 3);
    
    double err1 = RLA::realError(A_decay, Q1);
    double err2 = RLA::realError(A_decay, Q2);
    double err3 = RLA::realError(A_decay, Q3);
    
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "4.1: " << err1 << "  4.3: " << err2 << "  4.4: " << err3 << std::endl;
    
    // Matrix with flat spectrum (limited rank)
    auto A_flat = TestMat::matrixWithExponentialDecay(m, n, 0.001, rank, seed);
    
    std::cout << "\nFLAT SPECTRUM (rank 25):" << std::endl;
    Q1 = RLA::randomizedRangeFinder(A_flat, l);
    Q2 = RLA::randomizedPowerIteration(A_flat, l, 3);
    Q3 = RLA::randomizedSubspaceIteration(A_flat, l, 3);
    
    err1 = RLA::realError(A_flat, Q1);
    err2 = RLA::realError(A_flat, Q2);
    err3 = RLA::realError(A_flat, Q3);
    
    std::cout << "4.1: " << err1 << "  4.3: " << err2 << "  4.4: " << err3 << std::endl;
    
    std::cout << "\nExpected: 4.3 and 4.4 should be better on flat spectrum" << std::endl;
    
    return 0;
}

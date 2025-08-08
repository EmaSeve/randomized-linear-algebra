#include <iostream>
#include <iomanip>
#include <Eigen/SVD>
#include "../include/RandomizedLinearAlgebra.hpp"
#include "../include/TestMatrices.hpp"

using namespace randla;
using RLA = RandomizedLinearAlgebraD;
using TestMat = TestMatricesD;

void testAlgorithms(const RLA::Matrix& A, const std::string& name, int l) {
    std::cout << "\n" << name << std::endl;
    
    auto Q1 = RLA::randomizedRangeFinder(A, l);           
    auto Q2 = RLA::randomizedPowerIteration(A, l, 2);     
    auto Q3 = RLA::randomizedSubspaceIteration(A, l, 2);  
    
    double err1 = RLA::realError(A, Q1);
    double err2 = RLA::realError(A, Q2);
    double err3 = RLA::realError(A, Q3);
    
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Alg 4.1: " << err1 << "  ";
    std::cout << "Alg 4.3: " << err2 << "  ";
    std::cout << "Alg 4.4: " << err3 << std::endl;
    
    if (err1 > 1e-10) {
        std::cout << "Improvement vs 4.1: ";
        std::cout << "4.3=" << std::setprecision(1) << err1/err2 << "x  ";
        std::cout << "4.4=" << err1/err3 << "x" << std::endl;
    }
}

int main() {
    std::cout << "Testing algorithms on matrices with different singular spectra" << std::endl;
    std::cout << "Theory: Alg 4.1 works well with decay, poorly with flat spectrum" << std::endl;
    
    const int m = 200, n = 150, l = 50;
    
    // Matrix with exponential decay
    auto A_decay = TestMat::matrixWithExponentialDecay(m, n, 0.1, 123);
    testAlgorithms(A_decay, "Matrix with EXPONENTIAL DECAY", l);
    
    // Matrix with flat spectrum
    RLA::Vector sigma_flat = RLA::Vector::Zero(std::min(m, n));
    for (int i = 0; i < 80; ++i) {
        sigma_flat(i) = 1.0;
    }
    auto A_flat = TestMat::matrixWithSingularValues(m, n, sigma_flat, 123);
    testAlgorithms(A_flat, "Matrix with FLAT SPECTRUM", l);
    
    std::cout << "\nExpected: 4.3 and 4.4 should improve much more on flat spectrum" << std::endl;
    
    return 0;
}

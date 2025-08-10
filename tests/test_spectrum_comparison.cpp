#include <iostream>
#include <iomanip>
#include <randla/randla.hpp>

using namespace randla::algorithms;
using namespace randla::utils;

using RLA     = randla::RandomizedLinearAlgebraD;
using TestMat = randla::TestMatricesD;

int main() {
    std::cout << "Testing algorithms on different singular spectra" << std::endl;
    
    const int m = 1000, n = 800, l = 15, rank = 100, seed = 42;

    // Matrix with exponential decay
    auto A_decay = TestMat::matrixWithExponentialDecay(m, n, 0.1, rank, seed);
    
    std::cout << "\nEXPONENTIAL DECAY:" << std::endl;
    auto Q1 = RLA::randomizedRangeFinder(A_decay, l);
    auto Q2 = RLA::randomizedPowerIteration(A_decay, l, 3);
    auto Q3 = RLA::randomizedSubspaceIteration(A_decay, l, 3);
    
    double err1 = RLA::realError(A_decay, Q1);
    double err2 = RLA::realError(A_decay, Q2);
    double err3 = RLA::realError(A_decay, Q3);

    constexpr std::array<double, 3> tol = {0.1, 0.01};
    constexpr std::array<int, 2> r = {3, 10};
    double error;
    randla::Types<double>::Matrix Q_adaptive;
/* 
    for(const auto tol_ : tol){
        for(const auto r_ : r){
            Q_adaptive = RLA::adaptiveRangeFinder(A_decay, tol_, r_);
            error = RLA::realError(A_decay, Q_adaptive);
            
            std::cout<< "Adaptive Range Finder with tol = "<<tol_<<", r = "<<r_<<" error = "<<error<<std::endl;
        }
    } */
    
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "4.1: " << err1 << "  4.3: " << err2 << "  4.4: " << err3 << std::endl;
    
    // Matrix with flat spectrum (limited rank)
    auto A_flat = TestMat::matrixWithExponentialDecay(m, n, 0.01, rank, seed);
    
    std::cout << "\nFLAT SPECTRUM (rank 100):" << std::endl;
    Q1 = RLA::randomizedRangeFinder(A_flat, l);
    Q2 = RLA::randomizedPowerIteration(A_flat, l, 3);
    Q3 = RLA::randomizedSubspaceIteration(A_flat, l, 3);
    
    err1 = RLA::realError(A_flat, Q1);
    err2 = RLA::realError(A_flat, Q2);
    err3 = RLA::realError(A_flat, Q3);
/* 
     for(const auto tol_ : tol){
        for(const auto r_ : r){
            Q_adaptive = RLA::adaptiveRangeFinder(A_flat, tol_, r_);
            error = RLA::realError(A_flat, Q_adaptive);
            
            std::cout<< "Adaptive Range Finder with tol = "<<tol_<<", r = "<<r_<<" error = "<<error<<std::endl;
        }
    } */
    
    std::cout << "4.1: " << err1 << "  4.3: " << err2 << "  4.4: " << err3 << std::endl;
    
    std::cout << "\nExpected: 4.3 and 4.4 should be better on flat spectrum" << std::endl;
    
    return 0;
}

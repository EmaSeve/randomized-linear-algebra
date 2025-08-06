#pragma once

#include <random>
#include <chrono>

namespace StochasticLA {

template<typename FloatType>
typename RandomizedLinearAlgebra<FloatType>::Matrix 
RandomizedLinearAlgebra<FloatType>::multiply(const Matrix& A, const Matrix& B) {
    return A * B;
}

template<typename FloatType>
typename RandomizedLinearAlgebra<FloatType>::Scalar 
RandomizedLinearAlgebra<FloatType>::frobeniusNorm(const Matrix& A) {
    return A.norm();
}

template<typename FloatType>
typename RandomizedLinearAlgebra<FloatType>::Matrix 
RandomizedLinearAlgebra<FloatType>::randomMatrix(int rows, int cols, int seed) {
    Matrix result(rows, cols);
    
    std::mt19937 gen;
    if (seed >= 0) {
        gen.seed(seed);
    } else {
        gen.seed(std::chrono::steady_clock::now().time_since_epoch().count());
    }

    std::normal_distribution<FloatType> dist(0.0, 1.0);
    
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            result(i, j) = dist(gen);
        }
    }
    
    return result;
}


} // namespace StochasticLA

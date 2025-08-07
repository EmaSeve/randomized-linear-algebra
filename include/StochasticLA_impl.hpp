#pragma once

#include <random>
#include <chrono>

namespace StochasticLA {

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

template<typename FloatType>
typename RandomizedLinearAlgebra<FloatType>::Matrix 
RandomizedLinearAlgebra<FloatType>::randomizedPowerIteration(const Matrix& A, int l, int q) {
    Matrix result(A.rows(), l);
    Matrix Omega = randomMatrix(A.cols(), l);

    Matrix Y = A * Omega;  // First application: A * Ω
    
    for (int i = 0; i < q; ++i) {
        Y = A.transpose() * Y;  // Apply A*
        Y = A * Y;              // Apply A
    }
    
    // construct an m x l matrix W whose columns form an orthonormal basis for the range of Y
    // via the QR factorizatoin Y = Q * R
    Eigen::HouseholderQR<Matrix> qr(Y);
    Matrix Q = qr.householderQ();
    result = Q.leftCols(l); 

    return result;
}

template<typename FloatType>
typename RandomizedLinearAlgebra<FloatType>::Matrix 
RandomizedLinearAlgebra<FloatType>::randomizedSubspaceIteration(const Matrix& A, int l, int q) {
    Matrix Omega = randomMatrix(A.cols(), l);
    
    Matrix Y = A * Omega;
    Eigen::HouseholderQR<Matrix> qr0(Y);
    Matrix Q = qr0.householderQ().leftCols(l);
    
    for (int j = 1; j <= q; ++j) {
        Matrix Y_tilde = A.transpose() * Q;
        Eigen::HouseholderQR<Matrix> qr_tilde(Y_tilde);
        Matrix Q_tilde = qr_tilde.householderQ().leftCols(l);
        
        Y = A * Q_tilde;
        Eigen::HouseholderQR<Matrix> qr_j(Y);
        Q = qr_j.householderQ().leftCols(l);
    }
    
    return Q;
}


} // namespace StochasticLA

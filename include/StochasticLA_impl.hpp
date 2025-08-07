#pragma once

#include <random>
#include <chrono>
#include <cmath>
#include <Eigen/QR>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

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
typename RandomizedLinearAlgebra<FloatType>::Vector 
RandomizedLinearAlgebra<FloatType>::randomGaussianVector(int size, int seed) {
    Vector result(size);
    
    std::mt19937 gen;
    if (seed >= 0) {
        gen.seed(seed);
    } else {
        gen.seed(std::chrono::steady_clock::now().time_since_epoch().count());
    }

    std::normal_distribution<FloatType> dist(0.0, 1.0);
    
    for (int i = 0; i < size; ++i) {
        result(i) = dist(gen);
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
    Matrix Q = Matrix(qr.householderQ());
    result = Q.leftCols(l); 

    return result;
}

template<typename FloatType>
typename RandomizedLinearAlgebra<FloatType>::Matrix 
RandomizedLinearAlgebra<FloatType>::randomizedSubspaceIteration(const Matrix& A, int l, int q) {
    Matrix Omega = randomMatrix(A.cols(), l);
    
    Matrix Y = A * Omega;
    Eigen::HouseholderQR<Matrix> qr0(Y);
    Matrix Q = Matrix(qr0.householderQ()).leftCols(l);
    
    for (int j = 1; j <= q; ++j) {
        Matrix Y_tilde = A.transpose() * Q;
        Eigen::HouseholderQR<Matrix> qr_tilde(Y_tilde);
        Matrix Q_tilde = Matrix(qr_tilde.householderQ()).leftCols(l);
        
        Y = A * Q_tilde;
        Eigen::HouseholderQR<Matrix> qr_j(Y);
        Q = Matrix(qr_j.householderQ()).leftCols(l);
    }
    
    return Q;
}

template<typename FloatType>
typename RandomizedLinearAlgebra<FloatType>::Scalar 
RandomizedLinearAlgebra<FloatType>::posteriorErrorEstimation(const Matrix& A, const Matrix& Q, int r, int seed) {
    // Equation (4.3): ||(I - QQ*)A|| ≤ 10 * sqrt(2/π) * max_{i=1,...,r} ||(I - QQ*)Aω^(i)||
    
    const FloatType coeff = 10.0 * std::sqrt(2.0 / M_PI);
    FloatType max_norm = 0.0;
    
    std::mt19937 gen;
    if (seed >= 0) {
        gen.seed(seed);
    } else {
        gen.seed(std::chrono::steady_clock::now().time_since_epoch().count());
    }
    
    for (int i = 0; i < r; ++i) {
        Vector omega = randomGaussianVector(A.cols(), gen());
        
        Vector A_omega = A * omega;
        
        Vector QQt_A_omega = Q * (Q.transpose() * A_omega);
        
        Vector residual = A_omega - QQt_A_omega;
        
        FloatType norm = residual.norm();
        
        if (norm > max_norm) {
            max_norm = norm;
        }
    }
    
    return coeff * max_norm;
}

template<typename FloatType>
typename RandomizedLinearAlgebra<FloatType>::Scalar 
RandomizedLinearAlgebra<FloatType>::realError(const Matrix& A, const Matrix& Q) {
    // Compute the real error: ||A - QQ*A|| = ||(I - QQ*)A||
    
    Matrix QQt_A = Q * (Q.transpose() * A);  
    Matrix error_matrix = A - QQt_A;
    return error_matrix.norm();
}

} // namespace StochasticLA

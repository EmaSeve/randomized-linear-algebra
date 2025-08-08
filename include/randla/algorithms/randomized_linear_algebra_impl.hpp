#pragma once

#include <random>
#include <chrono>
#include <cmath>
#include <Eigen/QR>
#include <Eigen/SVD> 
#include <stdexcept>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace randla::algorithms {

template<typename FloatType>
typename RandomizedLinearAlgebra<FloatType>::Matrix 
RandomizedLinearAlgebra<FloatType>::randomGaussianMatrix(int rows, int cols, int seed) {
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

/**
 * 
 */
template<typename FloatType>
typename RandomizedLinearAlgebra<FloatType>::Matrix 
RandomizedLinearAlgebra<FloatType>::randomizedRangeFinder(const Matrix & A, int l){
    // step 1.
    Matrix omega = randomGaussianMatrix(A.cols(), l);
    // step 2.
    Matrix Y = A * omega;
    // step 3.
    Eigen::HouseholderQR<Matrix> qr(Y);
    Matrix thinQ = Matrix::Identity(Y.rows(), l);
    thinQ = qr.householderQ() * thinQ;

    return thinQ;
}

template<typename FloatType>
typename RandomizedLinearAlgebra<FloatType>::Matrix
RandomizedLinearAlgebra<FloatType>::adaptiveRangeFinder(const Matrix & A, double tol, int r){
    
    const size_t rows = A.rows();
    const size_t cols = A.cols();

    // draw 'r' standard gaussian vector: Matrix omega
    Matrix omega = randomGaussianMatrix(cols, r);
    // compute the vector y_i: Matrix Y
    Matrix Y(rows, r);
    Y = A * omega;

    int iteration = -1;

    // start with empty Q
    Matrix Q(rows, 0);

    const double threshold = tol / (10 * std::sqrt(2.0 / M_PI));
    size_t index;

    while(Y.colwise().norm().maxCoeff() >  threshold){
        iteration++;
        index = iteration % r;

        Vector y_i = (Matrix::Identity(rows, rows) - Q * Q.transpose()) * Y.col(index);

        // normalize and compute the new column q_i
        const double norm = y_i.norm();
        if(norm > 0) y_i.normalize();

        // add new column to Q
        Q.conservativeResize(Eigen::NoChange, Q.cols() + 1);
        Q.col(Q.cols() - 1) = y_i;
        const auto q_i = Q.col(Q.cols() - 1); 

        // draw standard gaussian vector w_i 
        Vector w_i = randomGaussianVector(cols);
        
        // replace the vector y_j with y_j+r 
        Y.col(index) = (Matrix::Identity(rows, rows) - Q * Q.transpose()) * (A * w_i);

        // update all the other vector except the new one
        for(size_t j = 0; j < r; j++){
            if(j == index) continue;
            // overwrite y_j = y_j - q_it * <q_it, y_j>
            Y.col(j) -= q_i * q_i.dot(Y.col(j));  
        }
    }

    return Q;
}

template<typename FloatType>
typename RandomizedLinearAlgebra<FloatType>::Matrix 
RandomizedLinearAlgebra<FloatType>::randomizedPowerIteration(const Matrix& A, int l, int q) {
    Matrix Omega = randomGaussianMatrix(A.cols(), l);

    Matrix Y = A * Omega;  // First application: A * Ω
    
    for (int i = 0; i < q; ++i) {
        Y = A.transpose() * Y;  // Apply A*
        Y = A * Y;              // Apply A
    }
    
    Eigen::HouseholderQR<Matrix> qr(Y);
    Matrix thinQ = Matrix::Identity(Y.rows(), l);
    thinQ = qr.householderQ() * thinQ;

    return thinQ;
}

template<typename FloatType>
typename RandomizedLinearAlgebra<FloatType>::Matrix 
RandomizedLinearAlgebra<FloatType>::randomizedSubspaceIteration(const Matrix& A, int l, int q) {
    Matrix Omega = randomGaussianMatrix(A.cols(), l);
    
    Matrix Y = A * Omega;
    Eigen::HouseholderQR<Matrix> qr0(Y);
    Matrix thinQ = Matrix::Identity(Y.rows(), l);
    thinQ = qr0.householderQ() * thinQ;
    
    for (int j = 1; j <= q; ++j) {
        Matrix Y_tilde = A.transpose() * thinQ;
        Eigen::HouseholderQR<Matrix> qr_tilde(Y_tilde);
        Matrix thinQ_tilde = Matrix::Identity(Y_tilde.rows(), l);
        thinQ_tilde = qr_tilde.householderQ() * thinQ_tilde;
        
        Y = A * thinQ_tilde;
        Eigen::HouseholderQR<Matrix> qr_j(Y);
        Matrix fresh_identity = Matrix::Identity(Y.rows(), l);
        thinQ = qr_j.householderQ() * fresh_identity;
    }
    
    return thinQ;
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
        std::normal_distribution<FloatType> dist(0.0, 1.0);
        Vector omega(A.cols());
        for (int j = 0; j < A.cols(); ++j) {
            omega(j) = dist(gen);
        }
        
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



// Stage B:

template<typename FloatType>
typename RandomizedLinearAlgebra<FloatType>::DirectSVDResult 
RandomizedLinearAlgebra<FloatType>::directSVD(const Matrix & A, const Matrix & Q, double tol){

    double error = realError(A, Q);
    if(error > tol) throw std::runtime_error("Error, directSVD: ||A - QQ*A|| > tol"); 

    Matrix B = Q.transpose() * A;

    // compute SVD on B
    Eigen::JacobiSVD<Matrix> svd(B, Eigen::ComputeThinU | Eigen::ComputeThinV);

    Matrix U_tilde = svd.matrixU();        
    Vector S = svd.singularValues();        
    Matrix V = svd.matrixV();          

    // Step 3: 
    Matrix U = Q * U_tilde;

    return DirectSVDResult{std::move(U), std::move(S), std::move(V)};
}




} // namespace randla::algorithms

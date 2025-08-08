#pragma once

#include <random>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <string>
#include <Eigen/QR>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace randla {

template<typename FloatType>
typename TestMatrices<FloatType>::Vector 
TestMatrices<FloatType>::randomGaussianVector(int size, int seed) {
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
typename TestMatrices<FloatType>::Matrix 
TestMatrices<FloatType>::randomSparseMatrix(int rows, int cols, Scalar density, int seed) {
    if (density <= 0.0 || density > 1.0) {
        throw std::invalid_argument("Density must be in range (0, 1]");
    }
    
    Matrix result = Matrix::Zero(rows, cols);
    
    std::mt19937 gen;
    if (seed >= 0) {
        gen.seed(seed);
    } else {
        gen.seed(std::chrono::steady_clock::now().time_since_epoch().count());
    }
    
    std::uniform_real_distribution<FloatType> uniform(0.0, 1.0);
    std::normal_distribution<FloatType> normal(0.0, 1.0);
    
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            if (uniform(gen) < density) {
                result(i, j) = normal(gen);
            }
        }
    }
    
    return result;
}

template<typename FloatType>
typename TestMatrices<FloatType>::Matrix 
TestMatrices<FloatType>::matrixWithExponentialDecay(int rows, int cols, Scalar decay_rate, int seed) {
    if (decay_rate < 0.0) {
        throw std::invalid_argument("Decay rate must be non-negative");
    }
    
    int min_dim = std::min(rows, cols);
    
    // Create random orthogonal matrices U and V
    Matrix U_full = RandomizedLinearAlgebra<FloatType>::randomGaussianMatrix(rows, rows, seed);
    Matrix V_full = RandomizedLinearAlgebra<FloatType>::randomGaussianMatrix(cols, cols, seed + 1);
    
    Eigen::HouseholderQR<Matrix> qr_u(U_full);
    Matrix U = qr_u.householderQ();
    
    Eigen::HouseholderQR<Matrix> qr_v(V_full);
    Matrix V = qr_v.householderQ();
    
    // Create diagonal matrix with exponentially decaying singular values
    Vector sigma = Vector::Zero(min_dim);
    for (int i = 0; i < min_dim; ++i) {
        sigma(i) = std::exp(-decay_rate * i);
    }
    
    // Construct A = U * Sigma * V^T
    Matrix S = Matrix::Zero(rows, cols);
    for (int i = 0; i < min_dim; ++i) {
        S(i, i) = sigma(i);
    }
    
    return U * S * V.transpose();
}

template<typename FloatType>
typename TestMatrices<FloatType>::Matrix 
TestMatrices<FloatType>::lowRankMatrixWithNoise(int rows, int cols, int rank, Scalar noise_level, int seed) {
    if (rank <= 0 || rank > std::min(rows, cols)) {
        throw std::invalid_argument("Rank must be positive and not exceed min(rows, cols)");
    }
    if (noise_level < 0.0) {
        throw std::invalid_argument("Noise level must be non-negative");
    }
    
    // Generate low-rank matrix
    Matrix U = RandomizedLinearAlgebra<FloatType>::randomGaussianMatrix(rows, rank, seed);
    Matrix V = RandomizedLinearAlgebra<FloatType>::randomGaussianMatrix(cols, rank, seed + 1);
    Matrix A_lowrank = U * V.transpose();
    
    // Add noise if requested
    if (noise_level > 0.0) {
        Matrix noise = RandomizedLinearAlgebra<FloatType>::randomGaussianMatrix(rows, cols, seed + 2);
        return A_lowrank + noise_level * noise;
    }
    
    return A_lowrank;
}

template<typename FloatType>
typename TestMatrices<FloatType>::Matrix 
TestMatrices<FloatType>::matrixWithSingularValues(int rows, int cols, const Vector& singular_values, int seed) {
    int min_dim = std::min(rows, cols);
    if (singular_values.size() != min_dim) {
        throw std::invalid_argument("Size of singular_values must equal min(rows, cols)");
    }
    
    // Check that singular values are non-negative and in descending order
    for (int i = 0; i < singular_values.size(); ++i) {
        if (singular_values(i) < 0.0) {
            throw std::invalid_argument("Singular values must be non-negative");
        }
        if (i > 0 && singular_values(i) > singular_values(i-1)) {
            throw std::invalid_argument("Singular values must be in non-increasing order");
        }
    }
    
    // Create random orthogonal matrices U and V
    Matrix U_full = RandomizedLinearAlgebra<FloatType>::randomGaussianMatrix(rows, rows, seed);
    Matrix V_full = RandomizedLinearAlgebra<FloatType>::randomGaussianMatrix(cols, cols, seed + 1);
    
    Eigen::HouseholderQR<Matrix> qr_u(U_full);
    Matrix U = qr_u.householderQ();
    
    Eigen::HouseholderQR<Matrix> qr_v(V_full);
    Matrix V = qr_v.householderQ();
    
    // Construct A = U * Sigma * V^T
    Matrix S = Matrix::Zero(rows, cols);
    for (int i = 0; i < min_dim; ++i) {
        S(i, i) = singular_values(i);
    }
    
    return U * S * V.transpose();
}

template<typename FloatType>
typename TestMatrices<FloatType>::Matrix 
TestMatrices<FloatType>::hankelMatrix(int size, int seed) {
    if (size <= 0) {
        throw std::invalid_argument("Matrix size must be positive");
    }
    
    Vector coeffs = randomGaussianVector(2 * size - 1, seed);
    Matrix result(size, size);
    
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            result(i, j) = coeffs(i + j);
        }
    }
    
    return result;
}

template<typename FloatType>
typename TestMatrices<FloatType>::Matrix 
TestMatrices<FloatType>::toeplitzMatrix(int rows, int cols, int seed) {
    if (rows <= 0 || cols <= 0) {
        throw std::invalid_argument("Matrix dimensions must be positive");
    }
    
    Vector coeffs = randomGaussianVector(rows + cols - 1, seed);
    Matrix result(rows, cols);
    
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            int idx = (rows - 1 - i) + j;
            result(i, j) = coeffs(idx);
        }
    }
    
    return result;
}

template<typename FloatType>
typename TestMatrices<FloatType>::Matrix 
TestMatrices<FloatType>::tridiagonalMatrix(int size, Scalar main_diag_value, Scalar off_diag_value, int seed) {
    if (size <= 0) {
        throw std::invalid_argument("Matrix size must be positive");
    }
    
    Matrix result = Matrix::Zero(size, size);
    
    // Set main diagonal
    for (int i = 0; i < size; ++i) {
        result(i, i) = main_diag_value;
    }
    
    // Set off-diagonals
    for (int i = 0; i < size - 1; ++i) {
        result(i, i + 1) = off_diag_value;     // Super-diagonal
        result(i + 1, i) = off_diag_value;     // Sub-diagonal
    }
    
    // Add small random perturbations if seed is provided
    if (seed >= 0) {
        std::mt19937 gen(seed);
        std::normal_distribution<FloatType> dist(0.0, 0.01); // Small perturbations
        
        for (int i = 0; i < size; ++i) {
            for (int j = 0; j < size; ++j) {
                if (std::abs(i - j) <= 1) { // Only perturb the tridiagonal part
                    result(i, j) += dist(gen);
                }
            }
        }
    }
    
    return result;
}

} // namespace randla

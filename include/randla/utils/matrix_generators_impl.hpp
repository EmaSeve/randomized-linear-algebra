#pragma once

#include <random>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <string>
#include <Eigen/QR>
#include "../algorithms/randomized_linear_algebra.hpp"

namespace randla::utils {

template<typename FloatType>
typename MatrixGenerators<FloatType>::Matrix
MatrixGenerators<FloatType>::randomSparseMatrix(int rows, int cols, Scalar density, int seed) {
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
typename MatrixGenerators<FloatType>::Matrix
MatrixGenerators<FloatType>::matrixWithExponentialDecay(int rows, int cols, Scalar decay_rate, int rank, int seed) {
    if (decay_rate < 0.0) {
        throw std::invalid_argument("Decay rate must be non-negative");
    }
    int min_dim = std::min(rows, cols);
    int effective_rank = (rank <= 0) ? min_dim : std::min(rank, min_dim);
    Matrix U_full = randla::algorithms::RandomizedLinearAlgebra<FloatType>::randomGaussianMatrix(rows, rows, seed);
    Matrix V_full = randla::algorithms::RandomizedLinearAlgebra<FloatType>::randomGaussianMatrix(cols, cols, seed + 1);
    Eigen::HouseholderQR<Matrix> qr_u(U_full);
    Matrix U = qr_u.householderQ();
    Eigen::HouseholderQR<Matrix> qr_v(V_full);
    Matrix V = qr_v.householderQ();
    typename MatrixGenerators<FloatType>::Vector sigma(min_dim);
    sigma.setZero();
    for (int i = 0; i < effective_rank; ++i) {
        sigma(i) = std::exp(-decay_rate * i);
    }
    Matrix S = Matrix::Zero(rows, cols);
    for (int i = 0; i < min_dim; ++i) {
        S(i, i) = sigma(i);
    }
    return U * S * V.transpose();
}

template<typename FloatType>
typename MatrixGenerators<FloatType>::Matrix
MatrixGenerators<FloatType>::matrixWithSingularValues(int rows, int cols, const Vector& singular_values, int seed) {
    int min_dim = std::min(rows, cols);
    if (singular_values.size() != min_dim) {
        throw std::invalid_argument("Size of singular_values must equal min(rows, cols)");
    }
    for (int i = 0; i < singular_values.size(); ++i) {
        if (singular_values(i) < 0.0) {
            throw std::invalid_argument("Singular values must be non-negative");
        }
        if (i > 0 && singular_values(i) > singular_values(i-1)) {
            throw std::invalid_argument("Singular values must be in non-increasing order");
        }
    }
    Matrix U_full = randla::algorithms::RandomizedLinearAlgebra<FloatType>::randomGaussianMatrix(rows, rows, seed);
    Matrix V_full = randla::algorithms::RandomizedLinearAlgebra<FloatType>::randomGaussianMatrix(cols, cols, seed + 1);
    Eigen::HouseholderQR<Matrix> qr_u(U_full);
    Matrix U = qr_u.householderQ();
    Eigen::HouseholderQR<Matrix> qr_v(V_full);
    Matrix V = qr_v.householderQ();
    Matrix S = Matrix::Zero(rows, cols);
    for (int i = 0; i < min_dim; ++i) {
        S(i, i) = singular_values(i);
    }
    return U * S * V.transpose();
}

} // namespace randla::utils

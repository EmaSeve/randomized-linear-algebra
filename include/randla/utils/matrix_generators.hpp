#pragma once

#include <Eigen/Dense>
#include <type_traits>
#include <stdexcept>
#include <random>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <string>
#include <Eigen/QR>
#include <randla/random/random_generator.hpp>
#include "../types.hpp"

namespace randla::utils {

/**
 * @brief Utility class for generating synthetic matrices with specific properties.
 *
 * Provides static methods to generate various types of matrices commonly used
 * when experimenting with randomized linear algebra algorithms: sparse random
 * matrices, matrices with controlled singular value decay, and matrices with
 * user-specified singular values.
 *
 * @tparam FloatType Floating point type (e.g., float, double).
 */
template<typename FloatType = double>
class MatrixGenerators : public randla::Types<FloatType> {
    static_assert(std::is_floating_point_v<FloatType>,
                  "FloatType must be a floating point type");

public:
    using typename randla::Types<FloatType>::Scalar;
    using typename randla::Types<FloatType>::Matrix;
    using typename randla::Types<FloatType>::Vector;
    using typename randla::Types<FloatType>::SparseMatrix;

    /**
     * @brief Generate a random sparse matrix with Gaussian entries.
     *
     * Each entry is nonzero with probability 'density' and drawn from N(0,1).
     * 
     * @param rows Number of rows.
     * @param cols Number of columns.
     * @param density Fraction of nonzero entries (in (0,1]).
     * @param seed Random seed (default: -1, uses current time).
     * @return SparseMatrix of shape (rows, cols).
     */
    static SparseMatrix randomSparseMatrix(int rows, int cols, Scalar density, int seed = -1) {
        if (density <= 0.0 || density > 1.0) {
            throw std::invalid_argument("Density must be in range (0, 1]");
        }

        std::mt19937 gen;
        if (seed >= 0) gen.seed(seed);
        else           gen.seed(std::chrono::steady_clock::now().time_since_epoch().count());

        std::uniform_real_distribution<FloatType> uniform(0.0, 1.0);
        std::normal_distribution<FloatType> normal(0.0, 1.0);

        using Triplet = Eigen::Triplet<Scalar, int>;
        std::vector<Triplet> triplets;
        triplets.reserve(static_cast<size_t>(rows * cols * density * 1.1)); 

        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                if (uniform(gen) < density) {
                    triplets.emplace_back(i, j, normal(gen));
                }
            }
        }

        SparseMatrix S(rows, cols);
        S.setFromTriplets(triplets.begin(), triplets.end());
        S.makeCompressed(); // compatto in CSC
        return S;
    }

    /**
     * @brief Generate a dense matrix with exponentially decaying singular values.
     *
     * The singular values decay as exp(-decay_rate * i) for i = 0, ..., rank-1.
     * 
     * @param rows Number of rows.
     * @param cols Number of columns.
     * @param decay_rate Exponential decay rate for singular values.
     * @param rank Effective rank (if <= 0, uses min(rows, cols)).
     * @param seed Random seed (default: -1).
     * @return Matrix with controlled singular value decay.
     */
    static Matrix matrixWithExponentialDecay(int rows, int cols, Scalar decay_rate, int rank, int seed  = -1) {
        if (decay_rate < 0.0) {
            throw std::invalid_argument("Decay rate must be non-negative");
        }
        int min_dim = std::min(rows, cols);
        int effective_rank = (rank <= 0) ? min_dim : std::min(rank, min_dim);
        Matrix U_full = randla::random::RandomGenerator<FloatType>::randomGaussianMatrix(rows, rows, seed);
        Matrix V_full = randla::random::RandomGenerator<FloatType>::randomGaussianMatrix(cols, cols, seed + 1);
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

    /**
     * @brief Generate a matrix with user-specified singular values.
     *
     * Constructs a matrix with prescribed singular values and random orthogonal factors.
     * 
     * @param rows Number of rows.
     * @param cols Number of columns.
     * @param singular_values Vector of singular values (size must be min(rows, cols)).
     * @param seed Random seed (default: -1).
     * @return Matrix with given singular values.
     */
    static Matrix matrixWithSingularValues(int rows, int cols, const Vector& singular_values, int seed  = -1) {
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
        Matrix U_full = randla::random::RandomGenerator<FloatType>::randomGaussianMatrix(rows, rows, seed);
        Matrix V_full = randla::random::RandomGenerator<FloatType>::randomGaussianMatrix(cols, cols, seed + 1);
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

    /**
     * @brief Generate a low-rank matrix plus scaled Gaussian noise.
     *
     * The low-rank part has singular values 1 (up to 'rank'), and noise is scaled to a fraction of the signal norm.
     * 
     * @param rows Number of rows.
     * @param cols Number of columns.
     * @param rank Rank of the signal part.
     * @param noise_level Relative noise level (fraction of signal Frobenius norm).
     * @param seed Random seed (default: -1).
     * @return Matrix of shape (rows, cols).
     */
    static Matrix lowRankPlusNoise(int rows, int cols, int rank, Scalar noise_level, int seed  = -1) {
        if (noise_level < 0.0) {
            throw std::invalid_argument("Noise level must be non-negative");
        }

        int min_dim = std::min(rows, cols);
        Vector sv = Vector::Zero(min_dim);
        for (int i = 0; i < std::min(rank, min_dim); ++i) {
            sv(i) = 1.0; 
        }

        Matrix A_lowrank = matrixWithSingularValues(rows, cols, sv, seed);

        Matrix noise = randla::random::RandomGenerator<FloatType>::randomGaussianMatrix(rows, cols, seed + 123);

        // Scale the noise to be a fraction of the signal norm
        Scalar norm_signal = A_lowrank.norm();
        Scalar norm_noise  = noise.norm();
        if (norm_noise > Scalar(0)) {
            noise *= (noise_level * norm_signal) / norm_noise;
        }

        return A_lowrank + noise;
    }

    /**
     * @brief Generate a random sparse Hermitian (symmetric) matrix.
     *
     * The matrix is real, square, and symmetric, with Gaussian entries and given density.
     * 
     * @param size Matrix size (square).
     * @param density Fraction of nonzero entries (in (0,1]).
     * @param seed Random seed (default: -1).
     * @return Sparse symmetric matrix.
     */
    static SparseMatrix randomHermitianSparseMatrix(int size, Scalar density, int seed = -1) {
        if (density <= 0.0 || density > 1.0) {
            throw std::invalid_argument("Density must be in range (0, 1]");
        }

        std::mt19937 gen(seed >= 0 ? seed : std::chrono::steady_clock::now().time_since_epoch().count());
        std::uniform_real_distribution<FloatType> prob(0.0, 1.0);
        std::normal_distribution<FloatType> dist(0.0, 1.0);

        using Triplet = Eigen::Triplet<Scalar, int>;
        std::vector<Triplet> triplets;
        triplets.reserve(static_cast<size_t>(size * size * density * 1.1)); 

        for (int i = 0; i < size; ++i) {
            triplets.emplace_back(i, i, dist(gen)); // real diagonal
            for (int j = i + 1; j < size; ++j) {
                if (prob(gen) < density) {
                    Scalar value = dist(gen);
                    triplets.emplace_back(i, j, value);
                    triplets.emplace_back(j, i, value); 
                }
            }
        }

        SparseMatrix H(size, size);
        H.setFromTriplets(triplets.begin(), triplets.end());
        H.makeCompressed();
        return H;
    }

    /**
     * @brief Generate a random positive semidefinite (PSD) matrix.
     *
     * Constructs A = B * B^T, where B is random Gaussian.
     * 
     * @param size Matrix size (square).
     * @param rank Rank of B (if < 0 or > size, uses full rank).
     * @param seed Random seed (default: -1).
     * @return Symmetric PSD matrix.
     */
    static Matrix randomPositiveSemidefiniteMatrix(int size, int rank = -1, int seed = -1) {
        if (rank < 0 || rank > size) {
            rank = size; 
        }

        Matrix B = randla::random::RandomGenerator<FloatType>::randomGaussianMatrix(size, rank, seed);
        return B * B.transpose();  
    }

    /**
     * @brief Generate a dense random matrix with i.i.d. standard normal entries.
     *
     * @param rows Number of rows.
     * @param cols Number of columns.
     * @param seed Random seed (default: -1).
     * @return Dense random matrix.
     */
    static Matrix randomDenseMatrix(int rows, int cols, int seed = -1) {
        return randla::random::RandomGenerator<FloatType>::randomGaussianMatrix(rows, cols, seed);
    }

    /**
     * @brief Generate a dense random real symmetric (Hermitian) matrix.
     *
     * The matrix is constructed as 0.5 * (A + A^T) where A is Gaussian.
     * 
     * @param n Matrix size (square).
     * @param seed Random seed (default: -1).
     * @return Symmetric dense matrix.
     */
    static Matrix randomHermitianMatrix(int n, int seed = -1) {
        Matrix A = randla::random::RandomGenerator<FloatType>::randomGaussianMatrix(n, n, seed);
        Matrix H = 0.5 * (A + A.transpose());
        return H;
    }

};

} // namespace randla::utils
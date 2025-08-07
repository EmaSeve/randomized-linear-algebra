#pragma once

#include <Eigen/Dense>
#include <type_traits>

namespace StochasticLA {

/**
 * @brief Utility class for generating test matrices with specific properties
 * 
 * This class provides static methods to generate various types of matrices
 * commonly used for testing randomized linear algebra algorithms, including
 * sparse matrices, matrices with controlled singular value decay, and 
 * structured matrices.
 */
template<typename FloatType = double>
class TestMatrices {
    static_assert(std::is_floating_point_v<FloatType>, 
                  "FloatType must be a floating point type");

public:
    using Scalar = FloatType;
    using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

    /**
     * @brief Generate a random matrix with standard Gaussian entries
     * @param rows Number of rows
     * @param cols Number of columns  
     * @param seed Random seed (if negative, uses current time)
     * @return Random matrix
     */
    static Matrix randomMatrix(int rows, int cols, int seed = -1);
    
    /**
     * @brief Generate a random Gaussian vector
     * @param size Vector size
     * @param seed Random seed (if negative, uses current time)
     * @return Random vector
     */
    static Vector randomGaussianVector(int size, int seed = -1);
    
    /**
     * @brief Generate a sparse matrix with given density
     * @param rows Number of rows
     * @param cols Number of columns
     * @param density Fraction of non-zero entries (0 < density <= 1)
     * @param seed Random seed (if negative, uses current time)
     * @return Sparse matrix
     */
    static Matrix randomSparseMatrix(int rows, int cols, Scalar density, int seed = -1);
    
    /**
     * @brief Generate matrix with exponentially decaying singular values: σ_i = exp(-decay_rate * i)
     * @param rows Number of rows
     * @param cols Number of columns
     * @param decay_rate Exponential decay rate (larger values = faster decay)
     * @param seed Random seed (if negative, uses current time)
     * @return Matrix with exponential singular value decay
     */
    static Matrix matrixWithExponentialDecay(int rows, int cols, Scalar decay_rate, int seed = -1);
    
    /**
     * @brief Generate low-rank matrix with additive noise: A = U * S * V^T + noise_level * N
     * @param rows Number of rows
     * @param cols Number of columns
     * @param rank Target rank of the low-rank component
     * @param noise_level Noise amplitude (0 = no noise)
     * @param seed Random seed (if negative, uses current time)
     * @return Low-rank matrix with noise
     */
    static Matrix lowRankMatrixWithNoise(int rows, int cols, int rank, Scalar noise_level, int seed = -1);
    
    /**
     * @brief Generate matrix with user-specified singular values
     * @param rows Number of rows
     * @param cols Number of columns
     * @param singular_values Vector of desired singular values (size must equal min(rows,cols))
     * @param seed Random seed (if negative, uses current time)
     * @return Matrix with specified singular values
     */
    static Matrix matrixWithSingularValues(int rows, int cols, const Vector& singular_values, int seed = -1);
    
    /**
     * @brief Generate Hankel matrix (constant along anti-diagonals)
     * @param size Matrix size (square matrix)
     * @param seed Random seed (if negative, uses current time)
     * @return Hankel matrix
     */
    static Matrix hankelMatrix(int size, int seed = -1);
    
    /**
     * @brief Generate Toeplitz matrix (constant along diagonals)
     * @param rows Number of rows
     * @param cols Number of columns
     * @param seed Random seed (if negative, uses current time)
     * @return Toeplitz matrix
     */
    static Matrix toeplitzMatrix(int rows, int cols, int seed = -1);
    
    /**
     * @brief Generate a tridiagonal matrix
     * @param size Matrix size (square matrix)
     * @param main_diag_value Value for main diagonal elements
     * @param off_diag_value Value for off-diagonal elements
     * @param seed Random seed (if negative, uses current time for random perturbations)
     * @return Tridiagonal matrix
     */
    static Matrix tridiagonalMatrix(int size, Scalar main_diag_value = 2.0, Scalar off_diag_value = -1.0, int seed = -1);
};

// Type aliases for convenience
using TestMatricesF = TestMatrices<float>;
using TestMatricesD = TestMatrices<double>;
using TestMatricesLD = TestMatrices<long double>;

} // namespace StochasticLA

// Include implementation
#include "TestMatrices_impl.hpp"

#pragma once

#include <Eigen/Dense>
#include <type_traits>
#include "../types.hpp"

namespace randla::utils {

/**
 * @brief Utility class for generating test matrices with specific properties
 * 
 * This class provides static methods to generate various types of matrices
 * commonly used for testing randomized linear algebra algorithms, including
 * sparse matrices, matrices with controlled singular value decay, and 
 * structured matrices.
 */
template<typename FloatType = double>
class TestMatrices : public randla::Types<FloatType> {
    static_assert(std::is_floating_point_v<FloatType>, 
                  "FloatType must be a floating point type");

public:
    // Inherit type aliases from base class
    using typename randla::Types<FloatType>::Scalar;
    using typename randla::Types<FloatType>::Matrix;
    using typename randla::Types<FloatType>::Vector;
    
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
     * @param rank Effective rank of the matrix (if <= 0, uses min(rows,cols))
     * @param seed Random seed (if negative, uses current time)
     * @return Matrix with exponential singular value decay
     */
    static Matrix matrixWithExponentialDecay(int rows, int cols, Scalar decay_rate, int rank = -1, int seed = -1);
    
    /**
     * @brief Generate matrix with user-specified singular values
     * @param rows Number of rows
     * @param cols Number of columns
     * @param singular_values Vector of desired singular values (size must equal min(rows,cols))
     * @param seed Random seed (if negative, uses current time)
     * @return Matrix with specified singular values
     */
    static Matrix matrixWithSingularValues(int rows, int cols, const Vector& singular_values, int seed = -1);
    
};

} // namespace randla::utils

// Include implementation
#include "test_matrices_impl.hpp"

namespace randla::utils {
using TestMatricesD = TestMatrices<double>;
}

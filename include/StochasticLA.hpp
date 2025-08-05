#pragma once

#include <Eigen/Dense>
#include <type_traits>

namespace StochasticLA {

/**
 * @brief Main class for randomized linear algebra operations
 * 
 * This class provides a generic interface for randomized linear algebra
 * algorithms, templated on the floating point type.
 * 
 * @tparam FloatType The floating point type (float, double, long double)
 */
template<typename FloatType = double>
class RandomizedLinearAlgebra {
    static_assert(std::is_floating_point_v<FloatType>, 
                  "FloatType must be a floating point type");

public:
    // Type aliases for convenience
    using Scalar = FloatType;
    using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

    /**
     * @brief Simple wrapper around Eigen's matrix multiplication
     * 
     * This is a test function to verify that Eigen integration works correctly.
     * 
     * @param A First matrix
     * @param B Second matrix
     * @return Matrix Product A * B
     */
    static Matrix multiply(const Matrix& A, const Matrix& B);

    /**
     * @brief Compute the Frobenius norm of a matrix
     * 
     * Another simple wrapper to test the setup.
     * 
     * @param A Input matrix
     * @return Scalar Frobenius norm of A
     */
    static Scalar frobeniusNorm(const Matrix& A);

    /**
     * @brief Generate a random matrix with specified dimensions
     * 
     * @param rows Number of rows
     * @param cols Number of columns
     * @param seed Random seed (optional)
     * @return Matrix Random matrix with entries in [0, 1]
     */
    static Matrix randomMatrix(int rows, int cols, int seed = -1);
};

// Convenient type aliases
using RandomizedLinearAlgebraF = RandomizedLinearAlgebra<float>;
using RandomizedLinearAlgebraD = RandomizedLinearAlgebra<double>;
using RandomizedLinearAlgebraLD = RandomizedLinearAlgebra<long double>;

} // namespace StochasticLA

// Include implementation
#include "StochasticLA_impl.hpp"

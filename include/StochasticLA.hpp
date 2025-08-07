#pragma once

#include <Eigen/Dense>
#include <type_traits>

namespace StochasticLA {

/**
 * @brief Main class implementing randomized linear algebra algorithms
 * 
 * This class provides implementations of state-of-the-art randomized algorithms
 * for low-rank matrix approximation, including randomized power iteration and
 * randomized subspace iteration, along with error estimation methods.
 */
template<typename FloatType = double>
class RandomizedLinearAlgebra {
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
     * @brief Generate a standard Gaussian random vector
     * @param size Vector size
     * @param seed Random seed (if negative, uses current time)
     * @return Random vector
     */
    static Vector randomGaussianVector(int size, int seed = -1);
    
    /**
     * @brief Algorithm 4.3: Randomized power iteration
     * @param A Input matrix
     * @param l Target subspace dimension
     * @param q Number of power iterations
     * @return Orthonormal matrix Q approximating the range of A
     */
    static Matrix randomizedPowerIteration(const Matrix& A, int l, int q);
    
    /**
     * @brief Algorithm 4.4: Randomized subspace iteration
     * @param A Input matrix
     * @param l Target subspace dimension
     * @param q Number of subspace iterations
     * @return Orthonormal matrix Q approximating the range of A
     */
    static Matrix randomizedSubspaceIteration(const Matrix& A, int l, int q);
    
    /**
     * @brief Posterior error estimation using Equation 4.3
     * @param A Input matrix
     * @param Q Computed orthonormal basis
     * @param r Number of test vectors for estimation
     * @param seed Random seed (if negative, uses current time)
     * @return Estimated approximation error
     */
    static Scalar posteriorErrorEstimation(const Matrix& A, const Matrix& Q, int r = 10, int seed = -1);
    
    /**
     * @brief Compute the exact approximation error: ||A - QQ*A||
     * @param A Input matrix
     * @param Q Computed orthonormal basis
     * @return Exact approximation error
     */
    static Scalar realError(const Matrix& A, const Matrix& Q);
};

// Type aliases for convenience
using RandomizedLinearAlgebraF = RandomizedLinearAlgebra<float>;
using RandomizedLinearAlgebraD = RandomizedLinearAlgebra<double>;
using RandomizedLinearAlgebraLD = RandomizedLinearAlgebra<long double>;

} // namespace StochasticLA

// Include implementation
#include "StochasticLA_impl.hpp"

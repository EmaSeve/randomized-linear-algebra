#pragma once

#include <Eigen/Dense>
#include <type_traits>
#include <random>
#include <randla/types.hpp>

namespace randla::algorithms {

/**
 * @brief Main class implementing randomized linear algebra algorithms
 * 
 * This class provides implementations of state-of-the-art randomized algorithms
 * for low-rank matrix approximation, including randomized power iteration and
 * randomized subspace iteration, along with error estimation methods.
 */
template<typename FloatType = double>
class RandomizedLinearAlgebra : public randla::Types<FloatType> {
    static_assert(std::is_floating_point_v<FloatType>, 
                  "FloatType must be a floating point type");

public:
    // Inherit type aliases from base class
    using typename randla::Types<FloatType>::Scalar;
    using typename randla::Types<FloatType>::Matrix;
    using typename randla::Types<FloatType>::Vector;
    using typename randla::Types<FloatType>::Complex;
    using typename randla::Types<FloatType>::CMatrix;
    using typename randla::Types<FloatType>::DirectSVDResult;

    /**
     * @brief Generate a random matrix with standard Gaussian entries
     * @param rows Number of rows
     * @param cols Number of columns
     * @param seed Random seed (if negative, uses current time)
     * @return Random matrix
     */
    static Matrix randomGaussianMatrix(int rows, int cols, int seed = -1);
    /**
     * @brief Generate a random matrix with standard Gaussian entries using an existing generator
     * @param rows Number of rows
     * @param cols Number of columns
     * @param gen  Pseudo-random number engine (state is advanced)
     * @return Random matrix
     */
    static Matrix randomGaussianMatrix(int rows, int cols, std::mt19937 & gen);
    
    /**
     * @brief Generate a standard Gaussian random vector
     * @param size Vector size
     * @param seed Random seed (if negative, uses current time)
     * @return Random vector
     */
    static Vector randomGaussianVector(int size, int seed = -1);
    /**
     * @brief Generate a standard Gaussian random vector using an existing generator
     * @param size Vector size
     * @param gen  Pseudo-random number engine (state is advanced)
     * @return Random vector
     */
    static Vector randomGaussianVector(int size, std::mt19937 & gen);
    

    // Stage A:
    // This section outlines algorithms for constructing a subspace that captures most of the action
    // of a matrix.

    /**
     * @brief   Algorithm 4.1: Randomize Range Finder
     * @param A Input matrix
     * @param l Target subspace dimension
     * @return  Orthonormal matrix Q approximating the range of A
     */
    static Matrix randomizedRangeFinder(const Matrix & A, int l, int seed = -1);

    /**
     * @brief     Algorithm 4.2: Adaptive Range Finder
     * @param A   Input matrix
     * @param tol Error tolerance 
     * @param r   Number of consecutive vectors q_i whose norms
     *            must be below the threshold τ = tol / (10 * sqrt(2/π)) before terminating.
     *            (We stop once we observe r consecutive such vectors)
     * @return    Orthonormal matrix Q approximating the range of A
     * */
    static Matrix adaptiveRangeFinder(const Matrix & A, double tol, int r, int seed = -1);

    /**
     * @brief Algorithm 4.3: Randomized power iteration
     * @param A Input matrix
     * @param l Target subspace dimension
     * @param q Number of power iterations
     * @return Orthonormal matrix Q approximating the range of A
     */
    static Matrix randomizedPowerIteration(const Matrix& A, int l, int q, int seed = -1);
    
    /**
     * @brief Adaptive version of randomized power iteration
     * @param A Input matrix
     * @param tol Error tolerance
     * @param r Number of consecutive vectors q_i whose norms
     * @param q Number of power iterations
     * @return Orthonormal matrix Q approximating the range of A
     */
    static Matrix adaptivePowerIteration(const Matrix& A, double tol, int r, int q = 10, int seed = -1);
    
    /**
     * @brief Algorithm 4.4: Randomized subspace iteration
     * @param A Input matrix
     * @param l Target subspace dimension
     * @param q Number of subspace iterations
     * @return Orthonormal matrix Q approximating the range of A
     */
    static Matrix randomizedSubspaceIteration(const Matrix& A, int l, int q, int seed = -1);
    
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
     * @brief Algorithm 4.5: Fast Randomized Range Finder (SRFT-based, complex)
     * @param A Input real matrix
     * @param l Target subspace dimension
     * @param seed Random seed
     * @return Orthonormal complex matrix Q approximating the range of A
     */
    static CMatrix fastRandomizedRangeFinder(const Matrix& A, int l, int seed = -1);

    /**
     * @brief Fixed-precision variant of the structured randomized range finder (Algorithm 4.5, complex).
     *        Starts with l0 samples, doubles l until the desired tolerance is met.
     *
     * @param A       Real input matrix (m x n).
     * @param tol     Target tolerance (Frobenius norm of residual).
     * @param l0      Initial number of samples (e.g., 32).
     * @param seed    RNG seed (if <0, uses time-based seed).
     * @return CMatrix Qc (m x l_final), complex orthonormal basis for the approximate range of A.
     */
    static CMatrix adaptiveFastRandomizedRangeFinder(const Matrix& A, double tol, int l0, int seed = -1);

    /**
     * @brief Compute the exact approximation error: ||A - QQ*A|| for real Q
     * @param A Input matrix
     * @param Q Computed orthonormal basis (real)
     * @return Exact approximation error
     */
    static Scalar realError(const Matrix& A, const Matrix& Q);

    /**
     * @brief Compute the exact approximation error: ||A - QcQc^*A|| for complex Qc
     * @param A Input matrix
     * @param Qc Computed orthonormal basis (complex)
     * @return Exact approximation error
     */
    static Scalar realError(const Matrix& A, const CMatrix& Qc);


    


    // Stage B: 
    // This section describes methods for approximating standard factorizations of 
    // A using the information in the basis Q.
    
    /**
     * @brief Computes an aproximate factorization A ≈ U S V^*
     * @param A
     * @param Q
     * @return Returns the object containing the three matrices
     */
    static DirectSVDResult directSVD(const Matrix & A, const Matrix & Q, double tol);

};

} // namespace randla::algorithms

// Include implementation
#include "randomized_linear_algebra_impl.hpp"

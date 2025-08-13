#pragma once

#include <Eigen/Dense>
#include <type_traits>
#include <random>
#include <randla/types.hpp>

namespace randla::algorithms {

// Forward declaration of workspace used by fast randomized range finder (defined in impl)
struct FRRWorkspace;

/**
 * @brief Main class implementing randomized linear algebra algorithms
 * 
 * This class provides implementations of state-of-the-art randomized algorithms
 * for low-rank matrix approximation, including randomized power iteration and
 * randomized subspace iteration, along with error estimation methods.
 */
template<typename FloatType = double>
class RandomizedRangeFinder : public randla::Types<FloatType> {
    static_assert(std::is_floating_point_v<FloatType>, 
                  "FloatType must be a floating point type");
public:
    // Inherit type aliases from base class
    using typename randla::Types<FloatType>::Scalar;
    using typename randla::Types<FloatType>::Matrix;
    using typename randla::Types<FloatType>::Vector;
    using typename randla::Types<FloatType>::Complex;
    using typename randla::Types<FloatType>::CMatrix;

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
    template<class MatLike>
    static Matrix randomizedRangeFinder(const MatLike & A, int l, int seed = -1);

    /**
     * @brief     Algorithm 4.2: Adaptive Range Finder
     * @param A   Input matrix
     * @param tol Error tolerance 
     * @param r   Number of consecutive vectors q_i whose norms
     *            must be below the threshold τ = tol / (10 * sqrt(2/π)) before terminating.
     *            (We stop once we observe r consecutive such vectors)
     * @return    Orthonormal matrix Q approximating the range of A
     * */
    template<class MatLike>
    static Matrix adaptiveRangeFinder(const MatLike & A, double tol, int r, int seed = -1);

    /**
     * @brief Algorithm 4.3: Randomized power iteration
     * @param A Input matrix
     * @param l Target subspace dimension
     * @param q Number of power iterations
     * @return Orthonormal matrix Q approximating the range of A
     */
    template<class MatLike>
    static Matrix randomizedPowerIteration(const MatLike& A, int l, int q, int seed = -1);
    
    /**
     * @brief Adaptive version of randomized power iteration
     * @param A Input matrix
     * @param tol Error tolerance
     * @param r Number of consecutive vectors q_i whose norms
     * @param q Number of power iterations
     * @return Orthonormal matrix Q approximating the range of A
     */
    template<class MatLike>
    static Matrix adaptivePowerIteration(const MatLike& A, double tol, int r, int q = 10, int seed = -1);
    
    /**
     * @brief Algorithm 4.4: Randomized subspace iteration
     * @param A Input matrix
     * @param l Target subspace dimension
     * @param q Number of subspace iterations
     * @return Orthonormal matrix Q approximating the range of A
     */
    template<class MatLike>
    static Matrix randomizedSubspaceIteration(const MatLike& A, int l, int q, int seed = -1);
    
    // Fast algorithms: restricted to dense Eigen matrices (MatrixBase).
    template<typename Derived>
    static CMatrix fastRandomizedRangeFinder(const Eigen::MatrixBase<Derived>& A, int l, int seed = -1);

    template<typename Derived>
    static CMatrix fastRandomizedRangeFinder(const Eigen::MatrixBase<Derived>& A, int l, FRRWorkspace& ws);

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
    template<typename Derived>
    static CMatrix adaptiveFastRandomizedRangeFinder(const Eigen::MatrixBase<Derived>& A, double tol, int l0, int seed = -1);

};

} // namespace randla::algorithms

// Include implementation definitions
#include "randomized_range_finder_impl.hpp"

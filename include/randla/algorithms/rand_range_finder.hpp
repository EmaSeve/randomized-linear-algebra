
#pragma once

#include <random>
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <numeric>
#include <complex>
#include <fftw3.h>
#include <Eigen/Dense>
#include <type_traits>
#include <randla/types.hpp>
#include <randla/metrics/error_estimators.hpp>
#include <randla/random/random_generator.hpp>

/**
 * @file rand_range_finder.hpp
 * @brief Randomized algorithms for range finding in linear algebra.
 *
 * For use in low-rank matrix approximation and randomized SVD.
 */

namespace randla::algorithms {

/**
 * @class RandRangeFinder
 * @brief Implements randomized range finding algorithms for matrices.
 *
 * Provides static methods for randomized range finding, power iteration,
 * subspace iteration, and fast (FFT-based) range finding.
 *
 * @tparam FloatType Floating point type (default: double)
 */
template<typename FloatType = double>
class RandRangeFinder : public randla::Types<FloatType> {
    static_assert(std::is_floating_point_v<FloatType>, 
                  "FloatType must be a floating point type");
public:

    using typename randla::Types<FloatType>::Scalar;
    using typename randla::Types<FloatType>::Matrix;
    using typename randla::Types<FloatType>::Vector;
    using typename randla::Types<FloatType>::Complex;
    using typename randla::Types<FloatType>::CMatrix;
    using typename randla::Types<FloatType>::CVector;

    /**
     * @brief Computes an approximate orthonormal basis for the range of A using a randomized algorithm.
     *
     * Implements the basic randomized range finder (Algorithm 4.1 in Halko et al.).
     *
     * @tparam MatLike Matrix-like type (must support .cols(), .rows(), operator*)
     * @param A Input matrix (m x n)
     * @param l Target subspace dimension (l >= target rank)
     * @param seed Random seed for reproducibility
     * @return Matrix Q (m x l) with orthonormal columns approximating the range of A
     */
    template<class MatLike>
    static Matrix randomizedRangeFinder(const MatLike & A, int l, int seed){
        auto gen = randla::random::RandomGenerator<FloatType>::make_generator(seed);

        // Step 1: Generate a random Gaussian test matrix Omega (n x l)
        Matrix omega = randla::random::RandomGenerator<FloatType>::randomGaussianMatrix(A.cols(), l, gen);

        // Step 2: Form the sample matrix Y = A * Omega (m x l)
        Matrix Y = A * omega;

        // Step 3: Compute a thin QR factorization of Y to obtain Q (m x l)
        Eigen::HouseholderQR<Matrix> qr(Y);
        Matrix Q(Y.rows(), l);
        Q.setIdentity();
        qr.householderQ().applyThisOnTheLeft(Q);

        return Q;
    }

    /**
     * @brief Computes an approximate range using randomized power iteration.
     * 
     * (Algorithm 4.3 in Halko et al.).
     *
     * @tparam MatLike Matrix-like type
     * @param A Input matrix (m x n)
     * @param l Target subspace dimension
     * @param q Number of power iterations (q >= 0)
     * @param seed Random seed
     * @return Matrix Q (m x l) with orthonormal columns
     */
    template<class MatLike>
    static Matrix randomizedPowerIteration(const MatLike& A, int l, int q, int seed) {
        auto gen = randla::random::RandomGenerator<FloatType>::make_generator(seed);
        Matrix Omega = randla::random::RandomGenerator<FloatType>::randomGaussianMatrix(A.cols(), l, gen);

        // Initial sample
        Matrix Y = A * Omega; 
        
        // Power iterations: alternate A^T and A
        for (int i = 0; i < q; ++i) {
            Y = A.transpose() * Y;  // Project onto row space
            Y = A * Y;              // Project back onto column space
        }
        
        // Thin QR to obtain orthonormal basis
        Eigen::HouseholderQR<Matrix> qr(Y);
        Matrix Q(Y.rows(), l);
        Q.setIdentity();
        qr.householderQ().applyThisOnTheLeft(Q);

        return Q;
    }

    /**
     * @brief Computes an approximate range using randomized subspace iteration.
     *
     * More numerically stable than power iteration for some problems.
     * (Algorithm 4.4 in Halko et al.)
     *
     * @tparam MatLike Matrix-like type
     * @param A Input matrix (m x n)
     * @param l Target subspace dimension
     * @param q Number of subspace iterations (q >= 0)
     * @param seed Random seed
     * @return Matrix Q (m x l) with orthonormal columns
     */
    template<class MatLike>
    static Matrix randomizedSubspaceIteration(const MatLike& A, int l, int q, int seed) {
        auto gen = randla::random::RandomGenerator<FloatType>::make_generator(seed);
        Matrix Omega = randla::random::RandomGenerator<FloatType>::randomGaussianMatrix(A.cols(), l, gen);
        
        // Initial sample and QR
        Matrix Y = A * Omega;
        Eigen::HouseholderQR<Matrix> qr0(Y);
        Matrix thinQ = Matrix::Identity(Y.rows(), l);
        thinQ = qr0.householderQ() * thinQ;
        
        // Subspace iterations
        for (int j = 1; j <= q; ++j) {
            // Project onto row space and orthonormalize
            Matrix Y_tilde = A.transpose() * thinQ;
            Eigen::HouseholderQR<Matrix> qr_tilde(Y_tilde);
            Matrix thinQ_tilde = Matrix::Identity(Y_tilde.rows(), l);
            thinQ_tilde = qr_tilde.householderQ() * thinQ_tilde;
            
            // Project back and orthonormalize
            Y = A * thinQ_tilde;
            Eigen::HouseholderQR<Matrix> qr_j(Y);
            Matrix fresh_identity = Matrix::Identity(Y.rows(), l);
            thinQ = qr_j.householderQ() * fresh_identity;
        }
        
        return thinQ;
    }

    /**
     * @brief Fast randomized range finder using subsampled randomized Fourier transform (SRFT).
     *
     * Uses random phase multiplication, FFT, and subsampling for efficient range finding.
     * (Algorithm 4.5 in Halko et al.)
     *
     * @tparam Derived Eigen dense matrix type
     * @param A Input matrix (m x n)
     * @param l Target subspace dimension
     * @param seed Random seed
     * @return Complex matrix Q (m x l) with orthonormal columns
     */
    template<typename Derived>
    static CMatrix fastRandRangeFinder(
        const Eigen::MatrixBase<Derived>& A, int l, int seed)
    {
        using Complex = std::complex<double>;

        // Ensure the matrix is dense (exclude sparse types)
        static_assert(!std::is_base_of_v<Eigen::SparseMatrixBase<Derived>, Derived>,
                      "fastRandRangeFinder supports only dense matrices");

        const int m = A.rows();
        const int n = A.cols();
        
        auto gen = randla::random::RandomGenerator<FloatType>::make_generator(seed);

        // 1. Create random indices for subsampling (R)
        std::vector<int> all(n);
        std::iota(all.begin(), all.end(), 0);
        std::vector<int> indices;
        indices.reserve(l);
        std::sample(all.begin(), all.end(),
                    std::back_inserter(indices),
                    l,
                    gen);

        // 2. Generate random phases (D) on the unit circle
        std::vector<Complex> phase(n);
        std::uniform_real_distribution<FloatType> dist_phase(0.0, 2.0 * M_PI);
        for (int j = 0; j < n; ++j) {
            double ang = dist_phase(gen);
            phase[j] = Complex(std::cos(ang), std::sin(ang));
        }

        // 3. Setup FFTW for in-place 1D FFT
        std::vector<fftw_complex> row(n);
        fftw_plan plan = fftw_plan_dft_1d(n, row.data(), row.data(), FFTW_FORWARD, FFTW_MEASURE);

        // 4. Scaling factor for output
        const FloatType scale = FloatType(1) / std::sqrt(FloatType(l));
        CMatrix Y(m, l);

        // 5. Process each row of A
        for (int i = 0; i < m; ++i) {
            // Apply random phase D to row
            for (int j = 0; j < n; ++j) {
                double aij = A(i, j);
                row[j][0] = aij * phase[j].real();
                row[j][1] = aij * phase[j].imag();
            }

            // Compute FFT in-place
            fftw_execute(plan);

            // Subsample and scale
            for (int t = 0; t < l; ++t) {
                int col_idx = indices[t];
                Y(i, t) = scale * Complex(row[col_idx][0], row[col_idx][1]);
            }
        }

        // Cleanup FFTW resources
        fftw_destroy_plan(plan);

        // 6. Compute thin QR factorization (complex)
        Eigen::HouseholderQR<CMatrix> qr(Y);
        CMatrix Q = CMatrix::Identity(m, l);
        qr.householderQ().applyThisOnTheLeft(Q);

        return Q;
    }

}; // class RandRangeFinder

} // namespace randla::algorithms

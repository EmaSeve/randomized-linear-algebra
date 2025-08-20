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


namespace randla::algorithms {

/**
 * @brief Main class implementing randomized linear algebra algorithms
 * 
 * This class provides implementations of state-of-the-art randomized algorithms
 * for low-rank matrix approximation, including randomized power iteration and
 * randomized subspace iteration, along with error estimation methods.
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

template<class MatLike>
static Matrix randomizedRangeFinder(const MatLike & A, int l, int seed){
    auto gen = randla::random::RandomGenerator<FloatType>::make_generator(seed);

    // Step 1: random test matrix generation
    Matrix omega = randla::random::RandomGenerator<FloatType>::randomGaussianMatrix(A.cols(), l, gen);

    // Step 2: sample the range Y = A * Omega
    Matrix Y = A * omega;

    // Step 3: thin QR to obtain Q
    Eigen::HouseholderQR<Matrix> qr(Y);
    Matrix Q(Y.rows(), l);
    Q.setIdentity();
    qr.householderQ().applyThisOnTheLeft(Q);

    return Q;
}


template<class MatLike>
static Matrix randomizedPowerIteration(const MatLike& A, int l, int q, int seed) {
    auto gen = randla::random::RandomGenerator<FloatType>::make_generator(seed);
    Matrix Omega = randla::random::RandomGenerator<FloatType>::randomGaussianMatrix(A.cols(), l, gen);

    Matrix Y = A * Omega; 
    
    for (int i = 0; i < q; ++i) {
        Y = A.transpose() * Y;  
        Y = A * Y;            
    }
    
    Eigen::HouseholderQR<Matrix> qr(Y);
    Matrix Q(Y.rows(), l);
    Q.setIdentity();
    qr.householderQ().applyThisOnTheLeft(Q);

    return Q;
}

template<class MatLike>
static Matrix randomizedSubspaceIteration(const MatLike& A, int l, int q, int seed) {
    auto gen = randla::random::RandomGenerator<FloatType>::make_generator(seed);
    Matrix Omega = randla::random::RandomGenerator<FloatType>::randomGaussianMatrix(A.cols(), l, gen);
    
    Matrix Y = A * Omega;
    Eigen::HouseholderQR<Matrix> qr0(Y);
    Matrix thinQ = Matrix::Identity(Y.rows(), l);
    thinQ = qr0.householderQ() * thinQ;
    
    for (int j = 1; j <= q; ++j) {
        Matrix Y_tilde = A.transpose() * thinQ;
        Eigen::HouseholderQR<Matrix> qr_tilde(Y_tilde);
        Matrix thinQ_tilde = Matrix::Identity(Y_tilde.rows(), l);
        thinQ_tilde = qr_tilde.householderQ() * thinQ_tilde;
        
        Y = A * thinQ_tilde;
        Eigen::HouseholderQR<Matrix> qr_j(Y);
        Matrix fresh_identity = Matrix::Identity(Y.rows(), l);
        thinQ = qr_j.householderQ() * fresh_identity;
    }
    
    return thinQ;
}

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

    // 1. Create random indices (R) 
    std::vector<int> all(n);
    std::iota(all.begin(), all.end(), 0);
    std::vector<int> indices;
    indices.reserve(l);
    std::sample(all.begin(), all.end(),
                std::back_inserter(indices),
                l,
                gen);

    // 2. Generate random phases (D)
    std::vector<Complex> phase(n);
    std::uniform_real_distribution<FloatType> dist_phase(0.0, 2.0 * M_PI);
    for (int j = 0; j < n; ++j) {
        double ang = dist_phase(gen);
        phase[j] = Complex(std::cos(ang), std::sin(ang));
    }

    // 3. Setup FFTW
    std::vector<fftw_complex> row(n);
    fftw_plan plan = fftw_plan_dft_1d(n, row.data(), row.data(), FFTW_FORWARD, FFTW_MEASURE);

    // 4. Scale factor
    const FloatType scale = FloatType(1) / std::sqrt(FloatType(l));
    CMatrix Y(m, l);

    // 5. Process each row
    for (int i = 0; i < m; ++i) {
        // Load and apply D
        for (int j = 0; j < n; ++j) {
            double aij = A(i, j);
            row[j][0] = aij * phase[j].real();
            row[j][1] = aij * phase[j].imag();
        }

        // FFT in-place
        fftw_execute(plan);

        // Subsampling + scaling
        for (int t = 0; t < l; ++t) {
            int col_idx = indices[t];
            Y(i, t) = scale * Complex(row[col_idx][0], row[col_idx][1]);
        }
    }

    // Cleanup FFTW
    fftw_destroy_plan(plan);

    // 6. Complex QR
    Eigen::HouseholderQR<CMatrix> qr(Y);
    CMatrix Q = CMatrix::Identity(m, l);
    qr.householderQ().applyThisOnTheLeft(Q);

    return Q;
}

}; // class RandRangeFinder

} // namespace randla::algorithms

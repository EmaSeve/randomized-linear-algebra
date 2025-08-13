#pragma once

#include <random>
#include <chrono>
#include <cmath>
#include <Eigen/QR>
#include <Eigen/SVD> 
#include <stdexcept>
#include <algorithm>
#include <numeric>
#include <complex>
#include <fftw3.h>

#include <randla/metrics/error_estimators.hpp>

namespace randla::algorithms {

inline std::mt19937 make_generator(int seed) {
    if (seed >= 0) return std::mt19937(seed);
    return std::mt19937(std::chrono::steady_clock::now().time_since_epoch().count());
}

template<typename FloatType>
typename RandomizedRangeFinder<FloatType>::Matrix 
RandomizedRangeFinder<FloatType>::randomGaussianMatrix(int rows, int cols, std::mt19937 & gen) {
    Matrix result(rows, cols);
    std::normal_distribution<FloatType> dist(0.0, 1.0);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            result(i, j) = dist(gen);
        }
    }
    return result;
}

template<typename FloatType>
typename RandomizedRangeFinder<FloatType>::Matrix 
RandomizedRangeFinder<FloatType>::randomGaussianMatrix(int rows, int cols, int seed){
    auto gen = make_generator(seed);
    return randomGaussianMatrix(rows, cols, gen);
}

template<typename FloatType>
typename RandomizedRangeFinder<FloatType>::Vector 
RandomizedRangeFinder<FloatType>::randomGaussianVector(int size, std::mt19937 & gen) {
    Vector result(size);
    std::normal_distribution<FloatType> dist(0.0, 1.0);
    for (int i = 0; i < size; ++i) {
        result(i) = dist(gen);
    }
    return result;
}

template<typename FloatType>
typename RandomizedRangeFinder<FloatType>::Vector 
RandomizedRangeFinder<FloatType>::randomGaussianVector(int size, int seed){
    auto gen = make_generator(seed);
    return randomGaussianVector(size, gen);
}

template<typename FloatType>
typename RandomizedRangeFinder<FloatType>::Matrix 
RandomizedRangeFinder<FloatType>::randomizedRangeFinder(const Matrix & A, int l, int seed){
    auto gen = make_generator(seed);
    // step 1: draw test matrix
    Matrix omega = randomGaussianMatrix(A.cols(), l, gen);
    // step 2.
    Matrix Y = A * omega;
    // step 3.
    Eigen::HouseholderQR<Matrix> qr(Y);
    Matrix Q(Y.rows(), l);
    Q.setIdentity();
    qr.householderQ().applyThisOnTheLeft(Q);

    return Q;
}

template<typename FloatType>
typename RandomizedRangeFinder<FloatType>::Matrix
RandomizedRangeFinder<FloatType>::adaptiveRangeFinder(const Matrix & A, double tol, int r, int seed){
    
    const size_t rows = A.rows();
    const size_t cols = A.cols();

    auto gen = make_generator(seed);
    // draw 'r' standard gaussian vectors: Matrix omega
    Matrix omega = randomGaussianMatrix(cols, r, gen);
    // compute the vector y_i: Matrix Y
    Matrix Y(rows, r);
    Y = A * omega;

    int iteration = -1;

    // start with empty Q
    Matrix Q(rows, 0);

    const double threshold = tol / (10.0 * std::sqrt(2.0 / M_PI));
    size_t index;

    while(Y.colwise().norm().maxCoeff() >  threshold){
        iteration++;
        index = iteration % r;

        Vector y_i = Y.col(index);
        y_i -= Q * (Q.transpose() * y_i);

        // normalize and compute the new column q_i
        const double norm = y_i.norm();
        if(norm > 0) y_i.normalize();

        // add new column to Q
        Q.conservativeResize(Eigen::NoChange, Q.cols() + 1);
        Q.col(Q.cols() - 1) = y_i;
        const auto q_i = Q.col(Q.cols() - 1); 

        // draw standard gaussian vector w_i 
        Vector w_i = randomGaussianVector(cols, gen);
        
        // replace the vector y_j with y_j+r 
        Vector temp = A * w_i;                       
        temp -= Q * (Q.transpose() * temp);          
        Y.col(index) = temp;

        // update all the other vector except the new one
        for(size_t j = 0; j < r; j++){
            if(j == index) continue;
            // overwrite y_j = y_j - q_it * <q_it, y_j>
            Y.col(j) -= q_i * q_i.dot(Y.col(j));  
        }
    }

    return Q;
}

template<typename FloatType>
typename RandomizedRangeFinder<FloatType>::Matrix 
RandomizedRangeFinder<FloatType>::randomizedPowerIteration(const Matrix& A, int l, int q, int seed) {
    auto gen = make_generator(seed);
    Matrix Omega = randomGaussianMatrix(A.cols(), l, gen);

    Matrix Y = A * Omega;  // First application: A * Ω
    
    for (int i = 0; i < q; ++i) {
        Y = A.transpose() * Y;  // Apply A*
        Y = A * Y;              // Apply A
    }
    
    Eigen::HouseholderQR<Matrix> qr(Y);
    Matrix Q(Y.rows(), l);
    Q.setIdentity();
    qr.householderQ().applyThisOnTheLeft(Q);

    return Q;
}

template<typename FloatType>
typename RandomizedRangeFinder<FloatType>::Matrix 
RandomizedRangeFinder<FloatType>::adaptivePowerIteration(const Matrix& A, double tol, int r, int q, int seed) {

    const size_t rows = A.rows();
    const size_t cols = A.cols();

    const double threshold = tol / (10.0 * std::sqrt(2.0 / M_PI));

    auto apply_power = [&](const Vector& w) -> Vector {
        Vector y = A * w;
        for (int t = 0; t < q; ++t) {
            y = A.transpose() * y;
            y = A * y;
        }
        return y;
    };

    auto gen = make_generator(seed);
    Matrix Omega = randomGaussianMatrix(cols, r, gen);
    Matrix Y(rows, r);
    for (int j = 0; j < r; ++j) {
        Y.col(j) = apply_power(Omega.col(j));
    }

    Matrix Q(rows, 0);
    int iteration = -1;
    size_t index;

    while (Y.colwise().norm().maxCoeff() > threshold) {
        iteration++;
        index = iteration % r;

        Vector y_i = Y.col(index);
        if (Q.cols() > 0) {
            y_i -= Q * (Q.transpose() * y_i);
        }

        const double norm = y_i.norm();
        if (norm > 0) y_i.normalize();

        Q.conservativeResize(Eigen::NoChange, Q.cols() + 1);
        Q.col(Q.cols() - 1) = y_i;
        const auto q_i = Q.col(Q.cols() - 1);

        Vector w_new = randomGaussianVector(cols, gen);
        Vector y_new = apply_power(w_new);

        y_new -= Q * (Q.transpose() * y_new);
        Y.col(index) = y_new;

        for (size_t j = 0; j < r; ++j) {
            if (j == index) continue;
            Y.col(j) -= q_i * q_i.dot(Y.col(j));
        }
    }

    return Q;
}

template<typename FloatType>
typename RandomizedRangeFinder<FloatType>::Matrix 
RandomizedRangeFinder<FloatType>::randomizedSubspaceIteration(const Matrix& A, int l, int q, int seed) {
    auto gen = make_generator(seed);
    Matrix Omega = randomGaussianMatrix(A.cols(), l, gen);
    
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

template<typename FloatType>
typename RandomizedRangeFinder<FloatType>::CMatrix
RandomizedRangeFinder<FloatType>::fastRandomizedRangeFinder(const Matrix& A, int l, int seed) {

    static_assert(std::is_same_v<FloatType,double>,
                  "This implementation uses FFTW double-precision");

    using Complex = std::complex<double>;

    auto gen = make_generator(seed);
    const int m = A.rows();
    const int n = A.cols();

    // 1. Random permutation (R)
    std::vector<int> indices(n);
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), gen);
    indices.resize(l);

    // 2. Precompute random unit phases (D)
    std::uniform_real_distribution<FloatType> dist_phase(0.0, 2.0 * M_PI);
    std::vector<Complex> phase(n);
    for (int j = 0; j < n; ++j) {
        double ang = dist_phase(gen);
        phase[j] = Complex(std::cos(ang), std::sin(ang));
    }

    // 3. FFT plan for a reusable row buffer
    std::vector<fftw_complex> row(n);
    fftw_plan plan = fftw_plan_dft_1d(
        n,
        row.data(),
        row.data(),
        FFTW_FORWARD,
        FFTW_MEASURE // FFTW_ESTIMATE if you prefer faster planning
    );

    // 4. Allocate Y and set correct scaling
    // FFTW's DFT is unnormalized -> scaling = 1 / sqrt(l)
    const FloatType scale = FloatType(1) / std::sqrt(FloatType(l));
    CMatrix Y(m, l);

    for (int i = 0; i < m; ++i) {
        // Load row of A, apply D (phases)
        for (int j = 0; j < n; ++j) {
            double aij = A(i, j);
            row[j][0] = aij * phase[j].real();
            row[j][1] = aij * phase[j].imag();
        }

        // Apply FFT (F)
        fftw_execute(plan);

        // Subsample R and scale
        for (int t = 0; t < l; ++t) {
            int col_idx = indices[t];
            Y(i, t) = scale * Complex(row[col_idx][0], row[col_idx][1]);
        }
    }

    fftw_destroy_plan(plan);

    // 5. Complex QR
    Eigen::HouseholderQR<CMatrix> qr(Y);
    CMatrix Q = CMatrix::Identity(m, l);
    qr.householderQ().applyThisOnTheLeft(Q);

    return Q;
}


template<typename FloatType>
typename RandomizedRangeFinder<FloatType>::CMatrix
RandomizedRangeFinder<FloatType>::adaptiveFastRandomizedRangeFinder(
    const Matrix& A,
    double tol,   // tolleranza assoluta
    int l0,       // campioni iniziali
    int seed      // seed RNG (negativo per time-based)
) {
    const int m = A.rows();
    const int n = A.cols();

    const int lmax = std::min(m, n);

    int l = l0;
    CMatrix Qc;

    while (true) {
        Qc = fastRandomizedRangeFinder(A, l, seed);

    double err_abs = ErrorEstimators<FloatType>::realError(A, Qc);

        if (err_abs <= tol || l >= lmax) {
            break;
        }

        l = std::min(l * 2, lmax);
    }

    return Qc;
}


// (Error and factorization related implementations moved to ErrorEstimators / MatrixFactorizer.)

} // namespace randla::algorithms

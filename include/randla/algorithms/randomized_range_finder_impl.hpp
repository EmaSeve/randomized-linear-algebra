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
template<class MatLike>
typename RandomizedRangeFinder<FloatType>::Matrix 
RandomizedRangeFinder<FloatType>::randomizedRangeFinder(const MatLike & A, int l, int seed){
    auto gen = make_generator(seed);

    Matrix omega = randomGaussianMatrix(A.cols(), l, gen);

    Matrix Y = A * omega;

    Eigen::HouseholderQR<Matrix> qr(Y);
    Matrix Q(Y.rows(), l);
    Q.setIdentity();
    qr.householderQ().applyThisOnTheLeft(Q);

    return Q;
}

template<typename FloatType>
template<class MatLike>
typename RandomizedRangeFinder<FloatType>::Matrix
RandomizedRangeFinder<FloatType>::adaptiveRangeFinder(const MatLike & A, double tol, int r, int seed){
    
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
template<class MatLike>
typename RandomizedRangeFinder<FloatType>::Matrix 
RandomizedRangeFinder<FloatType>::randomizedPowerIteration(const MatLike& A, int l, int q, int seed) {
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
template<class MatLike>
typename RandomizedRangeFinder<FloatType>::Matrix 
RandomizedRangeFinder<FloatType>::adaptivePowerIteration(const MatLike& A, double tol, int r, int q, int seed) {

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
template<class MatLike>
typename RandomizedRangeFinder<FloatType>::Matrix 
RandomizedRangeFinder<FloatType>::randomizedSubspaceIteration(const MatLike& A, int l, int q, int seed) {
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

struct FRRWorkspace {
    std::vector<std::complex<double>> phase; // fasi casuali
    std::vector<fftw_complex> row;           // buffer riga FFT
    fftw_plan plan;                          // piano FFTW
    std::mt19937 gen;                        // RNG
    int n;                                   // numero colonne

    FRRWorkspace(int n_, int seed)
        : phase(n_), row(n_), gen(make_generator(seed)), n(n_)
    {
        plan = fftw_plan_dft_1d(
            n,
            row.data(),
            row.data(),
            FFTW_FORWARD,
            FFTW_MEASURE
        );
    }

    ~FRRWorkspace() {
        fftw_destroy_plan(plan);
    }
};


template<typename FloatType>
template<typename Derived>
typename RandomizedRangeFinder<FloatType>::CMatrix
RandomizedRangeFinder<FloatType>::fastRandomizedRangeFinder(
    const Eigen::MatrixBase<Derived>& A, int l, FRRWorkspace& ws)
{
    using Complex = std::complex<double>;

    // Ensure the matrix is dense (exclude sparse types)
    static_assert(!std::is_base_of_v<Eigen::SparseMatrixBase<Derived>, Derived>,
                  "fastRandomizedRangeFinder supports only dense matrices");

    const int m = A.rows();
    const int n = A.cols();

    // 1. Estrai indici random (R) con std::sample (C++17 friendly)
    std::vector<int> all(n);
    std::iota(all.begin(), all.end(), 0);

    std::vector<int> indices;
    indices.reserve(l);
    std::sample(all.begin(), all.end(),
                std::back_inserter(indices),
                l,
                ws.gen);

    // 2. Rigenera fasi casuali (D) in-place
    std::uniform_real_distribution<FloatType> dist_phase(0.0, 2.0 * M_PI);
    for (int j = 0; j < n; ++j) {
        double ang = dist_phase(ws.gen);
        ws.phase[j] = Complex(std::cos(ang), std::sin(ang));
    }

    // 3. Scala corretto: FFTW non normalizza → 1/sqrt(l)
    const FloatType scale = FloatType(1) / std::sqrt(FloatType(l));
    CMatrix Y(m, l);

    // 4. Loop sulle righe
    for (int i = 0; i < m; ++i) {
        // Carica e applica D
        for (int j = 0; j < n; ++j) {
            double aij = A(i, j);
            ws.row[j][0] = aij * ws.phase[j].real();
            ws.row[j][1] = aij * ws.phase[j].imag();
        }

        // FFT in-place
        fftw_execute(ws.plan);

        // Subsampling + scaling
        for (int t = 0; t < l; ++t) {
            int col_idx = indices[t];
            Y(i, t) = scale * Complex(ws.row[col_idx][0], ws.row[col_idx][1]);
        }
    }

    // 5. Complex QR
    Eigen::HouseholderQR<CMatrix> qr(Y);
    CMatrix Q = CMatrix::Identity(m, l);
    qr.householderQ().applyThisOnTheLeft(Q);

    return Q;
}

template<typename FloatType>
template<typename Derived>
typename RandomizedRangeFinder<FloatType>::CMatrix
RandomizedRangeFinder<FloatType>::fastRandomizedRangeFinder(
    const Eigen::MatrixBase<Derived>& A, int l, int seed)
{
    FRRWorkspace ws(A.cols(), seed);
    return fastRandomizedRangeFinder(A, l, ws);
}


template<typename FloatType>
template<typename Derived>
typename RandomizedRangeFinder<FloatType>::CMatrix
RandomizedRangeFinder<FloatType>::adaptiveFastRandomizedRangeFinder(
    const Eigen::MatrixBase<Derived>& A,
    double tol,
    int l0,
    int seed
) {
    const int m = A.rows();
    const int n = A.cols();
    const int lmax = std::min(m, n);

    // Workspace persistente per tutto l’adaptive
    FRRWorkspace ws(n, seed);

    int l = l0;
    CMatrix Qc;

    while (true) {
        Qc = fastRandomizedRangeFinder(A, l, ws);

        double err_abs = ErrorEstimators<FloatType>::realError(A, Qc);

        if (err_abs <= tol || l >= lmax) {
            break;
        }

        l = std::min(l * 2, lmax);
    }

    return Qc;
}

} // namespace randla::algorithms

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
class RandomizedRangeFinder : public randla::Types<FloatType> {
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
static Matrix adaptiveRangeFinder(
    const MatLike& A, double tol, int r, int seed)
{
    using Matrix = typename RandomizedRangeFinder<FloatType>::Matrix;
    using Vector = typename RandomizedRangeFinder<FloatType>::Vector;

    const int m = static_cast<int>(A.rows());
    const int n = static_cast<int>(A.cols());
    if (r <= 0) return Matrix(m, 0);

    auto gen = randla::random::RandomGenerator<FloatType>::make_generator(seed);

    // --- calcolo della dimensioned del blocco in base al numero di thread
    auto ceil_div = [](int a, int b){ return (a + b - 1) / b; };
    int threads = std::max(1, Eigen::nbThreads());
    int b = ceil_div(r, threads);
    b = std::clamp(b, 8, 64);
    if (b < r) b = std::min(((b + 7) / 8) * 8, r);

    // --- inizializzazioni
    Matrix W = randla::random::RandomGenerator<FloatType>::randomGaussianMatrix(n, r, gen);
    Matrix Y = A * W; 

    const double threshold = tol / (10.0 * std::sqrt(2.0 / M_PI));

    std::vector<Matrix> Q_blocks;
    Q_blocks.reserve((r + b - 1) / b);

    Matrix Yb(m, b), Tb(m, b);

    auto max_col_norm = [&](){
        return Y.colwise().norm().maxCoeff();
    };

    // Proietta M sui blocchi correnti: M -= sum_i Qi * (Qi^T * M)
    auto project_against_blocks = [&](Matrix& M){
        for (const auto& Qi : Q_blocks) {
            Matrix Ti = Qi.transpose() * M; // (bi x k)
            M.noalias() -= Qi * Ti;         // BLAS3
        }
    };

    // Estrai nb colonne di Y a partire da "start"
    auto take_block_cyclic = [&](const Matrix& Src, Matrix& Dst, int start, int nb){
        const int end = start + nb;
        if (end <= r) {
            Dst.leftCols(nb).noalias() = Src.middleCols(start, nb);
        } else {
            const int k1 = r - start, k2 = nb - k1;
            Dst.leftCols(k1).noalias()  = Src.middleCols(start, k1);
            Dst.middleCols(k1, k2).noalias() = Src.leftCols(k2);
        }
    };

    // Scrivi nb colonne in Y a partire da "start"
    auto write_block_cyclic = [&](Matrix& Dst, const Matrix& Src, int start, int nb){
        const int end = start + nb;
        if (end <= r) {
            Dst.middleCols(start, nb).noalias() = Src.leftCols(nb);
        } else {
            const int k1 = r - start, k2 = nb - k1;
            Dst.middleCols(start, k1).noalias() = Src.leftCols(k1);
            Dst.leftCols(k2).noalias()          = Src.middleCols(k1, k2);
        }
    };

    int start = 0;
    while (max_col_norm() > threshold) {
        const int nb = std::min(b, r);

        // 1) Estrai blocco ciclico corrente Yb
        take_block_cyclic(Y, Yb, start, nb);

        // 2) Re-ortho bloccata rispetto ai blocchi già trovati
        project_against_blocks(Yb);

        // 3) TSQR/Householder QR su Yb -> Qb
        Eigen::HouseholderQR<Matrix> qr(Yb);
        Matrix Qb = qr.householderQ() * Matrix::Identity(m, nb);
        Q_blocks.emplace_back(std::move(Qb)); // accumula nuovo blocco

        // 4) nuove gaussiane e proietta su Q_blocks
        Matrix Wb_new = randla::random::RandomGenerator<FloatType>::randomGaussianMatrix(n, nb, gen);
        Tb.noalias() = A * Wb_new;
        project_against_blocks(Tb);
        // proietta anche sul blocco appena aggiunto (Q_blocks.back())
        {
            const Matrix& Qlast = Q_blocks.back();
            Matrix Tlast = Qlast.transpose() * Tb;
            Tb.noalias() -= Qlast * Tlast;
        }
        write_block_cyclic(Y, Tb, start, nb); // aggiorna Y nelle posizioni cicliche

        // 5) Aggiorna tutte le colonne rimanenti: Y -= Qb * (Qb^T * Y)
        {
            const Matrix& Qlast = Q_blocks.back();
            Matrix QtAll = Qlast.transpose() * Y;
            Y.noalias() -= Qlast * QtAll;
        }

        start = (start + nb) % r;
    }

    // --- Concatena alla fine
    int qcols = 0;
    for (const auto& Bi : Q_blocks) qcols += static_cast<int>(Bi.cols());
    Matrix Qfinal(m, qcols);
    {
        int off = 0;
        for (const auto& Bi : Q_blocks) {
            const int c = static_cast<int>(Bi.cols());
            Qfinal.middleCols(off, c).noalias() = Bi;
            off += c;
        }
    }
    return Qfinal;
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
static Matrix adaptivePowerIteration(const MatLike& A, double tol, int r, int q, int seed) {

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

    auto gen = randla::random::RandomGenerator<FloatType>::make_generator(seed);
    Matrix Omega = randla::random::RandomGenerator<FloatType>::randomGaussianMatrix(cols, r, gen);
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

        Vector w_new = randla::random::RandomGenerator<FloatType>::randomGaussianVector(cols, gen);
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
static CMatrix fastRandomizedRangeFinder(
    const Eigen::MatrixBase<Derived>& A, int l, int seed)
{
    using Complex = std::complex<double>;

    // Ensure the matrix is dense (exclude sparse types)
    static_assert(!std::is_base_of_v<Eigen::SparseMatrixBase<Derived>, Derived>,
                  "fastRandomizedRangeFinder supports only dense matrices");

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


template<typename Derived>
static CMatrix adaptiveFastRandomizedRangeFinder(
    const Eigen::MatrixBase<Derived>& A,
    double tol,
    int l0,
    int seed,
    double growth_factor = 1.5
) {
    const int m = A.rows();
    const int n = A.cols();
    const int lmax = std::min(m, n);

    int l = l0;
    CMatrix Qc;

    while (true) {
        Qc = fastRandomizedRangeFinder(A, l, seed);

        double err_abs = randla::metrics::ErrorEstimators<FloatType>::realError(A, Qc);

        if (err_abs <= tol || l >= lmax) {
            break;
        }

        // Decide next l based on growth_factor semantics
        int next_l;
        if (growth_factor > 1.0) {
            next_l = static_cast<int>(std::ceil(l * growth_factor));
        } else if (growth_factor > 0.0) {
            // additive relative to l0
            int step = static_cast<int>(std::ceil(growth_factor * l0));
            if (step <= 0) step = 1;
            next_l = l + step;
        } else {
            // fallback minimal additive growth
            next_l = l + 1;
        }
        if (next_l <= l) next_l = l + 1; // ensure progress
        l = std::min(next_l, lmax);
    }

    return Qc;
}

}; // class RandomizedRangeFinder

} // namespace randla::algorithms

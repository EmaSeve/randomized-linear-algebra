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
#include <randla/algorithms/rand_range_finder.hpp>


namespace randla::algorithms {

template<typename FloatType = double>
class AdaptiveRandRangeFinder : public randla::Types<FloatType> {
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
static Matrix adaptiveRangeFinder(
    const MatLike& A, double tol, int r, int seed)
{
    using Matrix = typename Types<FloatType>::Matrix;
    using Vector = typename Types<FloatType>::Vector;
    int b = Eigen::nbThreads();

    const int m = static_cast<int>(A.rows());
    const int n = static_cast<int>(A.cols());
    if (r <= 0) return Matrix(m, 0);
    b = std::max(1, std::min(b, r)); // clamp

    auto gen = randla::random::RandomGenerator<FloatType>::make_generator(seed);

    // --- initial random probes
    Matrix Omega = randla::random::RandomGenerator<FloatType>::randomGaussianMatrix(n, r, gen);
    Matrix Y = A * Omega;                      // m x r
    Matrix Q(m, 0);                            // start empty

    const double threshold = tol / (10.0 * std::sqrt(2.0 / M_PI));
    auto max_col_norm = [&](const Matrix& M){
        // avoids temporaries: compute squared norms then sqrt once
        Eigen::RowVectorXd s = M.colwise().squaredNorm();
        return std::sqrt(s.maxCoeff());
    };

    // For efficient appends, collect Q in blocks and concat at the end.
    std::vector<Matrix> Q_blocks;
    Q_blocks.reserve((r + b - 1) / b);

    int it = 0;
    while (max_col_norm(Y) > threshold)
    {
        // ---- pick a cyclic block of columns
        int start = (it * b) % r;
        Eigen::ArrayXi idx(b);
        for (int j = 0; j < b; ++j) idx(j) = (start + j) % r;

        // Gather V = Y(:, idx)  [m x b]
        Matrix V(m, b);
        for (int j = 0; j < b; ++j) V.col(j) = Y.col(idx(j));

        // Re-orthogonalize V against all previously found Q:  V -= Q*(Qᵀ V)
        if (!Q_blocks.empty()) {
            // Stack-view of Q without materializing if desired; for clarity, materialize:
            int qcols = 0;
            for (auto& Bl : Q_blocks) qcols += static_cast<int>(Bl.cols());
            Q.resize(m, qcols);
            int off = 0;
            for (auto& Bl : Q_blocks) { Q.block(0, off, m, Bl.cols()) = Bl; off += Bl.cols(); }
            V.noalias() -= Q * (Q.transpose() * V);
        }

        // Thin QR on V -> Qb (m x kb). Some columns may be near-zero; drop them.
        Eigen::ColPivHouseholderQR<Matrix> qr(V);
        qr.setThreshold(1e-12);
        const int kb = qr.rank();
        if (kb == 0) {
            // All directions collapsed; refresh the picked Y-columns and continue.
            Matrix W = randla::random::RandomGenerator<FloatType>::randomGaussianMatrix(n, b, gen);
            Matrix T = A * W;
            // Project off existing Q
            if (Q.cols() > 0) T.noalias() -= Q * (Q.transpose() * T);
            for (int j = 0; j < b; ++j) Y.col(idx(j)) = T.col(j);
            ++it; 
            continue;
        }
        Matrix Qb = qr.householderQ().setLength(kb); // m x kb, orthonormal

        // Normalize just in case (should already be orthonormal)
        // for (int j=0;j<kb;++j) Qb.col(j).normalize();

        // Append new block
        Q_blocks.push_back(std::move(Qb));
        int qcols = 0;
        for (auto& Bl : Q_blocks) qcols += static_cast<int>(Bl.cols());
        Q.resize(m, qcols);
        int off = 0;
        for (auto& Bl : Q_blocks) { Q.block(0, off, m, Bl.cols()) = Bl; off += Bl.cols(); }

        // ---- Refresh the chosen Y columns with new random directions in one shot
        Matrix W = randla::random::RandomGenerator<FloatType>::randomGaussianMatrix(n, b, gen);  // n x b
        Matrix T = A * W;                                                                        // m x b
        // Remove components along the *new* Q (all of it, but doing it once is fine)
        if (Q.cols() > 0) T.noalias() -= Q * (Q.transpose() * T);
        for (int j = 0; j < b; ++j) Y.col(idx(j)) = T.col(j);

        // ---- Update ALL Y by removing components along the newly added block only:
        //      Y := Y - Qb * (Qbᵀ Y)
        {
            const Matrix& Qb_last = Q_blocks.back();
            Y.noalias() -= Qb_last * (Qb_last.transpose() * Y);
        }

        ++it;
    }

    // Concatenate blocks to final Q (already done in-loop).
    return Q;
}


template<class MatLike>
static Matrix adaptivePowerIteration(const MatLike& A, double tol, int r, int q, int seed) {

    const size_t rows = A.rows();
    const size_t cols = A.cols();

    const double threshold = tol / (10.0 * std::sqrt(2.0 / M_PI));

    Matrix Q(rows, 0);

    auto apply_power = [&](Vector w) -> Vector {
        // step 0
        w.normalize();
        Vector y = A * w;

        for (int t = 0; t < q; ++t) {
            // opzionale: proietta su complemento di Q a ogni step:
            if (Q.cols() > 0) y -= Q * (Q.transpose() * y);
            y.normalize();

            Vector z = A.transpose() * y;
            if (Q.cols() > 0) z -= A.transpose() * (Q * (Q.transpose() * y)); // oppure solo normalize
            z.normalize();

            y = A * z;
        }
        return y;
    };


    auto gen = randla::random::RandomGenerator<FloatType>::make_generator(seed);
    Matrix Omega = randla::random::RandomGenerator<FloatType>::randomGaussianMatrix(cols, r, gen);
    Matrix Y(rows, r);
    for (int j = 0; j < r; ++j) {
        Y.col(j) = apply_power(Omega.col(j));
    }

    
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

template<typename Derived>
static CMatrix adaptiveFastRandRangeFinder(
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
        Qc = randla::algorithms::RandRangeFinder<FloatType>::fastRandRangeFinder(A, l, seed);

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

};
}
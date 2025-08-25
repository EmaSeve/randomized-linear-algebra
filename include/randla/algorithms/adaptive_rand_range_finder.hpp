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

/**
 * @brief Adaptive randomized range finder algorithms for low-rank matrix approximation.
 * 
 * Provides adaptive algorithms to construct an approximate orthonormal basis for the range of a matrix,
 * with error control and optional power iteration for improved accuracy.
 * 
 * @tparam FloatType Floating point type (float or double).
 */
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

/**
 * @brief Adaptive randomized range finder.
 * 
 * Iteratively builds an orthonormal basis Q such that ||A - QQ^T A|| <= tol, 
 * using an approximation of the error.
 * The algorithm adaptively increases the basis size until the error is below the threshold.
 * (Algorithm 4.2 in Halko et al.)
 * 
 * @tparam MatLike Matrix-like type supporting .rows(), .cols(), operator*.
 * @param A Input matrix.
 * @param tol Target absolute error tolerance.
 * @param r Number of random vectors to use per iteration.
 * @param seed Random seed for reproducibility.
 * @return Matrix Orthonormal basis Q.
 */
template<class MatLike> 
static Matrix adaptiveRangeFinder(
    const MatLike& A, double tol, int r, int seed)
{
    const size_t rows = A.rows();
    const size_t cols = A.cols();

    auto gen = randla::random::RandomGenerator<FloatType>::make_generator(seed);
    Matrix omega = randla::random::RandomGenerator<FloatType>::randomGaussianMatrix(cols, r, gen);
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
        Vector w_i = randla::random::RandomGenerator<FloatType>::randomGaussianVector(cols, gen);

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

/**
 * @brief Adaptive randomized range finder with power iteration.
 * 
 * Uses power iteration to improve the quality of the basis, 
 * especially for matrices with slow spectral decay.
 * 
 * @tparam MatLike Matrix-like type supporting .rows(), .cols(), operator*.
 * @param A Input matrix.
 * @param tol Target absolute error tolerance.
 * @param r Number of random vectors to use per iteration.
 * @param q Number of power iterations.
 * @param seed Random seed for reproducibility.
 * @return Matrix Orthonormal basis Q.
 */
template<class MatLike>
static Matrix adaptivePowerIteration(const MatLike& A, double tol, int r, int q, int seed) {

    const size_t rows = A.rows();
    const size_t cols = A.cols();

    // Compute error threshold for stopping
    const double threshold = tol / (10.0 * std::sqrt(2.0 / M_PI));

    // Start with empty orthonormal basis Q
    Matrix Q(rows, 0);

    // Lambda to apply q steps of power iteration to a vector w
    auto apply_power = [&](Vector w) -> Vector {
        w.normalize();
        Vector y = A * w;

        for (int t = 0; t < q; ++t) {
            // Orthogonalize against current Q
            if (Q.cols() > 0) y -= Q * (Q.transpose() * y);
            y.normalize();

            // Back-projection step
            Vector z = A.transpose() * y;
            if (Q.cols() > 0) z -= A.transpose() * (Q * (Q.transpose() * y));
            z.normalize();

            // Forward-projection step
            y = A * z;
        }
        return y;
    };

    // Generate r random Gaussian vectors and apply power iteration
    auto gen = randla::random::RandomGenerator<FloatType>::make_generator(seed);
    Matrix Omega = randla::random::RandomGenerator<FloatType>::randomGaussianMatrix(cols, r, gen);
    Matrix Y(rows, r);
    for (int j = 0; j < r; ++j) {
        Y.col(j) = apply_power(Omega.col(j));
    }

    int iteration = -1;
    size_t index;

    // Main adaptive loop: grow Q until error is below threshold
    while (Y.colwise().norm().maxCoeff() > threshold) {
        iteration++;
        index = iteration % r;

        // Orthogonalize y_i against Q
        Vector y_i = Y.col(index);
        if (Q.cols() > 0) {
            y_i -= Q * (Q.transpose() * y_i);
        }

        // Normalize and append new basis vector to Q
        const double norm = y_i.norm();
        if (norm > 0) y_i.normalize();

        Q.conservativeResize(Eigen::NoChange, Q.cols() + 1);
        Q.col(Q.cols() - 1) = y_i;
        const auto q_i = Q.col(Q.cols() - 1);

        // Draw new random vector, apply power iteration, and orthogonalize
        Vector w_new = randla::random::RandomGenerator<FloatType>::randomGaussianVector(cols, gen);
        Vector y_new = apply_power(w_new);

        y_new -= Q * (Q.transpose() * y_new);
        Y.col(index) = y_new;

        // Update all other Y columns to remain orthogonal to new q_i
        for (size_t j = 0; j < r; ++j) {
            if (j == index) continue;
            Y.col(j) -= q_i * q_i.dot(Y.col(j));
        }
    }

    return Q;
}

/**
 * @brief Adaptive fast randomized range finder.
 * 
 * Progressively increase the number of samples computed using fastRandRangeFinder
 * 
 * @tparam Derived Eigen matrix type.
 * @param A Input matrix (Eigen type).
 * @param tol Target absolute error tolerance.
 * @param l0 Initial sketch size.
 * @param seed Random seed for reproducibility.
 * @param growth_factor Multiplicative or additive growth factor for sketch size.
 * @return CMatrix Orthonormal basis Q (complex-valued).
 */
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
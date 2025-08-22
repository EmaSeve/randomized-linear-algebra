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
    const size_t rows = A.rows();
    const size_t cols = A.cols();

    auto gen = randla::random::RandomGenerator<FloatType>::make_generator(seed);
    // draw 'r' standard gaussian vectors: Matrix omega
    Matrix omega = randla::random::RandomGenerator<FloatType>::randomGaussianMatrix(cols, r, gen);
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
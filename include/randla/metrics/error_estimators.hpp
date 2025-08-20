#pragma once

#include <Eigen/Dense>
#include <random>
#include <cmath>
#include <randla/types.hpp>
#include <random>
#include <chrono>
#include <numbers>
#include <randla/random/random_generator.hpp>

namespace randla::metrics {

/**
 * @brief Error estimation utilities (moved from RandomizedRangeFinder Stage A)
 */
template<typename FloatType = double>
class ErrorEstimators : public randla::Types<FloatType> {
    static_assert(std::is_floating_point_v<FloatType>, "FloatType must be a floating point type");
public:
    using typename randla::Types<FloatType>::Scalar;
    using typename randla::Types<FloatType>::Matrix;
    using typename randla::Types<FloatType>::Vector;
    using typename randla::Types<FloatType>::Complex;
    using typename randla::Types<FloatType>::CMatrix;
    using typename randla::Types<FloatType>::CVector;


static Scalar posteriorErrorEstimation(const Matrix& A, const Matrix& Q, int r = 10, int seed = -1) {
    const FloatType coeff = static_cast<FloatType>(10.0) * std::sqrt(static_cast<FloatType>(2.0) / static_cast<FloatType>(M_PI));
    FloatType max_norm = 0.0;

    auto gen = randla::random::RandomGenerator<FloatType>::make_generator(seed);

    std::normal_distribution<FloatType> dist(0.0, 1.0);
    for (int i = 0; i < r; ++i) {
        Vector omega(A.cols());
        for (int j = 0; j < A.cols(); ++j) omega(j) = dist(gen);
        Vector residual = A * omega - Q * (Q.transpose() * (A * omega));
        FloatType norm = residual.norm();
        if (norm > max_norm) max_norm = norm;
    }
    return coeff * max_norm;
}

static Scalar realError(const Matrix& A, const Matrix& Q) {
    Matrix R = A - Q * (Q.transpose() * A);
    return R.norm();
}

static Scalar realError(const Matrix& A, const CMatrix& Qc) {
    CMatrix Ac = A.template cast<Complex>();
    CMatrix R = Ac - Qc * (Qc.adjoint() * Ac);
    return R.norm();
}

static Scalar estimateSpectralNorm(const CMatrix & E, int seed, int power_steps = 6) {
    const int n = E.cols();

    CVector z = randla::random::RandomGenerator<FloatType>::randomComplexGaussianVector(n, seed);
    z.normalize();

    for (int i = 0; i < power_steps; ++i) {
        z = E.adjoint() * (E * z);
        z.normalize();
    }

    return (E * z).norm();
}
};

} // namespace randla::algorithms

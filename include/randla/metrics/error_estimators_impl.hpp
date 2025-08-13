#pragma once

#include <random>
#include <chrono>
#include <numbers>

namespace randla::algorithms {

// helper to create generator
namespace detail {
inline std::mt19937 make_generator(int seed) {
    if (seed < 0) {
        auto now = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        return std::mt19937(static_cast<unsigned int>(now));
    }
    return std::mt19937(static_cast<unsigned int>(seed));
}
} // namespace detail

template<typename FloatType>
typename ErrorEstimators<FloatType>::Scalar
ErrorEstimators<FloatType>::posteriorErrorEstimation(const Matrix& A, const Matrix& Q, int r, int seed) {
    const FloatType coeff = static_cast<FloatType>(10.0) * std::sqrt(static_cast<FloatType>(2.0) / static_cast<FloatType>(M_PI));
    FloatType max_norm = 0.0;

    auto gen = detail::make_generator(seed);

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

template<typename FloatType>
typename ErrorEstimators<FloatType>::Scalar
ErrorEstimators<FloatType>::realError(const Matrix& A, const Matrix& Q) {
    Matrix R = A - Q * (Q.transpose() * A);
    return R.norm();
}

template<typename FloatType>
typename ErrorEstimators<FloatType>::Scalar
ErrorEstimators<FloatType>::realError(const Matrix& A, const CMatrix& Qc) {
    CMatrix Ac = A.template cast<Complex>();
    CMatrix R = Ac - Qc * (Qc.adjoint() * Ac);
    return R.norm();
}

} // namespace randla::algorithms

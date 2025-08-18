#pragma once

#include <Eigen/Dense>
#include <random>
#include <cmath>
#include <randla/types.hpp>

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

    /**
     * @brief Posterior error estimation using Equation (4.3)
     */
    static Scalar posteriorErrorEstimation(const Matrix& A, const Matrix& Q, int r = 10, int seed = -1);

    /** Exact residual norm ||A - QQ^T A|| */
    static Scalar realError(const Matrix& A, const Matrix& Q);

    /** Overload for complex Q (||A - Q Q^* A||) */
    static Scalar realError(const Matrix& A, const CMatrix& Qc);

    static Scalar estimateSpectralNorm(const CMatrix & E, int seed, int power_steps = 6);
};

} // namespace randla::algorithms

#include "error_estimators_impl.hpp"

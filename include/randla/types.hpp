#pragma once

#include <Eigen/Dense>

namespace randla {

/**
 * @brief Common type aliases used throughout the library
 */
template<typename FloatType>
struct Types {
    using Scalar = FloatType;
    using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
};

} // namespace randla

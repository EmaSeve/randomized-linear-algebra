#pragma once

#include <Eigen/Dense>
#include <cmath>

namespace randla {

/**
 * @brief Common type aliases used throughout the library
 */
template<typename FloatType>
struct Types {
    using Scalar = FloatType;
    using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    
    /**
     * @brief Result structure for SVD decomposition A ≈ U S V^*
     */
    struct DirectSVDResult {
        Matrix U;  ///< Left singular vectors
        Vector S;  ///< Singular values
        Matrix V;  ///< Right singular vectors
    };
};

} // namespace randla

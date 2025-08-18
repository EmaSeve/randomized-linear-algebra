#pragma once

#include <Eigen/Dense>
#include <Eigen/SparseCore>
#include <cmath>
#include <complex>

namespace randla {

/**
 * @brief Common type aliases used throughout the library
 */
template<typename FloatType>
struct Types {
    using Scalar = FloatType;
    using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

    using SparseMatrix = Eigen::SparseMatrix<Scalar, Eigen::ColMajor>;
    using SparseVector = Eigen::SparseVector<Scalar>;

    using Complex  = std::complex<Scalar>;
    using CMatrix  = Eigen::Matrix<Complex, Eigen::Dynamic, Eigen::Dynamic>;
    using CVector  = Eigen::Matrix<Complex, Eigen::Dynamic, 1>;
    
    /**
     * @brief Result structure for SVD decomposition A ≈ U S V^*
     */
    struct SVDResult {
        Matrix U;  ///< Left singular vectors
        Vector S;  ///< Singular values
        Matrix V;  ///< Right singular vectors
    };

    struct IDResult {
        CMatrix B;
        CMatrix P;
        std::vector<int> indices; // columns of A used to construct B
    };

    struct EigenvalueDecomposition {
        Matrix U;
        Matrix Lambda;
    };
};

} // namespace randla

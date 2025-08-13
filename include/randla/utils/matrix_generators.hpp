#pragma once

#include <Eigen/Dense>
#include <type_traits>
#include "../types.hpp"

namespace randla::utils {

/**
 * @brief Utility class for generating synthetic matrices with specific properties
 *
 * Provides static methods to generate various types of matrices commonly used
 * when experimenting with randomized linear algebra algorithms: sparse random
 * matrices, matrices with controlled singular value decay, and matrices with
 * user-specified singular values.
 */
template<typename FloatType = double>
class MatrixGenerators : public randla::Types<FloatType> {
    static_assert(std::is_floating_point_v<FloatType>,
                  "FloatType must be a floating point type");

public:
    using typename randla::Types<FloatType>::Scalar;
    using typename randla::Types<FloatType>::Matrix;
    using typename randla::Types<FloatType>::Vector;
    using typename randla::Types<FloatType>::SparseMatrix;

    static SparseMatrix randomSparseMatrix(int rows, int cols, Scalar density, int seed = -1);
    static Matrix matrixWithExponentialDecay(int rows, int cols, Scalar decay_rate, int rank = -1, int seed = -1);
    static Matrix matrixWithSingularValues(int rows, int cols, const Vector& singular_values, int seed = -1);
    static Matrix lowRankPlusNoise(int rows, int cols, int rank, Scalar noise_level, int seed = -1);

};

} // namespace randla::utils

#include "matrix_generators_impl.hpp"
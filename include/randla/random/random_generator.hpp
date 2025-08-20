#pragma once

#include <random>
#include <randla/types.hpp>

namespace randla::random {

/**
 * @brief Utility class for generating random matrices and vectors
 */
template<typename FloatType = double>
class RandomGenerator : public randla::Types<FloatType> {
    static_assert(std::is_floating_point_v<FloatType>, 
                  "FloatType must be a floating point type");

public:
    using typename randla::Types<FloatType>::Scalar;
    using typename randla::Types<FloatType>::Matrix;
    using typename randla::Types<FloatType>::Vector;
    using typename randla::Types<FloatType>::Complex;
    using typename randla::Types<FloatType>::CMatrix;
    using typename randla::Types<FloatType>::CVector;

    static inline std::mt19937 make_generator(int seed) {
        if (seed >= 0) return std::mt19937(seed);
        return std::mt19937(std::chrono::steady_clock::now().time_since_epoch().count());
    }

    static Matrix randomGaussianMatrix(int rows, int cols, std::mt19937& gen) {
        Matrix result(rows, cols);
        std::normal_distribution<FloatType> dist(0.0, 1.0);
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                result(i, j) = dist(gen);
            }
        }
        return result;
    }

    static Matrix randomGaussianMatrix(int rows, int cols, int seed) {
        auto gen = make_generator(seed);
        return randomGaussianMatrix(rows, cols, gen);
    }

    static CMatrix randomComplexGaussianMatrix(int rows, int cols, std::mt19937& gen) {
        std::normal_distribution<FloatType> dist(0.0, 1.0);
        CMatrix result(rows, cols);
        
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                FloatType re = dist(gen);
                FloatType im = dist(gen);
                result(i, j) = Complex(re, im);
            }
        }
        return result;
    }

    static CMatrix randomComplexGaussianMatrix(int rows, int cols, int seed) {
        auto gen = make_generator(seed);
        return randomComplexGaussianMatrix(rows, cols, gen);
    }

    static Vector randomGaussianVector(int size, std::mt19937& gen) {
        Vector result(size);
        std::normal_distribution<FloatType> dist(0.0, 1.0);
        for (int i = 0; i < size; ++i) {
            result(i) = dist(gen);
        }
        return result;
    }

    static Vector randomGaussianVector(int size, int seed) {
        auto gen = make_generator(seed);
        return randomGaussianVector(size, gen);
    }

    static CVector randomComplexGaussianVector(int size, std::mt19937& gen) {
        CVector result(size);
        std::normal_distribution<FloatType> dist(0.0, 1.0);
        for (int i = 0; i < size; ++i) {
            FloatType re = dist(gen);
            FloatType im = dist(gen);
            result(i) = Complex(re, im);
        }
        return result;
    }

    static CVector randomComplexGaussianVector(int size, int seed) {
        auto gen = make_generator(seed);
        return randomComplexGaussianVector(size, gen);
    }
};

} // namespace randla::random

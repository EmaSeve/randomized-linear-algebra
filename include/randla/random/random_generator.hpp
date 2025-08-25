#pragma once

#include <random>
#include <randla/types.hpp>

namespace randla::random {

/**
 * @brief Utility class for generating random matrices and vectors.
 * 
 * Provides static methods to generate real and complex Gaussian random matrices and vectors,
 * with optional seeding for reproducibility.
 * 
 * @tparam FloatType Floating point type (e.g., float, double).
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

    /**
     * @brief Create a random number generator.
     * 
     * @param seed If non-negative, use as seed; otherwise, use current time.
     * @return std::mt19937 Random number generator.
     */
    static inline std::mt19937 make_generator(int seed) {
        if (seed >= 0) return std::mt19937(seed);
        return std::mt19937(std::chrono::steady_clock::now().time_since_epoch().count());
    }

    /**
     * @brief Generate a real Gaussian random matrix.
     * 
     * @param rows Number of rows.
     * @param cols Number of columns.
     * @param gen Random number generator.
     * @return Matrix of size (rows, cols) with i.i.d. N(0,1) entries.
     */
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

    /**
     * @brief Generate a real Gaussian random matrix with optional seeding.
     * 
     * @param rows Number of rows.
     * @param cols Number of columns.
     * @param seed Seed for random number generator.
     * @return Matrix of size (rows, cols) with i.i.d. N(0,1) entries.
     */
    static Matrix randomGaussianMatrix(int rows, int cols, int seed) {
        auto gen = make_generator(seed);
        return randomGaussianMatrix(rows, cols, gen);
    }

    /**
     * @brief Generate a complex Gaussian random matrix.
     * 
     * @param rows Number of rows.
     * @param cols Number of columns.
     * @param gen Random number generator.
     * @return Complex matrix of size (rows, cols) with i.i.d. N(0,1) real and imaginary parts.
     */
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

    /**
     * @brief Generate a complex Gaussian random matrix with optional seeding.
     * 
     * @param rows Number of rows.
     * @param cols Number of columns.
     * @param seed Seed for random number generator.
     * @return Complex matrix of size (rows, cols) with i.i.d. N(0,1) real and imaginary parts.
     */
    static CMatrix randomComplexGaussianMatrix(int rows, int cols, int seed) {
        auto gen = make_generator(seed);
        return randomComplexGaussianMatrix(rows, cols, gen);
    }

    /**
     * @brief Generate a real Gaussian random vector.
     * 
     * @param size Length of the vector.
     * @param gen Random number generator.
     * @return Vector of length 'size' with i.i.d. N(0,1) entries.
     */
    static Vector randomGaussianVector(int size, std::mt19937& gen) {
        Vector result(size);
        std::normal_distribution<FloatType> dist(0.0, 1.0);
        for (int i = 0; i < size; ++i) {
            result(i) = dist(gen);
        }
        return result;
    }

    /**
     * @brief Generate a real Gaussian random vector with optional seeding.
     * 
     * @param size Length of the vector.
     * @param seed Seed for random number generator.
     * @return Vector of length 'size' with i.i.d. N(0,1) entries.
     */
    static Vector randomGaussianVector(int size, int seed) {
        auto gen = make_generator(seed);
        return randomGaussianVector(size, gen);
    }

    /**
     * @brief Generate a complex Gaussian random vector.
     * 
     * @param size Length of the vector.
     * @param gen Random number generator.
     * @return Complex vector of length 'size' with i.i.d. N(0,1) real and imaginary parts.
     */
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

    /**
     * @brief Generate a complex Gaussian random vector with optional seeding.
     * 
     * @param size Length of the vector.
     * @param seed Seed for random number generator.
     * @return Complex vector of length 'size' with i.i.d. N(0,1) real and imaginary parts.
     */
    static CVector randomComplexGaussianVector(int size, int seed) {
        auto gen = make_generator(seed);
        return randomComplexGaussianVector(size, gen);
    }
};

} // namespace randla::random

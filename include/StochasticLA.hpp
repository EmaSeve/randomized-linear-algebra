#pragma once

#include <Eigen/Dense>
#include <type_traits>

namespace StochasticLA {

template<typename FloatType = double>
class RandomizedLinearAlgebra {
    static_assert(std::is_floating_point_v<FloatType>, 
                  "FloatType must be a floating point type");

public:
    using Scalar = FloatType;
    using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

    static Matrix randomMatrix(int rows, int cols, int seed = -1);
    
    // Algorithm 4.3: randomized power iteration
    static Matrix randomizedPowerIteration(const Matrix& A, int l, int q);
    
    // Algorithm 4.4: randomized subspace iteration
    static Matrix randomizedSubspaceIteration(const Matrix& A, int l, int q);
        
};

using RandomizedLinearAlgebraF = RandomizedLinearAlgebra<float>;
using RandomizedLinearAlgebraD = RandomizedLinearAlgebra<double>;
using RandomizedLinearAlgebraLD = RandomizedLinearAlgebra<long double>;

} // namespace StochasticLA

// Include implementation
#include "StochasticLA_impl.hpp"

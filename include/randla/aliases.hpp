#pragma once
#include <randla/algorithms/randomized_linear_algebra.hpp>
#include <randla/utils/test_matrices.hpp>

namespace randla {
using RandomizedLinearAlgebraF  = algorithms::RandomizedLinearAlgebra<float>;
using RandomizedLinearAlgebraD  = algorithms::RandomizedLinearAlgebra<double>;
using RandomizedLinearAlgebraLD = algorithms::RandomizedLinearAlgebra<long double>;

using TestMatricesF  = utils::TestMatrices<float>;
using TestMatricesD  = utils::TestMatrices<double>;
using TestMatricesLD = utils::TestMatrices<long double>;
}

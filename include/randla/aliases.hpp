#pragma once
#include <randla/algorithms/randomized_linear_algebra.hpp>
#include <randla/utils/matrix_generators.hpp>

namespace randla {
using RandomizedLinearAlgebraF  = algorithms::RandomizedLinearAlgebra<float>;
using RandomizedLinearAlgebraD  = algorithms::RandomizedLinearAlgebra<double>;
using RandomizedLinearAlgebraLD = algorithms::RandomizedLinearAlgebra<long double>;

using MatrixGeneratorsF  = utils::MatrixGenerators<float>;
using MatrixGeneratorsD  = utils::MatrixGenerators<double>;
using MatrixGeneratorsLD = utils::MatrixGenerators<long double>;
}

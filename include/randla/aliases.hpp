#pragma once
#include <randla/algorithms/randomized_range_finder.hpp>
#include <randla/utils/matrix_generators.hpp>

namespace randla {
using RandomizedRangeFinderF  = algorithms::RandomizedRangeFinder<float>;
using RandomizedRangeFinderD  = algorithms::RandomizedRangeFinder<double>;
using RandomizedRangeFinderLD = algorithms::RandomizedRangeFinder<long double>;

using MatrixGeneratorsF  = utils::MatrixGenerators<float>;
using MatrixGeneratorsD  = utils::MatrixGenerators<double>;
using MatrixGeneratorsLD = utils::MatrixGenerators<long double>;
}

#pragma once
#include <randla/algorithms/rand_range_finder.hpp>
#include <randla/utils/matrix_generators.hpp>

namespace randla {
using RandRangeFinderF  = algorithms::RandRangeFinder<float>;
using RandRangeFinderD  = algorithms::RandRangeFinder<double>;
using RandRangeFinderLD = algorithms::RandRangeFinder<long double>;

using MatrixGeneratorsF  = utils::MatrixGenerators<float>;
using MatrixGeneratorsD  = utils::MatrixGenerators<double>;
using MatrixGeneratorsLD = utils::MatrixGenerators<long double>;
}

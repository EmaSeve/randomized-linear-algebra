#pragma once
#include <randla/algorithms/rand_range_finder.hpp>
#include <randla/algorithms/adaptive_rand_range_finder.hpp>
#include <randla/algorithms/matrix_factorizer.hpp>
#include <randla/utils/matrix_generators.hpp>

namespace randla {
using RandRangeFinderF  = algorithms::RandRangeFinder<float>;
using RandRangeFinderD  = algorithms::RandRangeFinder<double>;
using RandRangeFinderLD = algorithms::RandRangeFinder<long double>;

using AdaptiveRandRangeFinderF  = algorithms::AdaptiveRandRangeFinder<float>;
using AdaptiveRandRangeFinderD  = algorithms::AdaptiveRandRangeFinder<double>;
using AdaptiveRandRangeFinderLD = algorithms::AdaptiveRandRangeFinder<long double>;

using MatrixFactorizerF  = algorithms::MatrixFactorizer<float>;
using MatrixFactorizerD  = algorithms::MatrixFactorizer<double>;
using MatrixFactorizerLD = algorithms::MatrixFactorizer<long double>;

using MatrixGeneratorsF  = utils::MatrixGenerators<float>;
using MatrixGeneratorsD  = utils::MatrixGenerators<double>;
using MatrixGeneratorsLD = utils::MatrixGenerators<long double>;
}

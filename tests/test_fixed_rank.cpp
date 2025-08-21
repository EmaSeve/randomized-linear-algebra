#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <randla/randla.hpp>
#include <gtest/gtest.h>

using RRF     = randla::RandRangeFinderD;
using TestMat = randla::MatrixGeneratorsD;
using Err     = randla::metrics::ErrorEstimators<double>;

namespace {
    const int m = 200, n = 100, l = 20, q = 2, seed = 123;
    const double tol = 1e-2;

    Eigen::MatrixXd A1 = TestMat::matrixWithExponentialDecay(m, n, 0.5, seed);
    Eigen::MatrixXd A2 = TestMat::matrixWithExponentialDecay(m, n, 0.1, seed);
}

// ---------------- DENSE ----------------
TEST(RRF_Dense, RandomizedRangeFinder) {
    for (auto& A : {A1, A2}) {
        auto Q = RRF::randomizedRangeFinder(A, l, seed + 1);
        double err = Err::realError(A, Q);
        EXPECT_LT(err, tol);
    }
}

TEST(RRF_Dense, RandomizedPowerIteration) {
    for (auto& A : {A1, A2}) {
        auto Q = RRF::randomizedPowerIteration(A, l, q, seed + 2);
        double err = Err::realError(A, Q);
        EXPECT_LT(err, tol);
    }
}

TEST(RRF_Dense, RandomizedSubspaceIteration) {
    for (auto& A : {A1, A2}) {
        auto Q = RRF::randomizedSubspaceIteration(A, l, q, seed + 3);
        double err = Err::realError(A, Q);
        EXPECT_LT(err, tol);
    }
}

TEST(RRF_Dense, FastRandRangeFinder) {
    for (auto& A : {A1, A2}) {
        auto Q = RRF::fastRandRangeFinder(A, l, seed + 4);
        double err = Err::realError(A, Q);
        EXPECT_LT(err, tol);
    }
}
// tests/test_fixed_rank_dense_simple.cpp
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <randla/randla.hpp>
#include <gtest/gtest.h>

using RRF     = randla::RandRangeFinderD;
using TestMat = randla::MatrixGeneratorsD;
using Err     = randla::metrics::ErrorEstimators<double>;

namespace {
    constexpr int m = 200, n = 100, l = 20, q = 2, seed = 123;
    constexpr int rank = l;

    constexpr double tol_exp_decay = 1e-3;
    constexpr double tol_lowrank   = 1e-8;

    Eigen::MatrixXd& A_exp_decay() {
        static Eigen::MatrixXd mat = TestMat::matrixWithExponentialDecay(m, n, 0.5, seed);
        return mat;
    }
    Eigen::MatrixXd& A_lowrank0() {
        static Eigen::MatrixXd mat = TestMat::lowRankPlusNoise(m, n, rank, /*noise=*/0.0, seed);
        return mat;
    }
}

// ================= EXPONENTIAL DECAY =================

TEST(RRF_ExpDecay, RandomizedRangeFinder) {
    auto Q = RRF::randomizedRangeFinder(A_exp_decay(), l, seed + 1);
    EXPECT_LE(Err::realError(A_exp_decay(), Q), tol_exp_decay);
}

TEST(RRF_ExpDecay, RandomizedPowerIteration) {
    auto Q = RRF::randomizedPowerIteration(A_exp_decay(), l, q, seed + 2);
    EXPECT_LE(Err::realError(A_exp_decay(), Q), tol_exp_decay);
}

TEST(RRF_ExpDecay, RandomizedSubspaceIteration) {
    auto Q = RRF::randomizedSubspaceIteration(A_exp_decay(), l, q, seed + 3);
    EXPECT_LE(Err::realError(A_exp_decay(), Q), tol_exp_decay);
}

TEST(RRF_ExpDecay, FastRandRangeFinder) {
    auto Q = RRF::fastRandRangeFinder(A_exp_decay(), l, seed + 4);
    EXPECT_LE(Err::realError(A_exp_decay(), Q), tol_exp_decay);
}

// ================= LOW-RANK (NOISE 0) =================

TEST(RRF_LowRank0, RandomizedRangeFinder) {
    auto Q = RRF::randomizedRangeFinder(A_lowrank0(), l, seed + 1);
    EXPECT_LE(Err::realError(A_lowrank0(), Q), tol_lowrank);
}

TEST(RRF_LowRank0, RandomizedPowerIteration) {
    auto Q = RRF::randomizedPowerIteration(A_lowrank0(), l, q, seed + 2);
    EXPECT_LE(Err::realError(A_lowrank0(), Q), tol_lowrank);
}

TEST(RRF_LowRank0, RandomizedSubspaceIteration) {
    auto Q = RRF::randomizedSubspaceIteration(A_lowrank0(), l, q, seed + 3);
    EXPECT_LE(Err::realError(A_lowrank0(), Q), tol_lowrank);
}

TEST(RRF_LowRank0, FastRandRangeFinder) {
    auto Q = RRF::fastRandRangeFinder(A_lowrank0(), l, seed + 4);
    EXPECT_LE(Err::realError(A_lowrank0(), Q), tol_lowrank);
}

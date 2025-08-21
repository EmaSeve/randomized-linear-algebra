// tests/test_adaptive_fixed_precision_simple.cpp
#include <Eigen/Dense>
#include <randla/randla.hpp>
#include <gtest/gtest.h>

using ARRF = randla::AdaptiveRandRangeFinderD;
using Err  = randla::metrics::ErrorEstimators<double>;
using GM   = randla::utils::MatrixGenerators<double>;

namespace {
    constexpr int rows = 800, cols = 400, rank = 20;
    constexpr int r = 10, pwr = 2, seed = 42;
    constexpr int l_srft = 1;
    constexpr double growth_factor = 2.0;

    // tolleranze
    constexpr double tol_exact          = 1e-10;
    constexpr double tol_decay          = 1e-3;
    constexpr double tol_lowrank0       = 1e-10;
    constexpr double tol_lowrank_noise  = 5e-3;

    // Precomputiamo tutte le matrici
    Eigen::MatrixXd make_exact() {
        Eigen::VectorXd sv = Eigen::VectorXd::Zero(std::min(rows, cols));
        for (int i = 0; i < rank; ++i) sv(i) = 1.0;
        return GM::matrixWithSingularValues(rows, cols, sv, seed);
    }

    Eigen::MatrixXd make_decay() {
        double decay_rate = 0.5;
        return GM::matrixWithExponentialDecay(rows, cols, decay_rate, rank, seed);
    }

    Eigen::MatrixXd make_lowrank0() {
        return GM::lowRankPlusNoise(rows, cols, rank, 0.0, seed);
    }

    Eigen::MatrixXd make_lowrank_noise() {
        double sigma = 1e-3;
        return GM::lowRankPlusNoise(rows, cols, rank, sigma, seed);
    }

    const Eigen::MatrixXd A_exact         = make_exact();
    const Eigen::MatrixXd A_decay         = make_decay();
    const Eigen::MatrixXd A_lowrank0      = make_lowrank0();
    const Eigen::MatrixXd A_lowrank_noise = make_lowrank_noise();
}

// ===================== EXACT RANK =====================
TEST(AdaptiveFP_Exact, ARF) {
    auto Q = ARRF::adaptiveRangeFinder(A_exact, tol_exact, r, seed);
    EXPECT_LE(Err::realError(A_exact, Q), tol_exact);
}
TEST(AdaptiveFP_Exact, API) {
    auto Q = ARRF::adaptivePowerIteration(A_exact, tol_exact, r, pwr, seed);
    EXPECT_LE(Err::realError(A_exact, Q), tol_exact);
}
TEST(AdaptiveFP_Exact, SRFT) {
    auto Q = ARRF::adaptiveFastRandRangeFinder(A_exact, tol_exact, l_srft, seed, growth_factor);
    EXPECT_LE(Err::realError(A_exact, Q), tol_exact);
}

// ===================== EXPONENTIAL DECAY =====================
TEST(AdaptiveFP_ExpDecay, ARF) {
    auto Q = ARRF::adaptiveRangeFinder(A_decay, tol_decay, r, seed);
    EXPECT_LE(Err::realError(A_decay, Q), tol_decay);
}

TEST(AdaptiveFP_ExpDecay, API) {
    constexpr int r_expdecay   = r; 
    constexpr int pwr_expdecay = pwr; 

    auto Q = ARRF::adaptivePowerIteration(A_decay, tol_decay, r_expdecay, pwr_expdecay, seed);
    EXPECT_LE(Err::realError(A_decay, Q), tol_decay);
}

TEST(AdaptiveFP_ExpDecay, SRFT) {
    auto Q = ARRF::adaptiveFastRandRangeFinder(A_decay, tol_decay, l_srft, seed, growth_factor);
    EXPECT_LE(Err::realError(A_decay, Q), tol_decay);
}

// ===================== LOW-RANK (NOISE 0) =====================
TEST(AdaptiveFP_LowRank0, ARF) {
    auto Q = ARRF::adaptiveRangeFinder(A_lowrank0, tol_lowrank0, r, seed);
    EXPECT_LE(Err::realError(A_lowrank0, Q), tol_lowrank0);
}
TEST(AdaptiveFP_LowRank0, API) {
    auto Q = ARRF::adaptivePowerIteration(A_lowrank0, tol_lowrank0, r, pwr, seed);
    EXPECT_LE(Err::realError(A_lowrank0, Q), tol_lowrank0);
}
TEST(AdaptiveFP_LowRank0, SRFT) {
    auto Q = ARRF::adaptiveFastRandRangeFinder(A_lowrank0, tol_lowrank0, l_srft, seed, growth_factor);
    EXPECT_LE(Err::realError(A_lowrank0, Q), tol_lowrank0);
}

// ===================== LOW-RANK + NOISE =====================
TEST(AdaptiveFP_LowRankNoise, ARF) {
    auto Q = ARRF::adaptiveRangeFinder(A_lowrank_noise, tol_lowrank_noise, r, seed);
    EXPECT_LE(Err::realError(A_lowrank_noise, Q), tol_lowrank_noise);
}
TEST(AdaptiveFP_LowRankNoise, API) {
    auto Q = ARRF::adaptivePowerIteration(A_lowrank_noise, tol_lowrank_noise, r, pwr, seed);
    EXPECT_LE(Err::realError(A_lowrank_noise, Q), tol_lowrank_noise);
}
TEST(AdaptiveFP_LowRankNoise, SRFT) {
    auto Q = ARRF::adaptiveFastRandRangeFinder(A_lowrank_noise, tol_lowrank_noise, l_srft, seed, growth_factor);
    EXPECT_LE(Err::realError(A_lowrank_noise, Q), tol_lowrank_noise);
}

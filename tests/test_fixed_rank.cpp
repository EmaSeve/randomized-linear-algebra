#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <randla/randla.hpp>
#include <gtest/gtest.h>
#include <vector>

using RRF     = randla::RandRangeFinderD;
using TestMat = randla::MatrixGeneratorsD;
using Err     = randla::metrics::ErrorEstimators<double>;

namespace {
    const int m = 200, n = 100, l = 20, q = 2, seed = 123;
    const int rank = l;

    constexpr double tol_exp_decay = 1e-3;
    constexpr double tol_lowrank   = 1e-8;

    struct Case {
        Eigen::MatrixXd A;
        double tol;
    };

    std::vector<Case> cases = {
        { TestMat::matrixWithExponentialDecay(m, n, 0.5, seed),       tol_exp_decay },
        { TestMat::lowRankPlusNoise(m, n, rank, /*noise=*/0.0, seed), tol_lowrank   }
    };
}

template <typename Builder>
static void run_case(const Builder& buildQ) {
    for (const auto& c : cases) {
        auto Q   = buildQ(c.A);
        double e = Err::realError(c.A, Q);
        EXPECT_LT(e, c.tol);
    }
}

// ---------------- DENSE ----------------
TEST(RRF_Dense, RandomizedRangeFinder) {
    run_case([&](const Eigen::MatrixXd& A){
        return RRF::randomizedRangeFinder(A, l, seed + 1);
    });
}

TEST(RRF_Dense, RandomizedPowerIteration) {
    run_case([&](const Eigen::MatrixXd& A){
        return RRF::randomizedPowerIteration(A, l, q, seed + 2);
    });
}

TEST(RRF_Dense, RandomizedSubspaceIteration) {
    run_case([&](const Eigen::MatrixXd& A){
        return RRF::randomizedSubspaceIteration(A, l, q, seed + 3);
    });
}

TEST(RRF_Dense, FastRandRangeFinder) {
    run_case([&](const Eigen::MatrixXd& A){
        return RRF::fastRandRangeFinder(A, l, seed + 4);
    });
}

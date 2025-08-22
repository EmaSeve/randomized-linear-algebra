// tests/test_stageB_factorizations.cpp
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <randla/randla.hpp>
#include <gtest/gtest.h>

using RRF     = randla::RandRangeFinderD;
using ARRF    = randla::AdaptiveRandRangeFinderD;
using TestMat = randla::MatrixGeneratorsD;
using Err     = randla::metrics::ErrorEstimators<double>;
using MF      = randla::MatrixFactorizerD;

using Matrix   = randla::Types<double>::Matrix;
using SVDResult  = randla::Types<double>::SVDResult;
using EigResult  = randla::Types<double>::EigenvalueDecomposition;

namespace {
constexpr int rows = 100;
constexpr int cols = 50;
constexpr int size = 70;   // per matrici quadrate
constexpr int rank = 30;    // target rank per Q
constexpr int q    = 2;    // power iters
constexpr int seed = 42;

constexpr double tol_stageA   = 1e-4;   // tolleranza per costruire Q
constexpr double tol_factor   = 5e-2;   // tolleranza per ricostruzione Stage B
constexpr double tol_factor_loose = 1e-1;

// Helpers di ricostruzione
Matrix reconstruct(const SVDResult& svd) {
    return svd.U * svd.S.asDiagonal() * svd.V.transpose();
}
Matrix reconstruct(const EigResult& ed) {
    return ed.U * ed.Lambda * ed.U.transpose();
}
double rel_err(const Matrix& A, const Matrix& Ahat) {
    return (A - Ahat).norm() / std::max(1e-16, A.norm());
}

// Fixture che costruisce varie A e i rispettivi Q
struct StageBFixture : ::testing::Test {
    Matrix A_psd, A_lowrank, A_lowrank_noise, A_sv, A_expdecay, A_dense;
    Matrix Q_psd, Q_psd_rf, Q_lowrank, Q_lowrank_rf, Q_sv, Q_exp_rf, Q_dense_rf, Q_dense_pow;

    void SetUp() override {
        randla::threading::setThreads(1);
        // eigen thread
        Eigen::setNbThreads(1);

        A_psd           = TestMat::randomPositiveSemidefiniteMatrix(size);
        A_lowrank       = TestMat::lowRankPlusNoise(rows, cols, rank, 0.0, seed);
        A_lowrank_noise = TestMat::lowRankPlusNoise(rows, cols, rank, 0.5, seed);

        Eigen::VectorXd sv = Eigen::VectorXd::Zero(std::min(rows, cols));
        for (int i = 0; i < rank && i < sv.size(); ++i) sv(i) = 1.0;
        A_sv              = TestMat::matrixWithSingularValues(rows, cols, sv);
        A_expdecay        = TestMat::matrixWithExponentialDecay(rows, cols, 0.5, rank);
        A_dense           = TestMat::randomDenseMatrix(rows, cols);

        // Basis Q via Stage A (due varianti per alcuni casi)
        Q_psd       = ARRF::adaptivePowerIteration(A_psd, tol_stageA, rank, q, -1);
        Q_psd_rf    = ARRF::adaptiveRangeFinder(A_psd, tol_stageA, rank, -1);

        Q_lowrank   = ARRF::adaptivePowerIteration(A_lowrank, tol_stageA, rank, q, -1);
        Q_lowrank_rf= ARRF::adaptiveRangeFinder(A_lowrank, tol_stageA, rank, -1);

        Q_sv        = ARRF::adaptivePowerIteration(A_sv, tol_stageA, rank, q, -1);
        Q_exp_rf       = ARRF::adaptiveRangeFinder(A_expdecay, tol_stageA, rank, -1);

        Q_dense_rf  = ARRF::adaptiveRangeFinder(A_dense, tol_stageA, rank, -1);
        Q_dense_pow = ARRF::adaptivePowerIteration(A_dense, tol_stageA, rank, q, -1);
    }
};
} // namespace

// ---------- Sanity check di Stage A: ||A - QQ^T A|| ----------
TEST_F(StageBFixture, StageA_ProjectionErrorsAreSmall) {
    EXPECT_LE(Err::realError(A_psd,           Q_psd),        5e-2);
    EXPECT_LE(Err::realError(A_psd,           Q_psd_rf),     5e-2);
    EXPECT_LE(Err::realError(A_lowrank,       Q_lowrank),    1e-8);
    EXPECT_LE(Err::realError(A_lowrank,       Q_lowrank_rf), 1e-8);
    EXPECT_LE(Err::realError(A_sv,            Q_sv),         1e-6);
    EXPECT_LE(Err::realError(A_expdecay,      Q_exp_rf),     1e-2);
    EXPECT_LE(Err::realError(A_dense,         Q_dense_rf),   5e-2);
    EXPECT_LE(Err::realError(A_dense,         Q_dense_pow),  5e-2);  
}

// ---------- SVD: direct ----------
TEST_F(StageBFixture, DirectSVD_OnPSD_WithQpi) {
    SVDResult svd;
    ASSERT_NO_THROW(svd = MF::directSVD(A_psd, Q_psd, tol_stageA))   ;
    const double e = rel_err(A_psd, reconstruct(svd));
    EXPECT_LE(e, tol_factor);
}
TEST_F(StageBFixture, DirectSVD_OnLowRank_WithQrf) {
    SVDResult svd;
    ASSERT_NO_THROW(svd = MF::directSVD(A_lowrank, Q_lowrank_rf, tol_stageA));
    const double e = rel_err(A_lowrank, reconstruct(svd));
    EXPECT_LE(e, 1e-8);
}
TEST_F(StageBFixture, DirectSVD_OnDense_WithQrf) {
    SVDResult svd;
    ASSERT_NO_THROW(svd = MF::directSVD(A_dense, Q_dense_rf, tol_stageA));
    EXPECT_LE(rel_err(A_dense, reconstruct(svd)), tol_factor);
}

// ---------- SVD: via row extraction ----------
TEST_F(StageBFixture, SVD_ViaRowExtraction_OnPSD) {
    SVDResult svd;
    ASSERT_NO_THROW(svd = MF::SVDViaRowExtraction(A_psd, Q_psd, tol_stageA));
    EXPECT_LE(rel_err(A_psd, reconstruct(svd)), tol_factor);
}
TEST_F(StageBFixture, SVD_ViaRowExtraction_OnLowRank) {
    SVDResult svd;
    ASSERT_NO_THROW(svd = MF::SVDViaRowExtraction(A_lowrank, Q_lowrank, tol_stageA));
    EXPECT_LE(rel_err(A_lowrank, reconstruct(svd)), 1e-8);
}
TEST_F(StageBFixture, SVD_ViaRowExtraction_OnDense) {
    SVDResult svd;
    ASSERT_NO_THROW(svd = MF::SVDViaRowExtraction(A_dense, Q_dense_pow, 5.0));
    EXPECT_LE(rel_err(A_dense, reconstruct(svd)), tol_factor_loose);
}

// ---------- Eigen-decomposition: direct (simmetrica) ----------
TEST_F(StageBFixture, Eig_Direct_OnPSD) {
    EigResult ed;
    ASSERT_NO_THROW(ed = MF::directEigenvalueDecomposition(A_psd, Q_psd, tol_stageA));
    EXPECT_LE(rel_err(A_psd, reconstruct(ed)), tol_factor);
}

// ---------- Eigen-decomposition: via row extraction ----------
TEST_F(StageBFixture, Eig_ViaRowExtraction_OnPSD) {
    EigResult ed;
    ASSERT_NO_THROW(ed = MF::eigenvalueDecompositionViaRowExtraction(A_psd, Q_psd, tol_stageA));
    EXPECT_LE(rel_err(A_psd, reconstruct(ed)), tol_factor);
}

// ---------- Nystrom (SPD) ----------
TEST_F(StageBFixture, Eig_Nystrom_OnPSD) {
    EigResult ed;
    ASSERT_NO_THROW(ed = MF::eigenvalueDecompositionViaNystromMethod(A_psd, Q_psd, tol_stageA));
    EXPECT_LE(rel_err(A_psd, reconstruct(ed)), tol_factor_loose);
}

// ---------- One-pass (richiede Omega) ----------
TEST_F(StageBFixture, Eig_OnePass_OnPSD) {
    // Omega con stesso # colonne di Q
    Matrix Omega = TestMat::randomDenseMatrix(A_psd.cols(), Q_psd.cols());
    EigResult ed;
    ASSERT_NO_THROW(ed = MF::eigenvalueDecompositionInOnePass(A_psd, Q_psd, Omega, tol_stageA));
    EXPECT_LE(rel_err(A_psd, reconstruct(ed)), tol_factor_loose);
}
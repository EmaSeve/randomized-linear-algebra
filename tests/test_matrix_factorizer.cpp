
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
using CMatrix  = randla::Types<double>::CMatrix;
using SVDResult = randla::Types<double>::SVDResult;
using IDResult  = randla::Types<double>::IDResult;
using EigenvalueDecomposition = randla::Types<double>::EigenvalueDecomposition;

void check_approximation_error(const CMatrix & A, const IDResult & id, double r_t, double tol_fro = 1e-8) {
   const auto& B = id.B;
   const auto& P = id.P;
   const auto residual = (A - B * P);
   const double frobenius_error = residual.norm() / A.norm();
   EXPECT_LE(frobenius_error, tol_fro) << "Frobenius error too high for rank/tol " << r_t;
}

int numerical_rank(const Matrix& Q, double tol = -1){
   Eigen::JacobiSVD<Matrix> svd(Q);
   auto singularValues = svd.singularValues();
   double threshold = tol;
   if (threshold < 0) {
      threshold = std::max(Q.rows(), Q.cols()) * singularValues.array().abs().maxCoeff() * std::numeric_limits<double>::epsilon();
   }
   return (singularValues.array() > threshold).count();
}

// GoogleTest: Each test is a TEST or TEST_F

namespace {
constexpr int rows = 150;
constexpr int cols = 140;
constexpr int size = 150;
constexpr double density = 0.9;
constexpr int seed = 42;
constexpr double tol = 0.2;
constexpr int q = 2;
constexpr int l = 140;
constexpr int rank = 140;

Matrix sparse_A() {
   return TestMat::randomSparseMatrix(rows,cols,density);
}
Matrix hermitian_A() {
   return TestMat::randomHermitianMatrix(size);
}
Matrix psd_A() {
   return TestMat::randomPositiveSemidefiniteMatrix(size);
}
Matrix dense_A() {
   return TestMat::randomDenseMatrix(rows, cols);
}
Matrix singularValues_A() {
   Eigen::VectorXd sv = Eigen::VectorXd::Zero(std::min(rows, cols));
   for (int i = 0; i < rank; ++i) sv(i) = 1.0;
   return TestMat::matrixWithSingularValues(rows, cols, sv);
}

Matrix Q_psd() {
   return ARRF::adaptivePowerIteration(psd_A(), tol, rank, q, -1);
}
Matrix Q_psd2() {
   return ARRF::adaptiveRangeFinder(psd_A(), tol, rank, -1);
}
Matrix Q_singularValues() {
   return ARRF::adaptivePowerIteration(singularValues_A(), tol, rank, q, -1);
}
Matrix Q_hermitian() {
   return ARRF::adaptivePowerIteration(hermitian_A(), tol, rank, q, -1);
}
Matrix Q_dense() {
   return ARRF::adaptiveRangeFinder(dense_A(), tol, rank, -1);
}
Matrix Q_sparse() {
   return RRF::randomizedRangeFinder(sparse_A(), rank, -1);
}
}



TEST(MatrixFactorizer, IDFactorization_DenseQ) {
   int ID_oversampling = 10;
   int rank_Q_dense = numerical_rank(Q_dense());
   ASSERT_GT(rank_Q_dense, ID_oversampling);
   EXPECT_NO_THROW({
      IDResult id = MF::IDFactorization(Q_dense(), rank_Q_dense - ID_oversampling, seed);
      // check_approximation_error(Q_dense(), id, rank_Q_dense - ID_oversampling, 1e-8);
   });
}

TEST(MatrixFactorizer, IDFactorization_HermitianQ) {
   int ID_oversampling = 10;
   int rank_hermitian_Q = numerical_rank(Q_hermitian());
   ASSERT_GT(rank_hermitian_Q, ID_oversampling);
   EXPECT_NO_THROW({
      IDResult id = MF::IDFactorization(Q_hermitian(), rank_hermitian_Q - ID_oversampling, seed);
      // check_approximation_error(Q_hermitian(), id, rank_hermitian_Q - ID_oversampling, 1e-8);
   });
}

TEST(MatrixFactorizer, IDFactorization_PSDQ) {
   int ID_oversampling = 10;
   int rank_psd_Q = numerical_rank(Q_psd());
   ASSERT_GT(rank_psd_Q, ID_oversampling);
   EXPECT_NO_THROW({
      IDResult id = MF::IDFactorization(Q_psd(), rank_psd_Q - ID_oversampling, seed);
      // check_approximation_error(Q_psd(), id, rank_psd_Q - ID_oversampling, 1e-8);
   });
}

TEST(MatrixFactorizer, AdaptiveIDFactorization_Qdense) {
   double id_tol = 0.6;
   EXPECT_NO_THROW({
      IDResult id = MF::adaptiveIDFactorization(Q_dense(), seed);
      // check_approximation_error(Q_dense(), id, id_tol, 1.0); // looser tol
   });
}

TEST(MatrixFactorizer, AdaptiveIDFactorization_Qpsd) {
   double id_tol = 0.6;
   EXPECT_NO_THROW({
      IDResult id = MF::adaptiveIDFactorization(Q_psd(), seed);
      // check_approximation_error(Q_psd(), id, id_tol, 1.0);
   });
}

TEST(MatrixFactorizer, AdaptiveIDFactorization_Qpsd2) {
   double id_tol = 0.6;
   EXPECT_NO_THROW({
      IDResult id = MF::adaptiveIDFactorization(Q_psd2(), seed);
      // check_approximation_error(Q_psd2(), id, id_tol, 1.0);
   });
}

TEST(MatrixFactorizer, AdaptiveIDFactorization_Qsparse) {
   double id_tol = 0.6;
   EXPECT_NO_THROW({
      IDResult id = MF::adaptiveIDFactorization(Q_sparse(), seed);
      // check_approximation_error(Q_sparse(), id, id_tol, 1.0);
   });
}

TEST(MatrixFactorizer, AdaptiveIDFactorization_Qhermitian) {
   double id_tol = 0.6;
   EXPECT_NO_THROW({
      IDResult id = MF::adaptiveIDFactorization(Q_hermitian(), seed);
      // check_approximation_error(Q_hermitian(), id, id_tol, 1.0);
   });
}


TEST(MatrixFactorizer, DirectSVD_SparseA_Qsparse) {
   EXPECT_NO_THROW({
      SVDResult svd_A = MF::directSVD(sparse_A(), Q_sparse(), tol);
      auto A_approx = svd_A.U * svd_A.S.asDiagonal() * svd_A.V.transpose();
      double real_err = (sparse_A() - A_approx).norm();
      EXPECT_LE(real_err, 1.0); // loose bound
   });
}


TEST(MatrixFactorizer, SVDViaRowExtraction_SparseA_Qsparse) {
   EXPECT_NO_THROW({
      SVDResult svd_A = MF::SVDViaRowExtraction(sparse_A(), Q_sparse(), tol);
      auto A_approx = svd_A.U * svd_A.S.asDiagonal() * svd_A.V.transpose();
      double real_err = (sparse_A() - A_approx).norm();
      EXPECT_LE(real_err, 1.0);
   });
}


TEST(MatrixFactorizer, DirectEigenvalueDecomposition_HermitianA_Qhermitian) {
   EXPECT_NO_THROW({
      EigenvalueDecomposition ed = MF::directEigenvalueDecomposition(hermitian_A(), Q_hermitian(), tol);
      auto Hermitian_A_approx = ed.U * ed.Lambda * ed.U.transpose();
      double real_err = (hermitian_A() - Hermitian_A_approx).norm();
      EXPECT_LE(real_err, 1.0);
   });
}


TEST(MatrixFactorizer, EigenvalueDecompositionViaRowExtraction_HermitianA_Qhermitian) {
   EXPECT_NO_THROW({
      EigenvalueDecomposition ed = MF::eigenvalueDecompositionViaRowExtraction(hermitian_A(), Q_hermitian(), tol);
      auto Hermitian_A_approx = ed.U * ed.Lambda * ed.U.transpose();
      double real_err = (hermitian_A() - Hermitian_A_approx).norm();
      EXPECT_LE(real_err, 1.0);
   });
}


TEST(MatrixFactorizer, EigenvalueDecompositionViaNystromMethod_PsdA_Qpsd) {
   EXPECT_NO_THROW({
      EigenvalueDecomposition ed = MF::eigenvalueDecompositionViaNystromMethod(psd_A(), Q_psd(), tol);
      auto spd_A_approx = ed.U * ed.Lambda * ed.U.transpose();
      double real_err = (psd_A() - spd_A_approx).norm();
      EXPECT_LE(real_err, 1.0);
   });
}


TEST(MatrixFactorizer, EigenvalueDecompositionInOnePass_HermitianA_Qhermitian) {
   EXPECT_NO_THROW({
      Matrix Omega = TestMat::randomDenseMatrix(size, size);
      EigenvalueDecomposition ed = MF::eigenvalueDecompositionInOnePass(hermitian_A(), Q_hermitian(), Omega, tol);
      auto Hermitian_A_approx = ed.U * ed.Lambda * ed.U.transpose();
      double real_err = (hermitian_A() - Hermitian_A_approx).norm();
      EXPECT_LE(real_err, 1.0);
   });
}



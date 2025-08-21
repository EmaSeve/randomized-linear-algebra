#include <cassert>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <randla/randla.hpp>

using RRF     = randla::RandRangeFinderD;
using TestMat = randla::MatrixGeneratorsD;
using Err     = randla::metrics::ErrorEstimators<double>;

static void checkDense(const Eigen::MatrixXd& A, int l, int q, int seed, double tol) {
    auto Q_range    = RRF::randomizedRangeFinder(A, l, seed+1);
    auto Q_power    = RRF::randomizedPowerIteration(A, l, q, seed+2);
    auto Q_subspace = RRF::randomizedSubspaceIteration(A, l, q, seed+3);
    auto Q_fast     = RRF::fastRandRangeFinder(A, l, seed+4);

    assert(Err::realError(A, Q_range)    > tol);
    assert(Err::realError(A, Q_power)    < tol);
    assert(Err::realError(A, Q_subspace) < tol);
    assert(Err::realError(A, Q_fast)     < tol);
}

static void checkSparse(const Eigen::SparseMatrix<double>& A, int l, int q, int seed, double tol) {
    auto Q_range    = RRF::randomizedRangeFinder(A, l, seed+1);
    auto Q_power    = RRF::randomizedPowerIteration(A, l, q, seed+2);
    auto Q_subspace = RRF::randomizedSubspaceIteration(A, l, q, seed+3);

    assert(Err::realError(A, Q_range)    < tol);
    assert(Err::realError(A, Q_power)    < tol);
    assert(Err::realError(A, Q_subspace) < tol);
}

int main() {
    randla::threading::setThreads(1);

    const int m = 200, n = 100; // più piccolo per test
    const int seed = 123;
    const int l = 20, q = 2;
    const double tol = 1e-8;

    // Dense tests
    checkDense(TestMat::matrixWithExponentialDecay(m, n, 0.5, seed), l, q, seed, tol);
    checkDense(TestMat::matrixWithExponentialDecay(m, n, 0.1, seed), l, q, seed, tol);

    // Sparse test
    auto A_sparse = TestMat::randomSparseMatrix(m, n, 0.05, seed+10);
    checkSparse(A_sparse, 30, q, seed+10, tol);

    // Sparse as dense
    Eigen::MatrixXd A_dense(A_sparse);
    checkDense(A_dense, 30, q, seed+10, tol);

    return EXIT_SUCCESS;
}

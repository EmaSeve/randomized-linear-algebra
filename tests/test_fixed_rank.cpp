#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <random>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <randla/randla.hpp>

using RRF     = randla::RandRangeFinderD;
using TestMat = randla::MatrixGeneratorsD;
using Err     = randla::metrics::ErrorEstimators<double>;

// ===== DENSE =====
static void runAlgorithmsDense(const std::string& label,
                                const Eigen::MatrixXd& A,
                                int l, int q, int seed) {
    std::cout << "\n=== " << label << " ===\n";
    std::cout << "Shape: " << A.rows() << " x " << A.cols() << "\n";
    std::cout << "Norm(A)=" << A.norm() << "\n";

    auto tic = [](){return std::chrono::high_resolution_clock::now();};
    auto ms  = [](auto start, auto end){return std::chrono::duration<double, std::milli>(end-start).count();};

    auto t0 = tic();
    auto Q_range = RRF::RandRangeFinder(A, l, seed + 1);
    auto t1 = tic();
    auto Q_power = RRF::randomizedPowerIteration(A, l, q, seed + 2);
    auto t2 = tic();
    auto Q_subspace = RRF::randomizedSubspaceIteration(A, l, q, seed + 3);
    auto t3 = tic();
    auto Q_fast = RRF::fastRandRangeFinder(A, l, seed + 4);
    auto t4 = tic();

    std::cout << "[RRF]  cols=" << Q_range.cols()
              << " err=" << Err::realError(A, Q_range)
              << " time_ms=" << ms(t0,t1) << "\n";

    std::cout << "[RPI q=" << q << "] cols=" << Q_power.cols()
              << " err=" << Err::realError(A, Q_power)
              << " time_ms=" << ms(t1,t2) << "\n";

    std::cout << "[RSI q=" << q << "] cols=" << Q_subspace.cols()
              << " err=" << Err::realError(A, Q_subspace)
              << " time_ms=" << ms(t2,t3) << "\n";

    std::cout << "[FAST] cols=" << Q_fast.cols()
              << " err=" << Err::realError(A, Q_fast)
              << " time_ms=" << ms(t3,t4) << "\n";
}

// ===== SPARSE =====
static void runAlgorithmsSparse(const std::string& label,
                                 const Eigen::SparseMatrix<double>& A,
                                 int l, int q, int seed) {
    std::cout << "\n=== " << label << " ===\n";
    std::cout << "Shape: " << A.rows() << " x " << A.cols() << "\n";
    std::cout << "Norm(A)=" << A.norm() << "\n";

    auto tic = [](){return std::chrono::high_resolution_clock::now();};
    auto ms  = [](auto start, auto end){return std::chrono::duration<double, std::milli>(end-start).count();};

    auto t0 = tic();
    auto Q_range = RRF::RandRangeFinder(A, l, seed + 1);
    auto t1 = tic();
    auto Q_power = RRF::randomizedPowerIteration(A, l, q, seed + 2);
    auto t2 = tic();
    auto Q_subspace = RRF::randomizedSubspaceIteration(A, l, q, seed + 3);
    auto t3 = tic();

    std::cout << "[RRF]  cols=" << Q_range.cols()
              << " err=" << Err::realError(A, Q_range)
              << " time_ms=" << ms(t0,t1) << "\n";

    std::cout << "[RPI q=" << q << "] cols=" << Q_power.cols()
              << " err=" << Err::realError(A, Q_power)
              << " time_ms=" << ms(t1,t2) << "\n";

    std::cout << "[RSI q=" << q << "] cols=" << Q_subspace.cols()
              << " err=" << Err::realError(A, Q_subspace)
              << " time_ms=" << ms(t2,t3) << "\n";

    // niente FAST qui
}

int main() {
    try {
        std::cout << "Eigen nbThreads = " << Eigen::nbThreads() << "\n";
        std::cout << std::fixed << std::setprecision(6);
        const int m = 800, n = 400;
        const int seed = 123;

        // DENSE tests
        runAlgorithmsDense("Exponential decay (rate=0.5)",
            TestMat::matrixWithExponentialDecay(m, n, 0.5, seed),
            10, 2, seed);

        runAlgorithmsDense("Exponential decay (rate=0.1)",
            TestMat::matrixWithExponentialDecay(m, n, 0.1, seed),
            10, 2, seed);

        // SPARSE test
        auto A_sparse = TestMat::randomSparseMatrix(m, n, 0.05, seed + 10);
        runAlgorithmsSparse("Random sparse (dens=0.05)", A_sparse, 100, 2, seed + 10);

        // Sparse -> Dense (stessa matrice, ma testata come densa)
        Eigen::MatrixXd A_sparse_as_dense(A_sparse);
        runAlgorithmsDense("Random sparse as dense (dens=0.05)", A_sparse_as_dense, 100, 2, seed + 10);

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}

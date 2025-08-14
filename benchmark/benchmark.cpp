#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <random>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <randla/randla.hpp>
#include <randla/threading.hpp

using RLA     = randla::RandomizedRangeFinderD;
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
    auto Q_range = RLA::randomizedRangeFinder(A, l, seed + 1);
    auto t1 = tic();

    std::cout << "[RRF]  cols=" << Q_range.cols()
              << " err=" << Err::realError(A, Q_range)
              << " time_ms=" << ms(t0,t1) << "\n";
}

int main() {
    try {
        randla::threading::setThreads(8);
        std::cout << "Eigen nbThreads = " << Eigen::nbThreads() << "\n";
        std::cout << std::fixed << std::setprecision(6);
        const int m = 5000, n = 800, rank = 400;
        const int seed = 123;

        runAlgorithmsDense("Low-rank (rank=" + std::to_string(rank) + ")",
            TestMat::lowRankPlusNoise(m, n, rank, 0, seed),
            10, 2, seed);

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}

#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <random>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <randla/randla.hpp>
#include <randla/threading/threading.hpp>
#include <fstream>

using RLA     = randla::RandomizedRangeFinderD;
using TestMat = randla::MatrixGeneratorsD;
using Err     = randla::metrics::ErrorEstimators<double>;

// ===== DENSE =====
static void runAlgorithmsDense(const std::string& label,
                                const Eigen::MatrixXd& A,
                                int l, int q, int seed,
                                int threads,
                                std::ostream& csv) {
    std::cout << "\n=== " << label << " ===\n";
    std::cout << "Shape: " << A.rows() << " x " << A.cols() << "\n";
    double normA = A.norm();
    std::cout << "Norm(A)=" << normA << "\n";

    auto tic = [](){return std::chrono::high_resolution_clock::now();};
    auto ms  = [](auto start, auto end){return std::chrono::duration<double, std::milli>(end-start).count();};

    int used_seed = seed + 1;
    auto t0 = tic();
    auto Q_range = RLA::randomizedRangeFinder(A, l, used_seed);
    auto t1 = tic();

    double err = Err::realError(A, Q_range);
    double time_ms = ms(t0,t1);

    std::cout << "[RRF]  cols=" << Q_range.cols()
              << " err=" << err
              << " time_ms=" << time_ms << "\n";

    csv << label << ','
        << A.rows() << ','
        << A.cols() << ','
        << normA << ','
        << "RRF" << ','
        << l << ','
        << q << ','
        << used_seed << ','
        << threads << ','
        << Q_range.cols() << ','
        << err << ','
        << time_ms << '\n';
    csv.flush();
}

int main() {
    try {
        std::cout << std::fixed << std::setprecision(6);
        std::vector<int> threadCounts = {1, 2, 4, 8, 16};

        std::cout << std::fixed << std::setprecision(6);
        const int m = 1000, n = 800, rank = 400, l = 400, q = 2;
        const int seed = 123;

        std::ofstream csv("benchmark_results.csv", std::ios::trunc);
        if (!csv) {
            std::cerr << "Error: cannot open benchmark_results.csv for writing\n";
            return 1;
        }
        csv << "label,m,n,norm,method,l,q,seed,threads,cols,err,time_ms\n";

        auto A = TestMat::lowRankPlusNoise(m, n, rank, 0, seed);

        for (int t : threadCounts) {
            randla::threading::setThreads(t);
            std::cout << "\n--- Threads = " << t
                      << " (Eigen nbThreads=" << Eigen::nbThreads() << ") ---\n";

            runAlgorithmsDense("Low-rank (rank=" + std::to_string(rank) + ")",
                A,
                l, q, seed + t,
                t,
                csv);
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
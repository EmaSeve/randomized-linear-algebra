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

    auto runOne = [&](const char* method_name, int seed_offset, const auto& method) {
        auto t0 = tic();
        auto Q  = method(used_seed + seed_offset);
        auto t1 = tic();
        double err     = Err::realError(A, Q);
        double time_ms = ms(t0, t1);
        std::cout << "[" << method_name << "]  cols=" << Q.cols()
                  << " err=" << err
                  << " time_ms=" << time_ms << "\n";
        csv << label << ','
            << A.rows() << ','
            << A.cols() << ','
            << normA << ','
            << method_name << ','
            << l << ','
            << threads << ','
            << Q.cols() << ','
            << err << ','
            << time_ms << '\n';
        csv.flush();
    };

    // runOne("RRF", 0, [&](int s){ return RLA::randomizedRangeFinder(A, l, s); });
    // runOne("RPI", 1, [&](int s){ return RLA::randomizedPowerIteration(A, l, q, s); });
    runOne("RSI", 2, [&](int s){ return RLA::randomizedSubspaceIteration(A, l, q, s); });
    // runOne("FRF", 3, [&](int s){ return RLA::fastRandomizedRangeFinder(A, l, s); });
}

int main() {
    try {
        std::cout << std::fixed << std::setprecision(6);
        std::vector<int> threadCounts = {1, 2, 4, 8, 16};

        const int m = 5000, n = 2000, rank = 1000, l = 400, q = 2;
        const int seed = 123;

        std::ofstream csv("benchmark_results.csv", std::ios::trunc);
        if (!csv) {
            std::cerr << "Error: cannot open benchmark_results.csv for writing\n";
            return 1;
        }
        csv << "label,m,n,norm,method,l,q,seed,threads,cols,err,time_ms\n";

        std::vector<std::pair<std::string, Eigen::MatrixXd>> cases;
        cases.emplace_back(
            "Low-rank (rank=400)",
            TestMat::lowRankPlusNoise(m, n, rank, 0.0, seed)
        );
        // cases.emplace_back(
        //     "Low-rank + Noise (rank=400 - noise=0.05)",
        //     TestMat::lowRankPlusNoise(m, n, rank, 0.05, seed + 1)
        // );
        // cases.emplace_back(
        //     "ExpDecay(rate=0.1)",
        //     TestMat::matrixWithExponentialDecay(m, n, 0.1, std::min(m, n), seed + 2)
        // );
        // cases.emplace_back(
        //     "Gaussian",
        //     RLA::randomGaussianMatrix(m, n, seed + 3)
        // );

        for (int t : threadCounts) {
            randla::threading::setThreads(t);
            std::cout << "\n--- Threads = " << t
                      << " (Eigen nbThreads=" << Eigen::nbThreads() << ") ---\n";
            std::cout << "OpenMP threads: " << omp_get_max_threads() << "\n";

            for (const auto& [label, A] : cases) {
                runAlgorithmsDense(label, A, l, q, seed + t, t, csv);
            }
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
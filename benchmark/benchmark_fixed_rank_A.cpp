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

#ifdef RRF_USE_OPENMP
#include <omp.h>
#endif

using RRF     = randla::RandRangeFinderD;
using RandGen = randla::random::RandomGenerator<double>;
using TestMat = randla::MatrixGeneratorsD;
using Err     = randla::metrics::ErrorEstimators<double>;

static void performLightWarmup(int seed) {
    std::cout << "Performing light warmup...\n";
    const int warmup_m = 200;
    const int warmup_n = 150;
    const int warmup_rank = 20;
    const int warmup_l = 30;
    const int warmup_q = 1;
    
    auto warmupMatrix = TestMat::lowRankPlusNoise(warmup_m, warmup_n, warmup_rank, 0.0, seed);
    
    auto Q1 = RRF::randomizedRangeFinder(warmupMatrix, warmup_l, seed);
    auto Q2 = RRF::fastRandRangeFinder(warmupMatrix, warmup_l, seed+1);
    
    volatile double dummy = Q1.norm() + Q2.norm();
    std::cout << "Warmup complete. Matrix size: " << warmup_m << "x" << warmup_n << "\n";
}

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

    runOne("RRF", 0, [&](int s){ return RRF::randomizedRangeFinder(A, l, s); });
    runOne("RPI", 1, [&](int s){ return RRF::randomizedPowerIteration(A, l, q, s); });
    runOne("RSI", 2, [&](int s){ return RRF::randomizedSubspaceIteration(A, l, q, s); });
    runOne("FRF", 3, [&](int s){ return RRF::fastRandRangeFinder(A, l, s); });
}

int main() {
    try {
        std::cout << std::fixed << std::setprecision(6);
        
#ifdef RRF_USE_OPENMP
        std::vector<int> threadCounts = {1, 2, 4, 8};
#else
        std::vector<int> threadCounts = {1};
#endif

        const int m = 2000, n = 1300, rank = 500, l = 100, q = 2;
        const int seed = 123;

        std::ofstream csv("res_benchmark_fixed_rank_A.csv", std::ios::trunc);
        if (!csv) {
            std::cerr << "Error: cannot open benchmark_results.csv for writing\n";
            return 1;
        }
        csv << "label,m,n,norm,method,l,threads,cols,err,time_ms\n";

        // Perform a light system warmup before starting actual benchmarks
        performLightWarmup(seed);

        std::vector<std::pair<std::string, Eigen::MatrixXd>> cases;
        cases.emplace_back(
            "Low-rank (rank=400)",
            TestMat::lowRankPlusNoise(m, n, rank, 0.0, seed)
        );
        cases.emplace_back(
            "Low-rank + Noise (rank=400 - noise=0.05)",
            TestMat::lowRankPlusNoise(m, n, rank, 0.05, seed + 1)
        );
        cases.emplace_back(
            "ExpDecay(rate=0.1)",
            TestMat::matrixWithExponentialDecay(m, n, 0.1, std::min(m, n), seed + 2)
        );
        cases.emplace_back(
            "Gaussian",
            RandGen::randomGaussianMatrix(m, n, seed + 3)
        );

        for (int t : threadCounts) {
            randla::threading::setThreads(t);
            
#ifdef RRF_USE_OPENMP
            std::cout << "\n--- Threads = " << t
                      << " (Eigen=" << Eigen::nbThreads() 
                      << ", OpenMP=" << omp_get_max_threads() << ") ---\n";
#else
            std::cout << "\n--- Single-threaded mode (OpenMP disabled) ---\n";
            std::cout << "Eigen threads: " << Eigen::nbThreads() << "\n";
#endif

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
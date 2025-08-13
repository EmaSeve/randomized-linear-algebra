#include <iostream>
#include <iomanip>
#include <vector>
#include <Eigen/SVD>
#include <chrono>
#include <randla/randla.hpp>

using RLA       = randla::RandomizedRangeFinderD;
using TestMat   = randla::MatrixGeneratorsD;
using Err       = randla::algorithms::ErrorEstimators<double>;


struct TestConfig {
    std::string label;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> A;
    int l;    // target samples
    int q;    // power/subspace iters
    int seed; // base seed
};

static void runAlgorithms(const TestConfig& cfg) {
    std::cout << "\n=== " << cfg.label << " ===\n";
    std::cout << "Shape: " << cfg.A.rows() << " x " << cfg.A.cols() << "\n";
    std::cout << "Norm(A)=" << cfg.A.norm() << "\n";
    auto tic = [](){return std::chrono::high_resolution_clock::now();};
    auto ms  = [](auto start, auto end){return std::chrono::duration<double, std::milli>(end-start).count();};

    auto t0 = tic();
    auto Q_range = RLA::randomizedRangeFinder(cfg.A, cfg.l, cfg.seed + 1);
    auto t1 = tic();
    auto Q_power = RLA::randomizedPowerIteration(cfg.A, cfg.l, cfg.q, cfg.seed + 2);
    auto t2 = tic();
    auto Q_subspace = RLA::randomizedSubspaceIteration(cfg.A, cfg.l, cfg.q, cfg.seed + 3);
    auto t3 = tic();
    auto Q_fast = RLA::fastRandomizedRangeFinder(cfg.A, cfg.l, cfg.seed + 4);
    auto t4 = tic();

    std::cout << "[RRF]  cols=" << Q_range.cols()    << " err=" << Err::realError(cfg.A, Q_range)    << " time_ms=" << ms(t0,t1) << "\n";
    std::cout << "[RPI q=" << cfg.q << "] cols=" << Q_power.cols()    << " err=" << Err::realError(cfg.A, Q_power)    << " time_ms=" << ms(t1,t2) << "\n";
    std::cout << "[RSI q=" << cfg.q << "] cols=" << Q_subspace.cols() << " err=" << Err::realError(cfg.A, Q_subspace) << " time_ms=" << ms(t2,t3) << "\n";
    std::cout << "[FAST] cols=" << Q_fast.cols()     << " err=" << Err::realError(cfg.A, Q_fast)     << " time_ms=" << ms(t3,t4) << "\n";
}

int main() {
    try {
        std::cout << std::fixed << std::setprecision(6);

        const int m = 1000, n = 800;
        const int seed = 123;

        std::vector<TestConfig> tests;
        // 1. Spettro esponenziale veloce (baseline esistente)
        tests.push_back({
            "Exponential decay (rate=0.5)",
            TestMat::matrixWithExponentialDecay(m, n, 0.5, seed),
            10, 2, seed
        });

        // 1. Spettro esponenziale lento (baseline esistente)
        tests.push_back({
            "Exponential decay (rate=0.1)",
            TestMat::matrixWithExponentialDecay(m, n, 0.1, seed),
            10, 2, seed
        });

        // 2. Matrice sparsa random
        tests.push_back({
            "Random sparse (dens=0.05)",
            TestMat::randomSparseMatrix(m, n, 0.05, seed + 10),
            100, 2, seed + 10
        });

        // 2. Matrice dense random
        tests.push_back({
            "Random dense (dens=1.0)",
            TestMat::randomSparseMatrix(m, n, 1.0, seed + 10),
            100, 2, seed + 10
        });

        // 3. Low-rank + rumore
        tests.push_back({
            "Low-rank + noise (rank=50, noise=0.01)",
            TestMat::lowRankPlusNoise(m, n, 50, 0.01, seed + 20),
            100, 2, seed + 20
        });

    // Execute tests
        for (const auto& cfg : tests) {
            runAlgorithms(cfg);
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}

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

    // RRF
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

    // RPI (randomized power iteration)
    t0 = tic();
    auto Q_rpi = RLA::randomizedPowerIteration(A, l, q, used_seed + 1);
    t1 = tic();
    err = Err::realError(A, Q_rpi);
    time_ms = ms(t0,t1);
    std::cout << "[RPI]  cols=" << Q_rpi.cols()
              << " err=" << err
              << " time_ms=" << time_ms << "\n";
    csv << label << ','
        << A.rows() << ','
        << A.cols() << ','
        << normA << ','
        << "RPI" << ','
        << l << ','
        << q << ','
        << used_seed + 1 << ','
        << threads << ','
        << Q_rpi.cols() << ','
        << err << ','
        << time_ms << '\n';
    csv.flush();

    // RSI (randomized subspace iteration)
    t0 = tic();
    auto Q_rsi = RLA::randomizedSubspaceIteration(A, l, q, used_seed + 2);
    t1 = tic();
    err = Err::realError(A, Q_rsi);
    time_ms = ms(t0,t1);
    std::cout << "[RSI]  cols=" << Q_rsi.cols()
              << " err=" << err
              << " time_ms=" << time_ms << "\n";
    csv << label << ','
        << A.rows() << ','
        << A.cols() << ','
        << normA << ','
        << "RSI" << ','
        << l << ','
        << q << ','
        << used_seed + 2 << ','
        << threads << ','
        << Q_rsi.cols() << ','
        << err << ','
        << time_ms << '\n';
    csv.flush();

    // FRR (fast randomized range finder, complesso, fixed-rank)
    t0 = tic();
    auto Q_frr = RLA::fastRandomizedRangeFinder(A, l, used_seed + 3);
    t1 = tic();
    err = Err::realError(A, Q_frr);
    time_ms = ms(t0,t1);
    std::cout << "[FRR]  cols=" << Q_frr.cols()
              << " err=" << err
              << " time_ms=" << time_ms << "\n";
    csv << label << ','
        << A.rows() << ','
        << A.cols() << ','
        << normA << ','
        << "FRR" << ','
        << l << ','
        << q << ','
        << used_seed + 3 << ','
        << threads << ','
        << Q_frr.cols() << ','
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

        // Crea più matrici di test dense
        std::vector<std::pair<std::string, Eigen::MatrixXd>> cases;
        cases.emplace_back(
            "Low-rank (rank=400, noise=0.00)",
            TestMat::lowRankPlusNoise(m, n, rank, 0.0, seed)
        );
        cases.emplace_back(
            "Low-rank (rank=400, noise=0.05)",
            TestMat::lowRankPlusNoise(m, n, rank, 0.05, seed + 1)
        );
        cases.emplace_back(
            "ExpDecay(rate=0.01)",
            TestMat::matrixWithExponentialDecay(m, n, 0.01, std::min(m, n), seed + 2)
        );
        cases.emplace_back(
            "Gaussian N(0,1)",
            RLA::randomGaussianMatrix(m, n, seed + 3)
        );
        {
            // Spettro custom: step (1 per i<rank) + decadimento
            TestMat::Vector sv(std::min(m, n));
            for (int i = 0; i < sv.size(); ++i) {
                if (i < rank) sv(i) = 1.0;
                else           sv(i) = 0.1 * std::exp(-0.005 * double(i - rank));
            }
            cases.emplace_back(
                "Custom spectrum (step+decay)",
                TestMat::matrixWithSingularValues(m, n, sv, seed + 4)
            );
        }

        for (int t : threadCounts) {
            randla::threading::setThreads(t);
            std::cout << "\n--- Threads = " << t
                      << " (Eigen nbThreads=" << Eigen::nbThreads() << ") ---\n";

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
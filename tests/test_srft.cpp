// tests/test_srft.cpp
#include <iostream>
#include <iomanip>
#include <randla/randla.hpp>
#include "load_matrix_market.hpp"

using namespace randla::algorithms;
using namespace randla::utils;

using RLA     = randla::RandomizedLinearAlgebraD;
using TestMat = randla::MatrixGeneratorsD;

int main() {
    using Matrix = Eigen::MatrixXd;
    std::cout << std::fixed << std::setprecision(6);

    const int rows = 200, cols = 50, rank = 5;
    const int l = 12;           // oversampling moderato (rank < l << min(m,n))
    const int seed = 42;

    using GM = randla::utils::MatrixGenerators<double>;

    auto run = [&](const std::string& name, const Matrix& A) {
        std::cout << "\n--- " << name << " ---\n";
        std::cout << "Shape: " << A.rows() << "x" << A.cols()
                  << "  ||A||=" << A.norm() << "\n";

        // 4.5: SRFT -> Q complesso
        auto Qc = RLA::fastRandomizedRangeFinder(A, l, seed);            // :contentReference[oaicite:2]{index=2}
        double err_srft = RLA::realError(A, Qc);                          // overload complesso appena aggiunto
        // Check ortonormalità: ||Q*Q - I||_F
        auto I = (Qc.adjoint() * Qc).eval();
        double ortho = (I - decltype(I)::Identity(I.rows(), I.cols())).norm();

        // Baseline: Gaussian RRF (reale) con stesso l
        auto Qg = RLA::randomizedRangeFinder(A, l, seed);                 // :contentReference[oaicite:3]{index=3}
        double err_gauss = RLA::realError(A, Qg);                          // versione reale già presente

        std::cout << "[SRFT] l=" << l
                  << "  err=" << err_srft
                  << "  ortho(Q*Q-I)=" << ortho << "\n";
        std::cout << "[RRF ] l=" << l
                  << "  err=" << err_gauss << "\n";
    };

    // Test 1: rango esatto
    GM::Vector sv = GM::Vector::Zero(std::min(rows, cols));
    for (int i = 0; i < rank; ++i) sv(i) = 1.0;
    Matrix A_exact = GM::matrixWithSingularValues(rows, cols, sv, seed);
    run("Exact rank " + std::to_string(rank), A_exact);

    // Test 2: decadimento esponenziale
    double decay = 0.2;
    Matrix A_decay = GM::matrixWithExponentialDecay(rows, cols, decay, rank, seed);
    run("Exponential decay (rate=" + std::to_string(decay) + ")", A_decay);

    // Test 3: low-rank + noise
    double sigma = 1e-3;
    Matrix A_noise = GM::lowRankPlusNoise(rows, cols, rank, sigma, seed);
    run("Low-rank + noise (sigma=" + std::to_string(sigma) + ")", A_noise);

    // Test 4: matrice reale (facoltativo)
    try {
        const std::string path = "data/well1850.mtx";
        Matrix A_real = loadMatrixMarket(path);
        run(std::string("Real matrix (") + path + ")", A_real);
    } catch (const std::exception& e) {
        std::cerr << "Skipping real matrix test: " << e.what() << "\n";
    }

    std::cout << "\n=== SRFT tests completed ===\n";
    return 0;
}

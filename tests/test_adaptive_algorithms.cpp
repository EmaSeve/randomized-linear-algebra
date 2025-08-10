#include <iostream>
#include <iomanip>
#include "../include/randla/algorithms/randomized_linear_algebra.hpp"
#include "../include/randla/utils/test_matrices.hpp"

using namespace randla::algorithms;
using namespace randla::utils;

int main() {
    using FloatType = double;
    using Matrix = Eigen::MatrixXd;

    std::cout << "=== Test Algoritmi Adattivi (Semplificato) ===" << std::endl;
    std::cout << std::fixed << std::setprecision(6);

    // Parametri
    const int rows = 1000;
    const int cols = 800;
    const int rank = 10;          // rango esatto
    const double tol = 1;     // tolleranza ampia
    const int r = 10;             // numero sonde

    std::cout << "Parametri: " << rows << "x" << cols
              << ", rank=" << rank
              << ", tol=" << tol << ", r=" << r << std::endl;

    using TM = randla::utils::TestMatrices<FloatType>;

    // Crea vettore di valori singolari: primi k >0, resto = 0
    TM::Vector sv = TM::Vector::Zero(std::min(rows, cols));
    for (int i = 0; i < rank; ++i) {
        sv(i) = 1.0; // tutti uguali a 1
    }

    // Matrice a rango esatto k
    Matrix A = TM::matrixWithSingularValues(rows, cols, sv, /*seed=*/42);

    std::cout << "Norma di A: " << A.norm() << std::endl;

    // 1. Adaptive Range Finder
    std::cout << "\n1. Test Adaptive Range Finder" << std::endl;
    std::cout << "------------------------------" << std::endl;
    try {
        std::cout << "Iniziando adaptiveRangeFinder..." << std::endl;
        Matrix Q1 = RandomizedLinearAlgebra<FloatType>::adaptiveRangeFinder(A, tol, r);
        double error1 = RandomizedLinearAlgebra<FloatType>::realError(A, Q1);

        std::cout << "✓ Successo!" << std::endl;
        std::cout << "  Dimensioni Q: " << Q1.rows() << "x" << Q1.cols() << std::endl;
        std::cout << "  Errore: " << error1 << std::endl;
    } catch (const std::exception& e) {
        std::cout << "✗ Errore: " << e.what() << std::endl;
    }

    // 2. Adaptive Power Iteration
    std::cout << "\n2. Test Adaptive Power Iteration" << std::endl;
    std::cout << "---------------------------------" << std::endl;
    try {
        std::cout << "Iniziando adaptivePowerIteration..." << std::endl;
        const double threshold = tol / (10 * std::sqrt(2.0 / M_PI));
        std::cout << "Soglia interna: " << threshold << std::endl;

        Matrix Q2 = RandomizedLinearAlgebra<FloatType>::adaptivePowerIteration(A, tol, r, 2);
        double error2 = RandomizedLinearAlgebra<FloatType>::realError(A, Q2);

        std::cout << "✓ Successo!" << std::endl;
        std::cout << "  Dimensioni Q: " << Q2.rows() << "x" << Q2.cols() << std::endl;
        std::cout << "  Errore reale: " << error2 << std::endl;
    } catch (const std::exception& e) {
        std::cout << "✗ Errore: " << e.what() << std::endl;
    }

    std::cout << "\n=== Test completati ===" << std::endl;
    return 0;
}

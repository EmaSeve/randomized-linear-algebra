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
    
    // Parametri molto semplici
    const int rows = 200;
    const int cols = 100;
    const int rank = 10;
    const double noise_level = 0.1;
    const double tol = 0.1;  // tolleranza più larga
    const int r = 10; // finestra piccola
    
    std::cout << "Parametri: " << rows << "x" << cols 
              << ", rank=" << rank 
              << ", tol=" << tol << ", r=" << r << std::endl;
    
    // Matrice semplice low-rank
    Matrix A = TestMatrices<FloatType>::lowRankMatrixWithNoise(rows, cols, rank, noise_level, 42);
    std::cout << "Norma di A: " << A.norm() << std::endl;
    
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
    
    std::cout << "\n2. Test Adaptive Power Iteration" << std::endl;
    std::cout << "---------------------------------" << std::endl;
    
    try {
        std::cout << "Iniziando adaptivePowerIteration..." << std::endl;
        const double threshold = tol / (10 * std::sqrt(2.0 / 3.14159265));
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

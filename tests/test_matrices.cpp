#include <iostream>
#include <iomanip>
#include <Eigen/SVD>
#include "../include/StochasticLA.hpp"
#include "../include/TestMatrices.hpp"

using namespace StochasticLA;
using RLA = RandomizedLinearAlgebraD;
using TestMat = TestMatricesD;

void printSingularValues(const RLA::Matrix& A, const std::string& name, int num_to_print = 10) {
    Eigen::JacobiSVD<RLA::Matrix> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
    auto singular_values = svd.singularValues();
    
    std::cout << "\n" << name << " (size: " << A.rows() << "x" << A.cols() << ")" << std::endl;
    std::cout << "Singular values: ";
    
    int num_print = std::min(num_to_print, static_cast<int>(singular_values.size()));
    for (int i = 0; i < num_print; ++i) {
        std::cout << std::fixed << std::setprecision(4) << singular_values(i);
        if (i < num_print - 1) std::cout << ", ";
    }
    if (singular_values.size() > num_to_print) {
        std::cout << ", ...";
    }
    std::cout << std::endl;
    
    // Print decay rate between first few singular values
    if (singular_values.size() > 1) {
        std::cout << "Decay ratios: ";
        for (int i = 0; i < std::min(5, static_cast<int>(singular_values.size()) - 1); ++i) {
            double ratio = singular_values(i+1) / singular_values(i);
            std::cout << std::fixed << std::setprecision(3) << ratio;
            if (i < std::min(4, static_cast<int>(singular_values.size()) - 2)) std::cout << ", ";
        }
        std::cout << std::endl;
    }
}

void testMatrixTypes() {
    const int rows = 100, cols = 80;
    const int seed = 42;
    
    std::cout << "=== TEST DI DIVERSI TIPI DI MATRICI ===" << std::endl;
    
    // 1. Matrice densa random
    auto dense_matrix = RLA::randomMatrix(rows, cols, seed);
    printSingularValues(dense_matrix, "Matrice densa random");
    
    // 2. Matrice sparse (densità 10%)
    auto sparse_matrix = TestMat::randomSparseMatrix(rows, cols, 0.1, seed);
    printSingularValues(sparse_matrix, "Matrice sparse (densità 10%)");
    
    // 3. Matrice con decadimento esponenziale veloce
    auto exp_fast = TestMat::matrixWithExponentialDecay(rows, cols, 0.5, seed);
    printSingularValues(exp_fast, "Decadimento esponenziale veloce (rate=0.5)");
    
    // 4. Matrice con decadimento esponenziale lento
    auto exp_slow = TestMat::matrixWithExponentialDecay(rows, cols, 0.1, seed);
    printSingularValues(exp_slow, "Decadimento esponenziale lento (rate=0.1)");
    
    // 5. Matrice low-rank con rumore
    auto lowrank_clean = TestMat::lowRankMatrixWithNoise(rows, cols, 10, 0.0, seed);
    printSingularValues(lowrank_clean, "Matrice low-rank (rank=10, no noise)");
    
    auto lowrank_noisy = TestMat::lowRankMatrixWithNoise(rows, cols, 10, 0.1, seed);
    printSingularValues(lowrank_noisy, "Matrice low-rank (rank=10, noise=0.1)");
    
    // 6. Matrice con valori singolari specificati
    RLA::Vector custom_sv(std::min(rows, cols));
    for (int i = 0; i < custom_sv.size(); ++i) {
        custom_sv(i) = 1.0 / (1.0 + i * i);  // Decadimento quadratico
    }
    auto custom_matrix = TestMat::matrixWithSingularValues(rows, cols, custom_sv, seed);
    printSingularValues(custom_matrix, "Matrice con valori singolari custom");
    
    // 7. Matrici strutturate
    const int struct_size = 50;
    auto hankel = TestMat::hankelMatrix(struct_size, seed);
    printSingularValues(hankel, "Matrice di Hankel");
    
    auto toeplitz = TestMat::toeplitzMatrix(struct_size, struct_size, seed);
    printSingularValues(toeplitz, "Matrice di Toeplitz");
    
    // 8. Matrice tridiagonale (tipica di discretizzazioni PDE)
    auto tridiag_standard = TestMat::tridiagonalMatrix(struct_size, 2.0, -1.0, seed);
    printSingularValues(tridiag_standard, "Matrice tridiagonale standard (2, -1)");
    
    auto tridiag_custom = TestMat::tridiagonalMatrix(struct_size, 4.0, -1.5, seed + 1);
    printSingularValues(tridiag_custom, "Matrice tridiagonale custom (4, -1.5)");
}

void demonstrateAlgorithmPerformance() {
    std::cout << "\n\n=== PERFORMANCE DEGLI ALGORITMI SU DIVERSI TIPI DI MATRICI ===" << std::endl;
    
    const int m = 200, n = 150, l = 20, q = 2;
    const int seed = 123;
    
    // Test su matrice con decadimento veloce
    auto A_fast = TestMat::matrixWithExponentialDecay(m, n, 0.3, seed);
    auto Q_power_fast = RLA::randomizedPowerIteration(A_fast, l, q);
    auto Q_subspace_fast = RLA::randomizedSubspaceIteration(A_fast, l, q);
    
    std::cout << "\nMatrice con decadimento veloce:" << std::endl;
    std::cout << "Errore Power Iteration: " << RLA::realError(A_fast, Q_power_fast) << std::endl;
    std::cout << "Errore Subspace Iteration: " << RLA::realError(A_fast, Q_subspace_fast) << std::endl;
    std::cout << "Stima errore a posteriori (Power): " << RLA::posteriorErrorEstimation(A_fast, Q_power_fast, 10, seed) << std::endl;
    
    // Test su matrice con decadimento lento
    auto A_slow = TestMat::matrixWithExponentialDecay(m, n, 0.05, seed);
    auto Q_power_slow = RLA::randomizedPowerIteration(A_slow, l, q);
    auto Q_subspace_slow = RLA::randomizedSubspaceIteration(A_slow, l, q);
    
    std::cout << "\nMatrice con decadimento lento:" << std::endl;
    std::cout << "Errore Power Iteration: " << RLA::realError(A_slow, Q_power_slow) << std::endl;
    std::cout << "Errore Subspace Iteration: " << RLA::realError(A_slow, Q_subspace_slow) << std::endl;
    std::cout << "Stima errore a posteriori (Power): " << RLA::posteriorErrorEstimation(A_slow, Q_power_slow, 10, seed) << std::endl;
    
    // Test su matrice low-rank con rumore
    auto A_noisy = TestMat::lowRankMatrixWithNoise(m, n, 15, 0.05, seed);
    auto Q_power_noisy = RLA::randomizedPowerIteration(A_noisy, l, q);
    auto Q_subspace_noisy = RLA::randomizedSubspaceIteration(A_noisy, l, q);
    
    std::cout << "\nMatrice low-rank con rumore:" << std::endl;
    std::cout << "Errore Power Iteration: " << RLA::realError(A_noisy, Q_power_noisy) << std::endl;
    std::cout << "Errore Subspace Iteration: " << RLA::realError(A_noisy, Q_subspace_noisy) << std::endl;
    std::cout << "Stima errore a posteriori (Power): " << RLA::posteriorErrorEstimation(A_noisy, Q_power_noisy, 10, seed) << std::endl;
}

int main() {
    try {
        testMatrixTypes();
        demonstrateAlgorithmPerformance();

    } catch (const std::exception& e) {
        std::cerr << "Errore: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}

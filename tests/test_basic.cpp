#include <iostream>
#include <cassert>
#include <cmath>
#include "StochasticLA.hpp"

using namespace StochasticLA;

void testBasicOperations() {
    std::cout << "Testing basic operations...\n";
    
    // Test with double precision
    using RLA = RandomizedLinearAlgebraD;
    
    // Create test matrices
    RLA::Matrix A(2, 3);
    A << 1.0, 2.0, 3.0,
         4.0, 5.0, 6.0;
    
    RLA::Matrix B(3, 2);
    B << 1.0, 2.0,
         3.0, 4.0,
         5.0, 6.0;
    
    // Test matrix multiplication
    RLA::Matrix C = RLA::multiply(A, B);
    
    std::cout << "Matrix A:\n" << A << "\n\n";
    std::cout << "Matrix B:\n" << B << "\n\n";
    std::cout << "Matrix C = A * B:\n" << C << "\n\n";
    
    // Expected result: [[22, 28], [49, 64]]
    assert(std::abs(C(0, 0) - 22.0) < 1e-10);
    assert(std::abs(C(0, 1) - 28.0) < 1e-10);
    assert(std::abs(C(1, 0) - 49.0) < 1e-10);
    assert(std::abs(C(1, 1) - 64.0) < 1e-10);
    
    // Test Frobenius norm
    double norm = RLA::frobeniusNorm(A);
    double expected_norm = std::sqrt(1 + 4 + 9 + 16 + 25 + 36); // sqrt(91)
    
    std::cout << "Frobenius norm of A: " << norm << "\n";
    std::cout << "Expected norm: " << expected_norm << "\n\n";
    
    assert(std::abs(norm - expected_norm) < 1e-10);
    
    std::cout << "Basic operations test PASSED!\n\n";
}

void testRandomMatrix() {
    std::cout << "Testing random matrix generation...\n";
    
    using RLA = RandomizedLinearAlgebraD;
    
    // Test with fixed seed for reproducibility
    RLA::Matrix R1 = RLA::randomMatrix(3, 3, 42);
    RLA::Matrix R2 = RLA::randomMatrix(3, 3, 42);
    
    std::cout << "Random matrix R1 (seed=42):\n" << R1 << "\n\n";
    std::cout << "Random matrix R2 (seed=42):\n" << R2 << "\n\n";
    
    // With same seed, matrices should be identical
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            assert(std::abs(R1(i, j) - R2(i, j)) < 1e-10);
        }
    }
    
    // Test different seed
    RLA::Matrix R3 = RLA::randomMatrix(3, 3, 123);
    std::cout << "Random matrix R3 (seed=123):\n" << R3 << "\n\n";
    
    // Different seeds should produce different matrices
    bool different = false;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            if (std::abs(R1(i, j) - R3(i, j)) > 1e-10) {
                different = true;
                break;
            }
        }
        if (different) break;
    }
    assert(different);
    
    std::cout << "Random matrix generation test PASSED!\n\n";
}

void testDifferentPrecisions() {
    std::cout << "Testing different floating point precisions...\n";
    
    // Test float precision
    using RLAF = RandomizedLinearAlgebraF;
    RLAF::Matrix Af(2, 2);
    Af << 1.0f, 2.0f, 3.0f, 4.0f;
    
    float normF = RLAF::frobeniusNorm(Af);
    std::cout << "Float precision norm: " << normF << "\n";
    
    // Test double precision
    using RLAD = RandomizedLinearAlgebraD;
    RLAD::Matrix Ad(2, 2);
    Ad << 1.0, 2.0, 3.0, 4.0;
    
    double normD = RLAD::frobeniusNorm(Ad);
    std::cout << "Double precision norm: " << normD << "\n";
    
    // Test long double precision
    using RLALD = RandomizedLinearAlgebraLD;
    RLALD::Matrix Ald(2, 2);
    Ald << 1.0L, 2.0L, 3.0L, 4.0L;
    
    long double normLD = RLALD::frobeniusNorm(Ald);
    std::cout << "Long double precision norm: " << normLD << "\n\n";
    
    // All should be approximately equal to sqrt(30)
    double expected = std::sqrt(30.0);
    assert(std::abs(normF - expected) < 1e-6);
    assert(std::abs(normD - expected) < 1e-15);
    // Be more lenient with long double as precision varies by system
    assert(std::abs(normLD - expected) < 1e-14);
    
    std::cout << "Different precisions test PASSED!\n\n";
}

int main() {
    std::cout << "=== StochasticLA Library Test ===\n\n";
    
    try {
        testBasicOperations();
        testRandomMatrix();
        testDifferentPrecisions();
        
        std::cout << "All tests PASSED! ✅\n";
        std::cout << "Eigen integration is working correctly.\n";
        
    } catch (const std::exception& e) {
        std::cerr << "Test FAILED: " << e.what() << "\n";
        return 1;
    }
    
    return 0;
}

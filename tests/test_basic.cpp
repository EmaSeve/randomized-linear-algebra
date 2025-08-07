#include <iostream>
#include <cassert>
#include <cmath>
#include "StochasticLA.hpp"

using namespace StochasticLA;

void testRandomMatrix() {
    
    using RLA = RandomizedLinearAlgebraD;
    
    // Test with fixed seed for reproducibility
    RLA::Matrix R1 = RLA::randomMatrix(3, 3, 42);
    RLA::Matrix R2 = RLA::randomMatrix(3, 3, 42);
    
    // With same seed, matrices should be identical
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            assert(std::abs(R1(i, j) - R2(i, j)) < 1e-10);
        }
    }
    
    // Test different seed
    RLA::Matrix R3 = RLA::randomMatrix(3, 3, 123);
    
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
    
}

int main() {
    std::cout << "=== StochasticLA Library Test ===\n\n";
    
    try {
        testRandomMatrix();
        
    } catch (const std::exception& e) {
        std::cerr << "Test FAILED: " << e.what() << "\n";
        return 1;
    }
    
    return 0;
}

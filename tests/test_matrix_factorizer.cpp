#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <random>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <randla/randla.hpp>

using RLA     = randla::RandomizedRangeFinderD;
using TestMat = randla::MatrixGeneratorsD;
using Err     = randla::metrics::ErrorEstimators<double>;
using MF      = randla::algorithms::MatrixFactorizer<double>;


void test_IDfactorizationI(const RLA::Matrix & A,const std::vector<int> & rank, int seed){
     for(auto r : rank){
        auto id = MF::IDFactorizationI(A, r, seed);
        auto B = id.B;
        auto P = id.P;

        auto residual = (A - B * P);
        double frobenius_error = residual.norm() / A.norm();

        Eigen::BDCSVD<RLA::CMatrix> svdA(A);
        Eigen::BDCSVD<RLA::CMatrix> svdR(residual);
        double spec_error = svdR.singularValues()(0) / svdA.singularValues()(0);  
        
        double energy_ratio = 1.0 - residual.squaredNorm() / A.squaredNorm();

        std::cout<< "rank: "<<r<<std::endl;
        std::cout << "Relative Frobenius error: " << frobenius_error << "\n";
        std::cout << "Relative Spectral error  : " << spec_error << "\n";
        std::cout << "Preserved energy         : " << energy_ratio * 100 << "%\n\n";
   }
}

int main(void){

   int m = 100, n = 100;
   double density = 0.5;
   double decay_rate = 0.5; 
   double noise = 0.5;
   int rank = 110;
   int seed = 100;

   std::vector<int> ranks {10, 30, 50, 70, 90};

   RLA::Matrix A = TestMat::randomSparseMatrix(m ,n, density, seed);
   std::cout<< "- random sparse matrix -" <<std::endl;
   test_IDfactorizationI(A, ranks, seed);

   RLA::Matrix X = TestMat::matrixWithExponentialDecay(m ,n, decay_rate, rank, seed);
   std::cout<< "- exponential decay matrix (rank = 110, decay = 0.5) -" <<std::endl;
   test_IDfactorizationI(X, ranks, seed);
   
   RLA::Matrix Y = TestMat::lowRankPlusNoise(m ,n, rank, noise, seed);
   std::cout<< "- low rank noise matrix (rank = 110, noise = 0.5) -" <<std::endl;
   test_IDfactorizationI(Y, ranks, seed);

   return 0;
}




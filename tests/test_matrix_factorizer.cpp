#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <random>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <randla/randla.hpp>

using RRF     = randla::RandRangeFinderD;
using TestMat = randla::MatrixGeneratorsD;
using Err     = randla::metrics::ErrorEstimators<double>;
using MF      = randla::algorithms::MatrixFactorizer<double>;

void print_approximation_error(const RRF::Matrix & A, const randla::Types<double>::IDResult & id, double r_t){
        
   auto B = id.B;
   auto P = id.P;

   auto residual = (A - B * P);
   double frobenius_error = residual.norm() / A.norm();

   Eigen::BDCSVD<RRF::CMatrix> svdA(A);
   Eigen::BDCSVD<RRF::CMatrix> svdR(residual);
   double spec_error = svdR.singularValues()(0) / svdA.singularValues()(0);  
        
   double energy_ratio = 1.0 - residual.squaredNorm() / A.squaredNorm();

   std::cout<< "rank/tol: "<<r_t<<std::endl;
   std::cout << "Relative Frobenius error: " << frobenius_error << "\n";
   std::cout << "Relative Spectral error  : " << spec_error << "\n";
   std::cout << "Preserved energy         : " << energy_ratio * 100 << "%\n\n";

}
void test_IDfactorizationI(const RRF::Matrix & A, const std::vector<int> & rank, int seed){
     for(auto r : rank){
        auto id = MF::IDFactorizationI(A, r, seed);

        print_approximation_error(A, id, r);
   }
}

void test_adaptiveIDFactorization(const RRF::Matrix & A, const std::vector<double> & tols, int seed){

   for(auto tol : tols){
      auto id = MF::adaptiveIDFactorization(A, tol, seed);

      print_approximation_error(A, id, tol);
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

   RRF::Matrix A = TestMat::randomSparseMatrix(m ,n, density, seed);
   std::cout<< "- random sparse matrix -" <<std::endl;
   test_IDfactorizationI(A, ranks, seed);

/*    RRF::Matrix X = TestMat::matrixWithExponentialDecay(m ,n, decay_rate, rank, seed);
   std::cout<< "- exponential decay matrix (rank = 110, decay = 0.5) -" <<std::endl;
   test_IDfactorizationI(X, ranks, seed);
   
   RRF::Matrix Y = TestMat::lowRankPlusNoise(m ,n, rank, noise, seed);
   std::cout<< "- low rank noise matrix (rank = 110, noise = 0.5) -" <<std::endl;
   test_IDfactorizationI(Y, ranks, seed); */

   // Adaptive version, specify a tolerance
   std::cout<< "-- Adaptive ID factorization --"<<std::endl;

   std::vector<double> tols = {10, 1, 0.5, 0.2, 0.15, 0.1};

   std::cout<< "- random sparse matrix -" <<std::endl;
   test_adaptiveIDFactorization(A, tols, seed);

   return 0;
}




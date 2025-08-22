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
using FloatType = double;
using Matrix = Eigen::MatrixXd;

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
void test_IDfactorization(const RRF::Matrix & A, const std::vector<int> & rank, int seed){
     for(auto r : rank){
        auto id = MF::IDFactorization(A, r, seed);

        print_approximation_error(A, id, r);
   }
}

void test_adaptiveIDFactorization(const RRF::Matrix & A, const std::vector<double> & tols, int seed){

   for(auto tol : tols){
      RLA::IDResult id;
      bool success = true;

      try{
         id = MF::adaptiveIDFactorization(A, tol, seed);
      } catch(const std::runtime_error & err){
         std::cout<< err.what() <<std::endl;
         success = false;
      }

      if(success) print_approximation_error(A, id, tol);
   }

}

void test_directSVD(const Matrix& A, const Matrix& Q, double tol){
   
   RLA::SVDResult svd_A;
   bool success = true;

   try{
      svd_A = MF::directSVD(A, Q, tol);
   } catch(const std::runtime_error & err){
      std::cerr<< err.what() <<std::endl;
      success = false;
   }

   if(success){
      auto A_approx = svd_A.U * svd_A.S.asDiagonal() * svd_A.V.adjoint();
      double real_err = (A - A_approx).norm();
      std::cout<< "Real error of SVD approx of A, given Q is: "<< real_err <<std::endl;
   }

}

void test_SVDviaROwExtraction(const Matrix& A, const Matrix & Q, double tol){
   RLA::SVDResult svd_A;
   bool success = true;

   try{
      svd_A = MF::SVDViaRowExtraction(A, Q, tol);
   } catch(const std::runtime_error & err){
      std::cerr<< err.what() <<std::endl;
      success = false;
   }

   if(success){
      auto A_approx = svd_A.U * svd_A.S.asDiagonal() * svd_A.V.transpose();
      double real_err = (A - A_approx).norm();
      std::cout<< "Real error of SVD via row extracion approx of A, given Q is: "<< real_err <<std::endl;
   }

}

void test_directEigDecomposition(const Matrix & Hermitian_A, const Matrix & Q, double tol){
   RLA::EigenvalueDecomposition ed;
   bool success = true;

   try{
      ed = MF::directEigenvalueDecomposition(Hermitian_A, Q, tol);
   } catch(const std::runtime_error & err){
      std::cerr<< err.what() <<std::endl;
      success = false;
   }

   if(success){
      auto Hermitian_A_approx = ed.U * ed.Lambda * ed.U.transpose();
      double real_err = (Hermitian_A - Hermitian_A_approx).norm();
      std::cout<< "Real error of direct eigenvalue decomposition approximation of A, given Q is: "<< real_err <<std::endl;
   }

}



int main(void){

   randla::threading::setThreads(1);

/*************************************
***** Test ID factorization
*************************************/

   int m = 100, n = 100;
   double density = 0.5;
   double decay_rate = 0.5; 
   double noise = 0.5;
   int rank = 110;
   int seed = 100;

   std::vector<int> ranks {10, 30, 50, 70, 90};

   RRF::Matrix A = TestMat::randomSparseMatrix(m ,n, density, seed);
   std::cout<< "- random sparse matrix -" <<std::endl;
   test_IDfactorization(A, ranks, seed);

/*    RRF::Matrix X = TestMat::matrixWithExponentialDecay(m ,n, decay_rate, rank, seed);
   std::cout<< "- exponential decay matrix (rank = 110, decay = 0.5) -" <<std::endl;
   test_IDfactorization(X, ranks, seed);
   
   RRF::Matrix Y = TestMat::lowRankPlusNoise(m ,n, rank, noise, seed);
   std::cout<< "- low rank noise matrix (rank = 110, noise = 0.5) -" <<std::endl;
   test_IDfactorization(Y, ranks, seed); */

   // Adaptive version, specify a tolerance
   std::cout<< "-- Adaptive ID factorization --"<<std::endl;

   std::vector<double> tols = {10, 1, 0.5, 0.2, 0.15, 0.1};

   std::cout<< "- random sparse matrix -" <<std::endl;
   test_adaptiveIDFactorization(A, tols, seed);

/************************************    
****** Test of Algorithms of Stage B 
************************************/  

   const int rows = 100;
   const int cols = 80;
   const int size = 100; // for square matrix (size, size)
   density = 0.9;
   seed = 42;
   double tol = 0.1;
   int r = 20;

// ------------------------------------------------------------------------------------------------------
   auto B = TestMat::randomSparseMatrix(rows,cols,density,seed);
   Matrix Q = RLA::adaptiveRangeFinder(A, tol, r, seed);

   test_directSVD(A, Q, tol);
   test_SVDviaROwExtraction(A, Q, tol); 
// ------------------------------------------------------------------------------------------------------
   B = TestMat::randomHermitianSparseMatrix(size, density, seed + 1);
   Q = RLA::adaptiveFastRandomizedRangeFinder(B, tol, 15, seed +1).real();

double er = Err::realError(A, Q);
   std::cout<< "adaptive fast randomize range finder err:"<< er<<std::endl;

   test_directEigDecomposition(A, Q, tol);

   return 0;
}




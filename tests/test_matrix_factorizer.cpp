#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <randla/randla.hpp>

using RRF     = randla::RandRangeFinderD;
using ARRF    = randla::AdaptiveRandRangeFinderD;
using TestMat = randla::MatrixGeneratorsD;
using Err     = randla::metrics::ErrorEstimators<double>;
using MF      = randla::MatrixFactorizerD;

using Matrix   = randla::Types<double>::Matrix;
using CMatrix  = randla::Types<double>::CMatrix;
using SVDResult = randla::Types<double>::SVDResult;
using IDResult  = randla::Types<double>::IDResult;
using EigenvalueDecomposition = randla::Types<double>::EigenvalueDecomposition;


void print_approximation_error(const CMatrix & A, const IDResult & id, double r_t){
        
   auto B = id.B;
   auto P = id.P;

   auto residual = (A - B * P);
   double frobenius_error = residual.norm() / A.norm();

   
   Eigen::BDCSVD<Matrix> svdA(A.real());
   Eigen::BDCSVD<Matrix> svdR(residual.real());
   double spec_error = svdR.singularValues()(0) / svdA.singularValues()(0);  
        
   double energy_ratio = 1.0 - residual.squaredNorm() / A.squaredNorm();

   std::cout<< "rank/tol: "<<r_t<<std::endl;
   std::cout << "Relative Frobenius error: " << frobenius_error << "\n";
   std::cout << "Relative Spectral error  : " << spec_error << "\n";
   std::cout << "Preserved energy         : " << energy_ratio * 100 << "%\n\n";
}


void test_IDfactorization(const CMatrix & A, const std::vector<int> & rank, int seed){
     for(auto r : rank){
        auto id = MF::IDFactorization(A, r, seed);
        print_approximation_error(A, id, r);
   }
}


void test_adaptiveIDFactorization(const CMatrix & A, const std::vector<double> & tols, int seed){
   for(auto tol : tols){
      IDResult id;
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
   SVDResult svd_A;
   bool success = true;

   try{
      svd_A = MF::directSVD(A, Q, tol);
   } catch(const std::runtime_error & err){
      std::cerr<< err.what() <<std::endl;
      success = false;
   }

   if(success){
      auto A_approx = svd_A.U * svd_A.S.asDiagonal() * svd_A.V.transpose();
      double real_err = (A - A_approx).norm();
      std::cout<< "Real error of SVD approx of A, given Q is: "<< real_err <<std::endl;
   }
}

void test_SVDviaRowExtraction(const Matrix& A, const Matrix & Q, double tol){
   SVDResult svd_A;
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
   EigenvalueDecomposition ed;
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

   
   CMatrix A_cmat = TestMat::randomSparseMatrix(m ,n, density, seed).cast<std::complex<double>>();
   std::cout<< "- random sparse matrix -" <<std::endl;
   test_IDfactorization(A_cmat, ranks, seed);

   CMatrix X_cmat = TestMat::matrixWithExponentialDecay(m ,n, decay_rate, rank, seed).cast<std::complex<double>>();
   std::cout<< "- exponential decay matrix (rank = 110, decay = 0.5) -" <<std::endl;
   test_IDfactorization(X_cmat, ranks, seed);
   
   CMatrix Y_cmat = TestMat::lowRankPlusNoise(m ,n, rank, noise, seed).cast<std::complex<double>>();
   std::cout<< "- low rank noise matrix (rank = 110, noise = 0.5) -" <<std::endl;
   test_IDfactorization(Y_cmat, ranks, seed); 

   // Adaptive version, specify a tolerance
   std::cout<< "-- Adaptive ID factorization --"<<std::endl;
   std::vector<double> tols = {10, 1, 0.5, 0.2, 0.15, 0.1};
   std::cout<< "- random sparse matrix -" <<std::endl;
   test_adaptiveIDFactorization(A_cmat, tols, seed);
   
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
   Matrix A = TestMat::randomSparseMatrix(rows,cols,density,seed);
   Matrix Q = ARRF::adaptiveRangeFinder(A, tol, r, seed); // Aggiunto punto e virgola

   double er = Err::realError(A, Q);
   std::cout<< "adaptive range finder err:"<< er<<std::endl;

   test_directSVD(A, Q, tol);
   test_SVDviaRowExtraction(A, Q, tol); 
// ------------------------------------------------------------------------------------------------------
   Matrix B = TestMat::randomHermitianSparseMatrix(size, density, seed + 1);
   Q = ARRF::adaptiveFastRandRangeFinder(B, tol, 15, seed +1).real();
   
   test_directEigDecomposition(B, Q, tol);

   return 0;
}




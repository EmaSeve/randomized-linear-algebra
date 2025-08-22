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
         id = MF::adaptiveIDFactorization(A, tol, seed, 0.8);
      } catch(const std::runtime_error & err){
         std::cout<< err.what() <<std::endl;
         success = false;
      }

      if(success) print_approximation_error(A, id, tol);
   }
}

void test_adaptiveIDFactorization(const CMatrix & A, double & tol, int seed){
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
      std::cerr<< "test_SVDviaRowExtraction err: " << err.what() <<std::endl;
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
      std::cerr<< "Direct eig err throw: "<<err.what() <<std::endl;
      success = false;
   }

   if(success){
      auto Hermitian_A_approx = ed.U * ed.Lambda * ed.U.transpose();
      double real_err = (Hermitian_A - Hermitian_A_approx).norm();
      std::cout<< "Real error of direct eigenvalue decomposition approximation of A, given Q is: "<< real_err <<std::endl;
   }
}

void test_eigenvalueDecompositionViaRowExtraction(const Matrix & Hermitian_A, const Matrix & Q, double tol){
   EigenvalueDecomposition ed;
   bool success = true;

   try{
      ed = MF::eigenvalueDecompositionViaRowExtraction(Hermitian_A, Q, tol);
   } catch(const std::runtime_error & err){
      std::cerr<< " Row extraction err trhow: " <<err.what() <<std::endl;
      success = false;
   }

   if(success){
      auto Hermitian_A_approx = ed.U * ed.Lambda * ed.U.transpose();
      double real_err = (Hermitian_A - Hermitian_A_approx).norm();
      std::cout<< "Real error of Row Extraction eigenvalue decomposition approximation of A, given Q is: "<< real_err <<std::endl;
   }
}

void test_eigenvalueDecompositionViaNystromMethod(const Matrix & spd_A, const Matrix & Q, double tol){
   EigenvalueDecomposition ed;
   bool success = true;

   try{
      ed = MF::eigenvalueDecompositionViaNystromMethod(spd_A, Q, tol);
   } catch(const std::runtime_error & err){
      std::cerr<< "Nystrom err throw: " <<err.what() <<std::endl;
      success = false;
   }

   if(success){
      auto spd_A_approx = ed.U * ed.Lambda * ed.U.transpose();
      double real_err = (spd_A - spd_A_approx).norm();
      std::cout<< "Real error of Nystrom eigenvalue decomposition approximation of A, given Q is: "<< real_err <<std::endl;
   }
}

void test_eigenvalueDecompositionInOnePass(const Matrix & Hermitian_A, const Matrix & Q,const Matrix & Omega, double tol){
   EigenvalueDecomposition ed;
   bool success = true;

   try{
      ed = MF::eigenvalueDecompositionInOnePass(Hermitian_A, Q, Omega, tol);
   } catch(const std::runtime_error & err){
      std::cerr<< "One pass err throw: " <<err.what() <<std::endl;
      success = false;
   }

   if(success){
      auto Hermitian_A_approx = ed.U * ed.Lambda * ed.U.transpose();
      double real_err = (Hermitian_A - Hermitian_A_approx).norm();
      std::cout<< "Real error of One pass eigenvalue decomposition approximation of A, given Q is: "<< real_err <<std::endl;
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

   
/*    CMatrix A_cmat = TestMat::randomSparseMatrix(m ,n, density, seed).cast<std::complex<double>>();
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
   */ 
/************************************    
****** Test of Algorithms of Stage B 
************************************/  
   const int rows = 10;
   const int cols = 8;
   const int size = 10; // for square matrix (size, size)
   density = 0.9;
   seed = 42;
   double tol = 0.2;
   int r = 7;
   rank = 8;

/**
 * -----------  Building all type of matrices A
 **/ 

   Matrix sparse_A            = TestMat::randomSparseMatrix(rows,cols,density);
   Matrix sparse_hermitian_A  = TestMat::randomHermitianSparseMatrix(size, density);
   Matrix psd_A               = TestMat::randomPositiveSemidefiniteMatrix(size);
   Matrix lowRank_A           = TestMat::lowRankPlusNoise(rows, cols, rank, 0.0);
   Matrix lowRankNoise_A      = TestMat::lowRankPlusNoise(rows, cols, rank, 0.5);
   
   Eigen::VectorXd sv = Eigen::VectorXd::Zero(std::min(rows, cols));
   for (int i = 0; i < rank; ++i) sv(i) = 1.0;

   Matrix singularValues_A    = TestMat::matrixWithSingularValues(rows, cols, sv);
   Matrix expDecay_A          = TestMat::matrixWithExponentialDecay(rows, cols, 0.5, rank);
/**
 * -----------  Building corrisponding matrices Q
 **/ 

   // Adaptive method to compute Q (NOT for sparse matrix)
   Matrix Q_psd               = ARRF::adaptivePowerIteration(psd_A, tol, r, 2, -1);
   Matrix Q_psd2              = ARRF::adaptiveRangeFinder(psd_A, tol, r, -1);
   Matrix Q_lowRank           = ARRF::adaptivePowerIteration(lowRank_A, tol, r, 2, -1);
   Matrix Q_lowRank2          = ARRF::adaptiveRangeFinder(lowRank_A, tol, r, -1);
   Matrix Q_singularValues    = ARRF::adaptivePowerIteration(singularValues_A, tol, r, 2, -1);
   Matrix Q_expDecay          = ARRF::adaptiveRangeFinder(expDecay_A, tol, r, -1);

   // Non adaptive method for sparse
   Matrix Q_sparse = RRF::randomizedRangeFinder(sparse_A, r, -1);
   Matrix Q_sparse2 = RRF::randomizedPowerIteration(sparse_A, r, 2, -1);
   Matrix Q_hermitian_sparse = RRF::randomizedSubspaceIteration(sparse_hermitian_A, r, 4, -1);
   Matrix Q_hermitian_sparse2 = RRF::fastRandRangeFinder(sparse_hermitian_A, r, -1).real();
   
   
/**
 * Adaptive ID tested on Q
 *  */ 

 double id_tol = 1;

   std::cout<< "-- Adaptive ID factorization --"<<std::endl;
   std::cout<< "- Q_psd -" <<std::endl;
   test_adaptiveIDFactorization(Q_psd, id_tol, seed); 

   std::cout<< "- Q_psd-2 -" <<std::endl;
   test_adaptiveIDFactorization(Q_psd2, id_tol, seed);

   std::cout<< "- Q_lowRank -" <<std::endl;
   test_adaptiveIDFactorization(Q_lowRank, id_tol, seed);

   std::cout<< "- Q_lowRank-2 -" <<std::endl;
   test_adaptiveIDFactorization(Q_lowRank2, id_tol, seed);

   std::cout<< "- Q_singularValue -" <<std::endl;
   test_adaptiveIDFactorization(Q_singularValues, id_tol, seed);

   std::cout<< "- Q_expDecay -" <<std::endl;
   test_adaptiveIDFactorization(Q_expDecay, id_tol, seed);

   std::cout<< "- Q_sparse -" <<std::endl;
   test_adaptiveIDFactorization(Q_sparse, id_tol, seed);

   std::cout<< "- Q_sparse-2 -" <<std::endl;
   test_adaptiveIDFactorization(Q_sparse2, id_tol, seed);


   std::cout<< "- Q_Hermitiansparse -" <<std::endl;
   test_adaptiveIDFactorization(Q_hermitian_sparse, id_tol, seed);

   std::cout<< "- Q_Hermitiansparse-2 -" <<std::endl;
   test_adaptiveIDFactorization(Q_hermitian_sparse2, id_tol, seed);

/**
 *  Stage B methods 
 *  */ 

   std::cout << "--- Testing A = sparse_A, Q = Q_sparse ---" << std::endl;
   double err_sparse_A_Q_sparse = Err::realError(sparse_A, Q_sparse);
   std::cout << "||A - QQ^T A|| = " << err_sparse_A_Q_sparse << std::endl;

   test_directSVD(sparse_A, Q_sparse, tol);
   test_SVDviaRowExtraction(sparse_A, Q_sparse, tol);
   

   std::cout << "--- Testing A = sparse_A, Q = Q_sparse2 ---" << std::endl;
   double err_sparse_A_Q_sparse2 = Err::realError(sparse_A, Q_sparse2);
   std::cout << "||A - QQ^T A|| = " << err_sparse_A_Q_sparse2 << std::endl;

   test_directSVD(sparse_A, Q_sparse2, tol);
   test_SVDviaRowExtraction(sparse_A, Q_sparse2, tol);

   std::cout << "--- Testing A = sparse_hermitian_A, Q = Q_hermitian_sparse ---" << std::endl;
   double err_sparse_hermitian_A_Q_hermitian_sparse = Err::realError(sparse_hermitian_A, Q_hermitian_sparse);
   std::cout << "||A - QQ^T A|| = " << err_sparse_hermitian_A_Q_hermitian_sparse << std::endl;

   test_directSVD(sparse_hermitian_A, Q_hermitian_sparse, tol);
   test_SVDviaRowExtraction(sparse_hermitian_A, Q_hermitian_sparse, tol);
   test_directEigDecomposition(sparse_hermitian_A, Q_hermitian_sparse, tol);
   test_eigenvalueDecompositionViaRowExtraction(sparse_hermitian_A, Q_hermitian_sparse, tol);
   test_eigenvalueDecompositionInOnePass(sparse_hermitian_A, Q_hermitian_sparse, TestMat::randomDenseMatrix(cols, r), tol);

   std::cout << "--- Testing A = sparse_hermitian_A, Q = Q_hermitian_sparse2 ---" << std::endl;
   double err_sparse_hermitian_A_Q_hermitian_sparse2 = Err::realError(sparse_hermitian_A, Q_hermitian_sparse2);
   std::cout << "||A - QQ^T A|| = " << err_sparse_hermitian_A_Q_hermitian_sparse2 << std::endl;

   test_directSVD(sparse_hermitian_A, Q_hermitian_sparse2, tol);
   test_SVDviaRowExtraction(sparse_hermitian_A, Q_hermitian_sparse2, tol);
   test_directEigDecomposition(sparse_hermitian_A, Q_hermitian_sparse2, tol);
   test_eigenvalueDecompositionViaRowExtraction(sparse_hermitian_A, Q_hermitian_sparse2, tol);
   test_eigenvalueDecompositionInOnePass(sparse_hermitian_A, Q_hermitian_sparse2, TestMat::randomDenseMatrix(cols, r), tol);

   std::cout << "--- Testing A = psd_A, Q = Q_psd ---" << std::endl;
   double err_psd_A_Q_psd = Err::realError(psd_A, Q_psd);
   std::cout << "||A - QQ^T A|| = " << err_psd_A_Q_psd << std::endl;

   test_directSVD(psd_A, Q_psd, tol);
   test_SVDviaRowExtraction(psd_A, Q_psd, tol);
   test_eigenvalueDecompositionViaNystromMethod(psd_A, Q_psd, tol);

   std::cout << "--- Testing A = psd_A, Q = Q_psd2 ---" << std::endl;
   double err_psd_A_Q_psd2 = Err::realError(psd_A, Q_psd2);
   std::cout << "||A - QQ^T A|| = " << err_psd_A_Q_psd2 << std::endl;

   test_directSVD(psd_A, Q_psd2, tol);
   test_SVDviaRowExtraction(psd_A, Q_psd2, tol);
   test_eigenvalueDecompositionViaNystromMethod(psd_A, Q_psd2, tol);

   std::cout << "--- Testing A = lowRank_A, Q = Q_lowRank ---" << std::endl;
   double err_lowRank_A_Q_lowRank = Err::realError(lowRank_A, Q_lowRank);
   std::cout << "||A - QQ^T A|| = " << err_lowRank_A_Q_lowRank << std::endl;

   test_directSVD(lowRank_A, Q_lowRank, tol);
   test_SVDviaRowExtraction(lowRank_A, Q_lowRank, tol);
   

   std::cout << "--- Testing A = lowRank_A, Q = Q_lowRank2 ---" << std::endl;
   double err_lowRank_A_Q_lowRank2 = Err::realError(lowRank_A, Q_lowRank2);
   std::cout << "||A - QQ^T A|| = " << err_lowRank_A_Q_lowRank2 << std::endl;

   test_directSVD(lowRank_A, Q_lowRank2, tol);
   test_SVDviaRowExtraction(lowRank_A, Q_lowRank2, tol);
   

   std::cout << "--- Testing A = lowRankNoise_A, Q = Q_lowRank ---" << std::endl;
   double err_lowRankNoise_A_Q_lowRank = Err::realError(lowRankNoise_A, Q_lowRank);
   std::cout << "||A - QQ^T A|| = " << err_lowRankNoise_A_Q_lowRank << std::endl;

   test_directSVD(lowRankNoise_A, Q_lowRank, tol);
   test_SVDviaRowExtraction(lowRankNoise_A, Q_lowRank, tol);
   

   std::cout << "--- Testing A = lowRankNoise_A, Q = Q_lowRank2 ---" << std::endl;
   double err_lowRankNoise_A_Q_lowRank2 = Err::realError(lowRankNoise_A, Q_lowRank2);
   std::cout << "||A - QQ^T A|| = " << err_lowRankNoise_A_Q_lowRank2 << std::endl;

   test_directSVD(lowRankNoise_A, Q_lowRank2, tol);
   test_SVDviaRowExtraction(lowRankNoise_A, Q_lowRank2, tol);
   

   std::cout << "--- Testing A = singularValues_A, Q = Q_singularValues ---" << std::endl;
   double err_singularValues_A_Q_singularValues = Err::realError(singularValues_A, Q_singularValues);
   std::cout << "||A - QQ^T A|| = " << err_singularValues_A_Q_singularValues << std::endl;

   test_directSVD(singularValues_A, Q_singularValues, tol);
   test_SVDviaRowExtraction(singularValues_A, Q_singularValues, tol);
   

   std::cout << "--- Testing A = expDecay_A, Q = Q_expDecay ---" << std::endl;
   double err_expDecay_A_Q_expDecay = Err::realError(expDecay_A, Q_expDecay);
   std::cout << "||A - QQ^T A|| = " << err_expDecay_A_Q_expDecay << std::endl;

   test_directSVD(expDecay_A, Q_expDecay, tol);
   test_SVDviaRowExtraction(expDecay_A, Q_expDecay, tol);
   

   Matrix dense_A = TestMat::randomDenseMatrix(rows, cols);
   Matrix Q_dense = ARRF::adaptiveRangeFinder(dense_A, tol, r, -1);
   Matrix Q_dense2 = ARRF::adaptivePowerIteration(dense_A, tol, r, 2, -1);

   std::cout << "--- Testing A = dense_A, Q = Q_dense ---" << std::endl;
   double err_dense = Err::realError(dense_A, Q_dense);
   std::cout << "||A - QQ^T A|| = " << err_dense << std::endl;

   test_directSVD(dense_A, Q_dense, tol);
   test_SVDviaRowExtraction(dense_A, Q_dense, 5.0);

   std::cout << "--- Testing A = dense_A, Q = Q_dense2 ---" << std::endl;
   double err_dense2 = Err::realError(dense_A, Q_dense2);
   std::cout << "||A - QQ^T A|| = " << err_dense2 << std::endl;

   test_directSVD(dense_A, Q_dense2, tol);
   test_SVDviaRowExtraction(dense_A, Q_dense2, 5.0);
   


/* // ------------------------------------------------------------------------------------------------------
   Matrix A = TestMat::randomSparseMatrix(rows,cols,density,seed);
   Matrix Q = ARRF::adaptiveRangeFinder(A, tol, r, seed); // Aggiunto punto e virgola

   double er = Err::realError(A, Q);
   std::cout<< "adaptive range finder err:"<< er<<std::endl;

   test_directSVD(A, Q, tol);

    
// ------------------------------------------------------------------------------------------------------
   Matrix B = TestMat::randomPositiveSemidefiniteMatrix(size, density, seed + 1);
   Q = ARRF::adaptiveFastRandRangeFinder(B, tol, 15, seed +1).real();

   er = Err::realError(B, Q);
   std::cout<< "(SPD matrix A) adaptive fast range finder err:"<< er<<std::endl;

   test_SVDviaRowExtraction(B, Q, tol); 
   test_directEigDecomposition(B, Q, tol); */

   return 0;
}




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


void print_approximation_error(const CMatrix & A, const IDResult & id, double r_t) {
    const auto& B = id.B;
    const auto& P = id.P;
    const auto residual = (A - B * P);

    const double frobenius_error = residual.norm() / A.norm();
    const double energy_ratio = 1.0 - residual.squaredNorm() / A.squaredNorm();

    // Spectral analisys
    Eigen::BDCSVD<Matrix> svdA(A.real());
    Eigen::BDCSVD<Matrix> svdR(residual.real());

    double sigma_kp1 = 0.0;
    int k = B.cols();
    if (svdA.singularValues().size() > k)
        sigma_kp1 = svdA.singularValues()(k);

    const double relative_spec_error = svdR.singularValues()(0) / (svdA.singularValues()(0) + 1e-14);
    const double error_vs_sigma_kp1 = svdR.singularValues()(0) / (sigma_kp1 + 1e-14);

    // Norm and Conditioning
    const double p_max = P.cwiseAbs().maxCoeff();

    Eigen::JacobiSVD<CMatrix> svdB(B, Eigen::ComputeThinU | Eigen::ComputeThinV);
    const auto s_vals = svdB.singularValues();
    const double cond_B = s_vals(0) / (s_vals.tail(1)(0) + 1e-14);

    std::cout << "-------------------------------------------------\n";
    std::cout << "rank/tol                 : " << r_t << "\n";
    std::cout << "Relative Frobenius error : " << frobenius_error << "\n";
    std::cout << "Relative Spectral error  : " << relative_spec_error << "\n";
    std::cout << "‖A - BP‖₂ / σₖ₊₁(A)       : " << error_vs_sigma_kp1 << "\n";
    std::cout << "Preserved energy         : " << energy_ratio * 100 << "%\n";
    std::cout << "‖P‖_∞                    : " << p_max << "\n";
    std::cout << "cond(B)                  : " << cond_B << "\n";
    std::cout << "-------------------------------------------------\n\n";
}



int numerical_rank(const Matrix& Q, double tol = -1){
   Eigen::JacobiSVD<Matrix> svd(Q);
   auto singularValues = svd.singularValues();
   double threshold = tol;

   if (threshold < 0) {
      threshold = std::max(Q.rows(), Q.cols()) * singularValues.array().abs().maxCoeff() * std::numeric_limits<double>::epsilon();
   }

   return (singularValues.array() > threshold).count();
}


void test_IDfactorization(const CMatrix & A, const std::vector<int> & rank, int seed){
     for(auto r : rank){
      IDResult id;
      bool success = true;
      try{
         id = MF::IDFactorization(A, r, seed);
      } catch(std::runtime_error& err){
         std::cout<< err.what() <<std::endl;
         success = false;
      }
        
      if(success) print_approximation_error(A, id, r);
   }
}

void test_IDfactorization(const CMatrix & A, int rank, int seed = -1){
   bool success = true;
   IDResult id;
   try{
      id = MF::IDFactorization(A, rank, seed);
   } catch(std::runtime_error& err){
      std::cout<< err.what() <<std::endl;
      success = false;
   }
   
   if(success) print_approximation_error(A, id, rank);   
}


void test_adaptiveIDFactorization(const CMatrix & A, const std::vector<double> & tols, int seed){
   for(auto tol : tols){
      IDResult id;
      bool success = true;

      try{
         id = MF::adaptiveIDFactorization(A, seed);
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
         id = MF::adaptiveIDFactorization(A, seed);
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
 
   const int rows = 150;
   const int cols = 140;
   const int size = 150; // for square matrix (size, size)
   double density = 0.9;
   int seed = 42;
   double tol = 0.2;
   int q = 2;
   int l = 140;
   int rank = 140;

/**
 * -----------  Building all type of matrices A
 **/ 

   Matrix sparse_A            = TestMat::randomSparseMatrix(rows,cols,density);
   Matrix hermitian_A         = TestMat::randomHermitianMatrix(size);
   Matrix psd_A               = TestMat::randomPositiveSemidefiniteMatrix(size);
   Matrix dense_A             = TestMat::randomDenseMatrix(rows, cols);
   
   Eigen::VectorXd sv = Eigen::VectorXd::Zero(std::min(rows, cols));
   for (int i = 0; i < rank; ++i) sv(i) = 1.0;

   Matrix singularValues_A    = TestMat::matrixWithSingularValues(rows, cols, sv);
/**
 * -----------  Building corrisponding matrices Q
 **/ 

   // Adaptive method to compute Q (NOT for sparse matrix)
   Matrix Q_psd               = ARRF::adaptivePowerIteration(psd_A, tol, rank, q, -1);
   Matrix Q_psd2              = ARRF::adaptiveRangeFinder(psd_A, tol, rank, -1);
   Matrix Q_singularValues    = ARRF::adaptivePowerIteration(singularValues_A, tol, rank, q, -1);
   Matrix Q_hermitian         = ARRF::adaptivePowerIteration(hermitian_A, tol, rank, q, -1);
   Matrix Q_dense             = ARRF::adaptiveRangeFinder(dense_A, tol, rank, -1);

   // Non adaptive method for sparse
   Matrix Q_sparse            = RRF::randomizedRangeFinder(sparse_A, rank, -1);


/*************************************
***** Test ID factorization
*************************************/

/**
 * ID tested on different A, knowing previusly its rank 
 *  */ 

   int ID_oversampling = 10;

   int rank_Q_dense = numerical_rank(Q_dense);
   std::cout<< "rank of dense Q : " << rank_Q_dense<<std::endl;
   test_IDfactorization(Q_dense, rank_Q_dense - ID_oversampling);

   int rank_hermitian_Q = numerical_rank(Q_hermitian);
   std::cout<< "rank of hermitian Q : " << rank_hermitian_Q<<std::endl;
   test_IDfactorization(Q_hermitian, rank_hermitian_Q - ID_oversampling);

   int rank_psd_Q = numerical_rank(Q_psd);
   std::cout<< "rank of psd Q : " << rank_psd_Q<<std::endl;
   test_IDfactorization(Q_psd, rank_psd_Q - ID_oversampling);

/**
 * Adaptive ID tested on different Q
 *  */ 

 double id_tol = 0.6;

   std::cout<< "-- Adaptive ID factorization --"<<std::endl;

   std::cout<< "- Q_dense -" <<std::endl;
   test_adaptiveIDFactorization(Q_dense, id_tol, seed); 

   std::cout<< "- Q_psd -" <<std::endl;
   test_adaptiveIDFactorization(Q_psd, id_tol, seed); 

   std::cout<< "- Q_psd-2 -" <<std::endl;
   test_adaptiveIDFactorization(Q_psd2, id_tol, seed);

   std::cout<< "- Q_sparse -" <<std::endl;
   test_adaptiveIDFactorization(Q_sparse, id_tol, seed);

   std::cout<< "- Q_Hermitian -" <<std::endl;
   test_adaptiveIDFactorization(Q_hermitian, id_tol, seed);


/************************************    
****** Test of Algorithms of Stage B 
************************************/ 

   std::cout << "\n--- Testing A = sparse_A, Q = Q_sparse ---" << std::endl;
   double err_sparse_A_Q_sparse = Err::realError(sparse_A, Q_sparse);
   std::cout << "||A - QQ^T A|| = " << err_sparse_A_Q_sparse << std::endl;

   test_directSVD(sparse_A, Q_sparse, tol);
   test_SVDviaRowExtraction(sparse_A, Q_sparse, tol);

   std::cout << "\n--- Testing A = hermitian_A, Q = Q_hermitian ---" << std::endl;
   double err_hermitian_A_Q_hermitian = Err::realError(hermitian_A, Q_hermitian);
   std::cout << "||A - QQ^T A|| = " << err_hermitian_A_Q_hermitian << std::endl;

   test_directSVD(hermitian_A, Q_hermitian, tol);
   test_SVDviaRowExtraction(hermitian_A, Q_hermitian, tol);
   test_directEigDecomposition(hermitian_A, Q_hermitian, tol);
   test_eigenvalueDecompositionViaRowExtraction(hermitian_A, Q_hermitian, tol);
   test_eigenvalueDecompositionInOnePass(hermitian_A, Q_hermitian, TestMat::randomDenseMatrix(size, size), tol);

   std::cout << "\n--- Testing A = psd_A, Q = Q_psd ---" << std::endl;
   double err_psd_A_Q_psd = Err::realError(psd_A, Q_psd);
   std::cout << "||A - QQ^T A|| = " << err_psd_A_Q_psd << std::endl;

   test_directSVD(psd_A, Q_psd, tol);
   test_SVDviaRowExtraction(psd_A, Q_psd, tol);
   test_eigenvalueDecompositionViaNystromMethod(psd_A, Q_psd, tol);

   std::cout << "\n--- Testing A = psd_A, Q = Q_psd2 ---" << std::endl;
   double err_psd_A_Q_psd2 = Err::realError(psd_A, Q_psd2);
   std::cout << "||A - QQ^T A|| = " << err_psd_A_Q_psd2 << std::endl;

   test_directSVD(psd_A, Q_psd2, tol);
   test_SVDviaRowExtraction(psd_A, Q_psd2, tol);
   test_eigenvalueDecompositionViaNystromMethod(psd_A, Q_psd2, tol);

   std::cout << "\n--- Testing A = singularValues_A, Q = Q_singularValues ---" << std::endl;
   double err_singularValues_A_Q_singularValues = Err::realError(singularValues_A, Q_singularValues);
   std::cout << "||A - QQ^T A|| = " << err_singularValues_A_Q_singularValues << std::endl;

   test_directSVD(singularValues_A, Q_singularValues, tol);
   test_SVDviaRowExtraction(singularValues_A, Q_singularValues, tol);

   std::cout << "\n--- Testing A = dense_A, Q = Q_dense ---" << std::endl;
   double err_dense = Err::realError(dense_A, Q_dense);
   std::cout << "||A - QQ^T A|| = " << err_dense << std::endl;

   test_directSVD(dense_A, Q_dense, tol);
   test_SVDviaRowExtraction(dense_A, Q_dense, 5.0);


   return 0;
}




#pragma once

#include <Eigen/Dense>
#include <randla/types.hpp>

namespace randla::algorithms {

/**
 * @brief Stage B algorithms: obtain factorizations from a computed range basis.
 *
 * This class groups routines that, given an (approximate) orthonormal basis Q for
 * the range of a matrix A, compute standard matrix factorizations.
 */
template<typename FloatType = double>
class MatrixFactorizer : public randla::Types<FloatType> {
	static_assert(std::is_floating_point_v<FloatType>, "FloatType must be a floating point type");
public:
	using typename randla::Types<FloatType>::Scalar;
	using typename randla::Types<FloatType>::Matrix;
	using typename randla::Types<FloatType>::CMatrix;
	using typename randla::Types<FloatType>::Vector;
	using typename randla::Types<FloatType>::SVDResult;
	using typename randla::Types<FloatType>::IDResult;

	/**
	 * @brief Direct SVD factorization (Algorithm: project then SVD) A ≈ U S V^T
	 * @param A  Input matrix (m x n)
	 * @param Q  Orthonormal basis (m x k) approximating range(A)
	 * @param tol If ||A - QQ^T A|| > tol throw (ensures Q captures A to desired precision)
	 * @return SVDResult containing U (m x k), singular values (k), V (n x k)
	 */
	static SVDResult directSVD(const Matrix & A, const Matrix & Q, double tol);

	/**
	 * Computes a randomized Interpolative Decomposition (ID) of a complex matrix A using Algorithm I 
	 * ( meant to be used when efficient procedures for applying the matrix A and its adjoint A*
	 * to arbitrary vectors are available )
	 *
	 * Given a complex input matrix A (m x n), this function computes an approximate factorization:
	 *
	 *     A ≈ B * P
	 *
	 * where:
	 *   - B (m x k) is formed by selecting k columns from A,
	 *   - P (k x n) is a coefficient matrix such that the leading k x k block is the identity,
	 *     and ideally no entry in P exceeds 2 in absolute value (not enforced in this version).
	 *
	 * The algorithm proceeds in three main steps:
	 *
	 * 1. Generate a random Gaussian matrix R (l x m), with l = k + oversampling,
	 *    where entries are sampled i.i.d. from the complex standard normal distribution.
	 * 
	 * 2. Form the sketch matrix Y = R * A (l x n), and perform a column-pivoted QR decomposition 
	 *    on Y^T to identify the k most informative columns. These define the index set used to extract B.
	 *
	 * 3. Solve a least-squares problem to compute P such that B * P ≈ A, i.e.:
	 *       P = B⁺ * A
	 *    using the Moore-Penrose pseudo-inverse of B computed via SVD.
	 *
	 * Note:
	 *   - The spectral error ||A - B*P||_2 is expected to be close to the (k+1)-th singular value of Y.
	 *   - The output does not explicitly enforce the |P_ij| ≤ 2 constraint.
	 *
	 * Parameters:
	 *   A    - Input complex matrix of size (m x n)
	 *   k    - Target rank for the decomposition (k < min(m, n))
	 *   seed - Random seed for reproducibility
	 *
	 * Returns:
	 *   An IDResult structure containing:
	 *     - B: (m x k) matrix of selected columns from A,
	 *     - P: (k x n) coefficient matrix,
	 *     - indices: vector of selected column indices
	 */
	static IDResult IDFactorizationI(const CMatrix & A, int rank, int seed);

	/**
	 * @param A
	 * @param Q
	 * @param tol
	 * @return SVDResult 
	 *  */ 
	static SVDResult SVDViaRowExtraction(const Matrix & A, const Matrix & Q, double tol);

	/**
	 * @brief Given an Hermiatian matrix A and a basis Q, this algorithm computes an aproximate 
	 * 		  eigenvalue decomposition A ≈ U Λ U^*
	 * @param A
	 * @param Q
	 * @param tol
	 * @return EigenvalueDecomposition
	 *  */ 
	static EigenvalueDecomposition directEigenvalueDecomposition(const Matrix & Hermitian_A, const Matrix & Q, double tol);
 
	static EigenvalueDecomposition EigenvalueDecompositionViaRowExtraction(const Matrix & Hermitian_A, const Matrix & Q, double tol);

	static EigenvalueDecomposition EigenvalueDecompositionViaNystromMethod(const Matrix & PSD_A, const Matrix & Q, double tol);

	static EigenvalueDecomposition EigenvalueDecompositionInOnePass(const Matrix & Hermitian_A, const Matrix & Q, const Matrix & Random_test_omega, double tol)
};


} // namespace randla::algorithms

#include "matrix_factorizer_impl.hpp"

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
	using typename randla::Types<FloatType>::EigenvalueDecomposition;

	/**
	 * @brief Direct SVD factorization (Algorithm: project then SVD) A ≈ U S V^T
	 * @param A  	- Input matrix (m x n)
	 * @param Q 	- Orthonormal basis (m x k) approximating range(A)
	 * @param tol 	- If ||A - QQ^T A|| > tol throw (ensures Q captures A to desired precision)
	 * @return 	SVDResult containing U (m x k), singular values (k), V (n x k)
	 */
	static SVDResult directSVD(const Matrix & A, const Matrix & Q, double tol);

	/**
	 * @brief
	 * Computes a randomized Interpolative Decomposition (ID) of a complex matrix A using Algorithm I 
	 * (meant to be used when efficient procedures for applying the matrix A and its adjoint A*
	 * to arbitrary vectors are available)
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
	 * @param A    - Input complex matrix of size (m x n)
	 * @param k    - Target rank for the decomposition (k < min(m, n))
	 * @param  seed - Random seed for reproducibility
	 * @return An IDResult structure containing:
	 *    	- B: (m x k) matrix of selected columns from A,
	 *     	- P: (k x n) coefficient matrix,
	 *     	- indices: vector of selected column indices
	 */
	static IDResult IDFactorizationI(const CMatrix & A, int rank, int seed);

	/**
	 * @brief 
	 * Computes an approximate truncated SVD of the input matrix A using Algorithm 5.2,
	 * which avoids computing matrix–matrix products by leveraging an interpolative decomposition of Q.
	 *
	 * Given:
	 *   - A ∈ ℝ^{m x n}, the target matrix,
	 *   - Q ∈ ℝ^{m x k}, an orthonormal basis such that A ≈ QQ^*A,
	 *
	 * this function produces an approximate factorization:
	 *
	 *     A ≈ U Σ V^*
	 *
	 * where U and V are orthonormal matrices and Σ is a nonnegative diagonal matrix.
	 *
	 * The algorithm proceeds as follows:
	 *
	 * 1. Perform an interpolative decomposition of the rows of Q by applying ID to Q^T, obtaining:
	 *        Q ≈ X Q(J, :)
	 *
	 * 2. Extract the corresponding rows A(J, :) and compute a QR factorization:
	 *        A(J, :) = R^* W^*
	 *
	 * 3. Form Z = X R^*, then compute the SVD:
	 *        Z = U Σ V̂^*
	 *
	 * 4. Set V = W V̂ to complete the factorization A ≈ U Σ V^*.
	 *
	 * Note:
	 *   - This method is significantly faster than full SVD postprocessing (e.g., Algorithm 5.1),
	 *     but may result in slightly reduced accuracy, since the error ||A - XB|| is usually larger 
	 * 	   than the initial error ||A - QQ^*A||, especially when the dimension of A is large.
	 * 
	 * @param  A   - Input matrix (m x n)
	 * @param  Q   - Orthonormal basis matrix (m x k)
	 * @param tol - Tolerance for accepting Q as a valid basis for the range of A
	 * @return SVDResult structure
	 */
	static SVDResult SVDViaRowExtraction(const Matrix & A, const Matrix & Q, double tol);

	/**
	 * @brief
	 * Computes an approximate eigenvalue decomposition of a Hermitian matrix A using Algorithm 5.3.
	 * This method performs a projection of A onto a low-dimensional subspace defined by an orthonormal basis Q.
	 *
	 * Given:
	 *   - A ∈ ℝ^{n x n}, a Hermitian matrix,
	 *   - Q ∈ ℝ^{n x k}, an orthonormal matrix such that A ≈ QQ^*AQQ^*,
	 *
	 * this function computes an approximate factorization:
	 *
	 *     A ≈ U Λ U^*
	 *
	 * where:
	 *   - Λ is a real diagonal matrix of eigenvalues,
	 *   - U contains the approximate eigenvectors of A as U = QV,
	 *     with V being the eigenvectors of the projected matrix B = Q^* A Q.
	 *
	 * The algorithm proceeds as follows:
	 *
	 * 1. Project A onto the subspace spanned by Q: B = Q^* A Q
	 * 2. Compute the eigenvalue decomposition of B: B = V Λ V^*
	 * 3. Lift the eigenvectors back to the original space: U = Q V
	 *
	 * Note:
	 *   - This method is accurate when Q well approximates both the row and column space of A.
	 *   - The spectral error is bounded by ||A - U Λ U^*|| ≤ 2ε, with ε = ||A - QQ^*A||.
	 * 
	 * @param Hermitian_A   - Hermitian input matrix (n x n)
	 * @param Q   - Orthonormal basis matrix (n x k)
	 * @param tol - Tolerance for validating the quality of Q
	 * @return EigenvalueDecomposition structure
	 */
	static EigenvalueDecomposition directEigenvalueDecomposition(const Matrix & Hermitian_A, const Matrix & Q, double tol);
 
	/**
	 * @brief
	 * Computes an approximate eigenvalue decomposition of a Hermitian matrix A using Algorithm 5.4,
	 * which leverages row extraction and interpolative decomposition to reduce computational cost.
	 *
	 * Given:
	 *   - A ∈ ℝ^{n x n}, a Hermitian matrix,
	 *   - Q ∈ ℝ^{n x k}, an orthonormal basis such that A ≈ QQ^*AQQ^*,
	 *
	 * this function computes an approximate factorization:
	 *
	 *     A ≈ U Λ U^*
	 *
	 * where:
	 *   - Λ is a real diagonal matrix of eigenvalues,
	 *   - U is an orthonormal matrix of approximate eigenvectors, built without explicitly projecting A.
	 *
	 * The algorithm proceeds as follows:
	 *
	 * 1. Perform an interpolative decomposition of the rows of Q: Q ≈ X Q(J, :)
	 * 2. Compute a QR factorization of X: X = V R
	 * 3. Construct a small core matrix: Z = R A(J,J) R^*
	 * 4. Compute the eigenvalue decomposition of Z: Z = W Λ W^*
	 * 5. Form the approximate eigenvector matrix: U = V W
	 *
	 * Note:
	 *   - This method avoids computing Q^* A Q explicitly, and is significantly faster than Algorithm 5.3.
	 *   - The trade-off is reduced accuracy, especially when the ID step does not capture the dominant structure well.
	 *
	 * @param Hermitian_A   - Hermitian input matrix (n x n)
	 * @param Q   - Orthonormal basis matrix (n x k)
	 * @param tol - Tolerance for validating the quality of Q
	 * @return EigenvalueDecomposition structure 
	 */
	static EigenvalueDecomposition EigenvalueDecompositionViaRowExtraction(const Matrix & Hermitian_A, const Matrix & Q, double tol);

	/**
	 * @brief
	 * Computes an approximate eigenvalue decomposition of a positive semidefinite matrix A
	 * using the Nyström method (Algorithm 5.5), which exploits the PSD structure of A to improve accuracy.
	 *
	 * Given:
	 *   - A ∈ ℝ^{n x n}, a positive semidefinite matrix,
	 *   - Q ∈ ℝ^{n x k}, an orthonormal matrix such that A ≈ QQ^*AQQ^*,
	 *
	 * this function computes an approximate factorization:
	 *
	 *     A ≈ U Λ U^*
	 *
	 * where:
	 *   - Λ is a real diagonal matrix with nonnegative entries,
	 *   - U is an orthonormal matrix of approximate eigenvectors.
	 *
	 * The algorithm proceeds as follows:
	 *
	 * 1. Form B_1 = A Q and B_2 = Q^* A Q
	 * 2. Compute the Cholesky factorization B_2 = C C^*
	 * 3. Solve for F = B_1 C^{-1} using a triangular system
	 * 4. Perform SVD of F = U Σ V^*, then set Λ = Σ^2
	 *
	 * Note:
	 *   - This method is more accurate than direct projection (e.g., Algorithm 5.3),
	 *     especially when A is PSD, and requires only one pass over A.
	 *   - The approximation preserves the symmetric structure of A via a factorization A ≈ FF^*.
	 *
	 * @param PSD_A   - Positive semidefinite input matrix (n x n)
	 * @param Q   - Orthonormal basis matrix (n x k)
	 * @param tol - Tolerance for validating the quality of Q
	 * @return EigenvalueDecomposition structure
	 */
	static EigenvalueDecomposition EigenvalueDecompositionViaNystromMethod(const Matrix & PSD_A, const Matrix & Q, double tol);

	/**
	 * @brief
	 * Computes an approximate eigenvalue decomposition of a Hermitian matrix A using Algorithm 5.6,
	 * which enables a single-pass approximation suitable for memory-constrained or streaming settings.
	 *
	 * Given:
	 *   - A ∈ ℝ^{n x n}, a Hermitian matrix,
	 *   - Ω ∈ ℝ^{n x l}, a random test matrix,
	 *   - Q ∈ ℝ^{n x k}, an orthonormal basis such that A ≈ QQ^*AQQ^* and Y = QQ^*Y with Y = AΩ,
	 *
	 * this function computes an approximate factorization:
	 *
	 *     A ≈ U Λ U^*
	 *
	 * where:
	 *   - Λ is a real diagonal matrix of eigenvalues,
	 *   - U is an orthonormal matrix of approximate eigenvectors.
	 *
	 * The algorithm proceeds as follows:
	 *
	 * 1. Use a least-squares solver to estimate a Hermitian matrix B_approx satisfying:
	 *        B_approx (Q^* Ω) ≈ Q^* Y
	 *
	 * 2. Compute the eigendecomposition: B_approx = V Λ V^*
	 * 3. Form the approximate eigenvector matrix: U = Q V
	 *
	 * Note:
	 *   - This method requires only one pass over A to compute Y = A Ω.
	 *   - It is well suited for very large matrices, but may suffer from numerical instability if Q^* Ω is ill-conditioned.
	 *   - Oversampling (l > k) is recommended to improve stability.
	 *
	 * @param Hermitian_A       - Hermitian input matrix (n x n)
	 * @param Q       - Orthonormal basis matrix (n x k)
	 * @param Omega   - Random test matrix used to build the sketch Y = A Ω
	 * @param tol     - Tolerance for validating the residual ||A - QQ^*A||
	 * @return EigenvalueDecomposition structure
	 */
	static EigenvalueDecomposition EigenvalueDecompositionInOnePass(const Matrix & Hermitian_A, const Matrix & Q, const Matrix & Omega, double tol);
};


} // namespace randla::algorithms

#include "matrix_factorizer_impl.hpp"

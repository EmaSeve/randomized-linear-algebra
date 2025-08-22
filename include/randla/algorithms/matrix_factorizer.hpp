#pragma once

#include <Eigen/Dense>
#include <randla/types.hpp>
#include <iostream>

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
	using typename randla::Types<FloatType>::Vector;
	using typename randla::Types<FloatType>::CVector;
	using typename randla::Types<FloatType>::Matrix;
	using typename randla::Types<FloatType>::CMatrix;
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
	template<class MatLike>
	static SVDResult directSVD(const MatLike & A, const MatLike & Q, double tol){

		double error = randla::metrics::ErrorEstimators<FloatType>::realError(A, Q);
		if (error > tol)
			throw std::runtime_error("MatrixFactorizer - direct SVD: residual norm exceeds tolerance");

		Matrix B = Q.transpose() * A; // k x n

		Eigen::JacobiSVD<Matrix> svd(B, Eigen::ComputeThinU | Eigen::ComputeThinV);
		Matrix U_tilde = svd.matrixU();
		Vector S = svd.singularValues();
		Matrix V = svd.matrixV();

		Matrix U = Q * U_tilde; // m x k

		return SVDResult{std::move(U), std::move(S), std::move(V)};
	}

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
	static IDResult IDFactorization(const CMatrix & A, int rank, int seed){

		const size_t m = A.rows();
		const size_t n = A.cols();
		const size_t l = rank + 10;

		if (rank < 1 || rank > std::min(m, n))
			throw std::runtime_error("MatrixFactorizer - IDFactorization: wrong rank value");

		// Step 1:
		auto gen = randla::random::RandomGenerator<FloatType>::make_generator(seed);
		CMatrix R = randla::random::RandomGenerator<FloatType>::randomComplexGaussianMatrix(l, m, gen); 
		CMatrix Y = R * A;                                   

		// Step 2:
		Eigen::ColPivHouseholderQR<CMatrix> qr(Y.transpose());
		Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> perm = qr.colsPermutation();

		Vector indices = perm.indices().template cast<Scalar>();
		std::vector<int> selectedCols(rank);
		for (size_t i = 0; i < rank; ++i)
			selectedCols[i] = indices(i);

		CMatrix B(m, rank);
		for (size_t i = 0; i < rank; ++i)
			B.col(i) = A.col(selectedCols[i]);

		// Step 3:
		CMatrix P;
		Eigen::BDCSVD<CMatrix> svd(B, Eigen::ComputeThinU | Eigen::ComputeThinV);
		P = svd.solve(A);

		return IDResult{std::move(B), std::move(P), std::move(selectedCols)};
	}

	/**
	 * @brief 
	 * Computes an approximate Interpolative Decomposition (ID) of a complex matrix A using
	 * an adaptive rank selection strategy based on spectral norm error estimation.
	 *
	 * This method performs multiple calls to the standard ID (Algorithm I), starting from a small
	 * target rank k and doubling it until the approximation error ||A - B P||_2 / ||A||_2 
	 * falls below a user-specified relative tolerance.
	 *
	 * At each iteration:
	 *   1. A rank-k interpolative decomposition A ≈ B * P is computed.
	 *   2. The residual E = A - B * P is formed.
	 *   3. The spectral norm of the residual is estimated via power iteration.
	 *   4. If the relative error (||E||_2 / ||A||_2) is below tol, the current approximation is returned.
	 *
	 * The method guarantees a good approximation without requiring the rank a priori,
	 * but it does not reuse intermediate computations across iterations (e.g., R*A),
	 * which can make it more expensive than necessary in practice. !!
	 *
	 * Note:
	 *   - The stopping criterion is based on the estimated relative spectral norm error.
	 *   - If no acceptable approximation is found within rank ≤ min(m, n), the function throws.
	 * 
	 * @param A    - Input complex matrix (m x n)
	 * @param tol  - Relative spectral error tolerance (e.g., 1e-1 for 10%)
	 * @param seed - Seed for random vector generation (used in power iteration)
	 * @return  IDResult
	 */
	static IDResult adaptiveIDFactorization(const CMatrix & A, double tol, int seed){

		const size_t m = A.rows();
		const size_t n = A.cols();
		int k = 10;
		const int k_max = std::min(m, n);

		FloatType norm_A = randla::metrics::ErrorEstimators<FloatType>::estimateSpectralNorm(A, seed);

		while(true){
			// Run standard ID with current k
			IDResult result = IDFactorization(A, k, seed);

			// Compute residual matrix: E = A - B * P
			CMatrix A_approx = result.B * result.P;
			CMatrix E = A - A_approx;

			FloatType spectral_norm = randla::metrics::ErrorEstimators<FloatType>::estimateSpectralNorm(E, seed);

			FloatType relative_error = spectral_norm / norm_A;
			if(relative_error < tol)
				return result;

			// Double the rank for next iteration
			k *= 2;
			if(k > k_max)
				break;
		}

		throw std::runtime_error("MatrixFactorizer::adaptiveIDFactorization: failed to reach tolerance within allowed rank.");
	}

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
	template<class MatLike>
	static SVDResult SVDViaRowExtraction(const MatLike & A, const MatLike & Q, double tol){
		
		double error = randla::metrics::ErrorEstimators<FloatType>::realError(A, Q);
		if (error > tol) 
			throw std::runtime_error("MatrixFactorizer - SVD via row extraction: residual norm exceeds tolerance");

		const size_t m = A.rows();
		const size_t n = A.cols();

		// Step 1: ID of Q's rows, performing and ID on Q^T
		double id_tol = 0.5 * tol;
		int seed = 42;
		IDResult ID;
		try{
			ID = adaptiveIDFactorization(Q.transpose(), id_tol, seed);
		} catch(const std::runtime_error& err){
			std::cerr << "Adaptive ID failed: " << err.what() << std::endl;
			throw;
		}

		// Q ≈ X * Q(J, :)  ->  X = ID.P^T, Q_J = ID.B^T
		Matrix Q_J = ID.B.transpose().real();      // (k x n)
		Matrix X   = ID.P.transpose().real();      // (m x k)

		// Step 2: extract the corresponding rows of A
		const auto& row_indices = ID.indices;
		const int k = static_cast<int>(row_indices.size());
		Matrix A_J(k, n); 
		for (size_t i = 0; i < k; ++i)
			A_J.row(i) = A.row(row_indices[i]);

		// QR of A(J, :) = R * W^*
		Eigen::HouseholderQR<Matrix> qr(A_J);
		Matrix R = qr.matrixQR().topLeftCorner(k, k).template triangularView<Eigen::Upper>(); 
		Matrix W = qr.householderQ();                            

		// Step 3: Z = X * R^*
		Matrix Z = X * R.adjoint();

		// Step 4: SVD of Z
		Eigen::BDCSVD<Matrix> svd(Z, Eigen::ComputeThinU | Eigen::ComputeThinV);
		Matrix U = svd.matrixU();                       
		Vector S = svd.singularValues();                 
		Matrix Vhat = svd.matrixV();                     

		// Step 5: V = W * V̂
		Matrix V = W * Vhat;

		return SVDResult{std::move(U), std::move(S), std::move(V)};
	}

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
	template<class MatLike>
	static EigenvalueDecomposition directEigenvalueDecomposition(const MatLike & Hermitian_A, const MatLike & Q, double tol){
		
		// Alias of Hermitian_A
		const auto& A = Hermitian_A;

		double error = randla::metrics::ErrorEstimators<FloatType>::realError(A, Q);
        if (error > tol) 
            throw std::runtime_error("MatrixFactorizer - direct eigenvalue decomposition: residual norm exceeds tolerance");

        bool isHermitian = A.isApprox(A.adjoint());
        if(!isHermitian)
            throw std::runtime_error("MatrixFactorizer - direct eigenvalue decomposition: matrix A is NOT Hermitian");

        // Step 1: B = Q^* A Q
        Matrix B = Q.adjoint() * A * Q;

        // Step 2: eigenvalue decomposition of B
        Eigen::SelfAdjointEigenSolver<Matrix> eigensolver(B);
        if (eigensolver.info() != Eigen::Success)
            throw std::runtime_error("MatrixFactorizer - direct Eigenvalue decomposition: eigen decomposition failed");

        Vector eigenvalues = eigensolver.eigenvalues();
        Matrix Lambda = eigenvalues.asDiagonal();
        Matrix V = eigensolver.eigenvectors();            

        // Step 3: U = Q V
        Matrix U = Q * V;

        return EigenvalueDecomposition{std::move(U), std::move(Lambda)};
	}
 
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
	template<class MatLike>
	static EigenvalueDecomposition EigenvalueDecompositionViaRowExtraction(const MatLike & Hermitian_A, const MatLike & Q, double tol){
		
		// Alias of Hermitian_A
		const auto& A = Hermitian_A;

		bool isHermitian = A.isApprox(A.adjoint());
		if(!isHermitian)
			throw std::runtime_error("MatrixFactorizer - direct eigenvalue decomposition: matrix A is NOT Hermitian");

		double error = randla::metrics::ErrorEstimators<FloatType>::realError(A, Q);
		if (error > tol) 
			throw std::runtime_error("MatrixFactorizer - direct eigenvalue decomposition: residual norm exceeds tolerance");

		const size_t m = A.rows();
		const size_t n = A.cols();

		// Step 1: Interpolative Decomposition over Q rows
		double id_tol = 0.5 * tol;
		int seed = 42;
		IDResult ID;
		try{
			ID = adaptiveIDFactorization(Q.transpose(), id_tol, seed);
		} catch(const std::runtime_error& err){
			std::cerr << "Adaptive ID failed: " << err.what() << std::endl;
			throw;
		}

		Matrix Q_J = ID.B.transpose().real();  
		Matrix X = ID.P.transpose().real();   

		const auto& row_indices = ID.indices;
		const int k = static_cast<int>(row_indices.size());

		// Step 2: QR of X = V R
		Eigen::HouseholderQR<Matrix> qr_X(X);
		Matrix R = qr_X.matrixQR().topLeftCorner(k, k).template triangularView<Eigen::Upper>();
		Matrix V = qr_X.householderQ(); 

		// Step 3: Z = R A(J,J) R^*
		Matrix A_JJ(k, k);
		for (size_t i = 0; i < k; ++i)
			for (size_t j = 0; j < k; ++j)
				A_JJ(i, j) = A(row_indices[i], row_indices[j]);

		Matrix Z = R * A_JJ * R.adjoint();

		// Step 4: eigenvalue decomposition of Z
		Eigen::SelfAdjointEigenSolver<Matrix> eig(Z);
		if (eig.info() != Eigen::Success)
			throw std::runtime_error("MatrixFactorizer::EigenvalueDecompositionViaRowExtraction: eigen decomposition failed");

		Vector eigenvalues = eig.eigenvalues(); 
		Matrix Lambda = eigenvalues.asDiagonal();
		Matrix W = eig.eigenvectors();           

		// Step 5: U = V W
		Matrix U = V.leftCols(k) * W;

		return EigenvalueDecomposition{std::move(U), std::move(Lambda)};
	}

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
	template<class MatLike>
	static EigenvalueDecomposition EigenvalueDecompositionViaNystromMethod(const MatLike & PSD_A, const MatLike & Q, double tol){
		
		// Alias of PSD_A
		const auto& A = PSD_A;

		const size_t m = A.rows();
		const size_t k = Q.cols();

		double error = randla::metrics::ErrorEstimators<FloatType>::realError(A, Q);
		if (error > tol)
			throw std::runtime_error("MatrixFactorizer::EigenvalueDecompositionViaNystromMethod: residual error exceeds tolerance");

		Eigen::SelfAdjointEigenSolver<Matrix> eig(A);
		if (eig.eigenvalues().minCoeff() < -1e-10)
			throw std::runtime_error("MatrixFactorizer::EigenvalueDecompositionViaNystromMethod: matrix A is not positive semidefinite");

		// Step 1: B_1 = A Q ; B_2 = Q* A Q
		Matrix B1 = A * Q;
		Matrix B2 = Q.adjoint() * B1;

		// Step 2: Compute Cholesky factorization B_2 = C C*
		Eigen::LLT<Matrix> chol(B2);
		if (chol.info() != Eigen::Success)
			throw std::runtime_error("MatrixFactorizer::EigenvalueDecompositionViaNystromMethod: Cholesky decomposition failed");
		Matrix C = chol.matrixL();

		// Step 3: Solve triangular system F = B_1 * C^{-1}
		Matrix F = C.template triangularView<Eigen::Lower>().solve(B1.transpose()).transpose();

		// Step 4: Compute SVD F = U Σ V* and set Λ = Σ^2
		Eigen::BDCSVD<Matrix> svd(F, Eigen::ComputeThinU | Eigen::ComputeThinV);
		Matrix U = svd.matrixU();
		Vector Sigma = svd.singularValues();
		Vector eigenvalues_square = Sigma.array().square();
		Matrix Lambda = eigenvalues_square.asDiagonal();

		return EigenvalueDecomposition{std::move(U), std::move(Lambda)};
	}

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
	template<class MatLike>
	static EigenvalueDecomposition EigenvalueDecompositionInOnePass(const MatLike & Hermitian_A, const MatLike & Q, const MatLike & Omega, double tol){
		
		// Alias of Hermitian_A
		const auto& A = Hermitian_A;

		bool isHermitian = A.isApprox(A.adjoint());
		if(!isHermitian)
			throw std::runtime_error("MatrixFactorizer - eigenvalue decomposition in one pass: matrix A is NOT Hermitian");

		double error = randla::metrics::ErrorEstimators<FloatType>::realError(A, Q);
		if (error > tol) 
			throw std::runtime_error("MatrixFactorizer - eigenvalue decomposition in one pass: residual norm exceeds tolerance");

		// Step 1: Form the sample matrix Y = A * Omega
		Matrix Y = A * Omega;

		// Step 2: Solve least squares: Bapprox * (Q^* * Omega) ≈ Q^* * Y
		Matrix Q_omega = Q.adjoint() * Omega;
		Matrix Q_Y     = Q.adjoint() * Y;

		// Solve Bapprox from: B Q_omega ≈ Q_Y
		// This is an overdetermined system if Omega has more columns than Q has rows
		Matrix Bapprox;
		Bapprox = Q_Y * Q_omega.completeOrthogonalDecomposition().solve(Matrix::Identity(Q_omega.cols(), Q_omega.cols()));

		// Step 3: Eigendecomposition of Bapprox
		Eigen::SelfAdjointEigenSolver<Matrix> eigensolver(Bapprox);
		if (eigensolver.info() != Eigen::Success)
			throw std::runtime_error("MatrixFactorizer::EigenvalueDecompositionInOnePass: eigen decomposition failed");

		Vector eigenvalues = eigensolver.eigenvalues();
		Matrix Lambda = eigenvalues.asDiagonal();
		Matrix V = eigensolver.eigenvectors();

		// Step 4: Form U = Q * V
		Matrix U = Q * V;

		return EigenvalueDecomposition{std::move(U), std::move(Lambda)};
	}
};

} // namespace randla::algorithms


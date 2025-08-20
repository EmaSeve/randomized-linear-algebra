#pragma once




namespace randla::algorithms {

template<typename FloatType>
typename MatrixFactorizer<FloatType>::SVDResult
MatrixFactorizer<FloatType>::directSVD(const Matrix & A, const Matrix & Q, double tol) {

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


template<typename FloatType>
typename MatrixFactorizer<FloatType>::IDResult
MatrixFactorizer<FloatType>::IDFactorizationI(const CMatrix & A, int k, int seed){

    const size_t m = A.rows();
    const size_t n = A.cols();
    const size_t l = k + 10;

    if (k < 1 || k > std::min(m, n))
        throw std::runtime_error("MatrixFactorizer - IDFactorizationI: wrong rank value");

    // Step 1:
    auto gen = randla::random::RandomGenerator<FloatType>::make_generator(seed);
    CMatrix R = randla::random::RandomGenerator<FloatType>::randomComplexGaussianMatrix(l, m, gen); 
    CMatrix Y = R * A;                                   

    // Step 2:
    Eigen::ColPivHouseholderQR<CMatrix> qr(Y.transpose());
    Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> perm = qr.colsPermutation();

    Vector indices = perm.indices().template cast<Scalar>();
    std::vector<int> selectedCols(k);
    for (size_t i = 0; i < k; ++i)
        selectedCols[i] = indices(i);

    CMatrix B(m, k);
    for (size_t i = 0; i < k; ++i)
        B.col(i) = A.col(selectedCols[i]);

    // Step 3:
    CMatrix P;
    Eigen::BDCSVD<CMatrix> svd(B, Eigen::ComputeThinU | Eigen::ComputeThinV);
    P = svd.solve(A);

    return IDResult{std::move(B), std::move(P), std::move(selectedCols)};
}

template<typename FloatType>
typename MatrixFactorizer<FloatType>::IDResult
MatrixFactorizer<FloatType>::adaptiveIDFactorization(const CMatrix & A, double tol, int seed){

    const size_t m = A.rows();
    const size_t n = A.cols();
    int k = 10;
    const int k_max = std::min(m, n);

    FloatType norm_A = randla::metrics::ErrorEstimators<FloatType>::estimateSpectralNorm(A, seed);

    while(true){
        // Run standard ID with current k
        IDResult result = IDFactorizationI(A, k, seed);

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

template<typename FloatType>
typename MatrixFactorizer<FloatType>::SVDResult
MatrixFactorizer<FloatType>::SVDViaRowExtraction(const Matrix & A, const Matrix & Q, double tol){
    
    double error = randla::metrics::ErrorEstimators<FloatType>::realError(A, Q);
	if (error > tol) 
        throw std::runtime_error("MatrixFactorizer - SVD via row extraction: residual norm exceeds tolerance");

    const size_t m = A.rows();
    const size_t n = A.cols();

// TODO: change the IDFactorizationI such that rank will not be a parameter anymore, the function need to become 'adaptive'

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

template<typename FloatType>
typename MatrixFactorizer<FloatType>::EigenvalueDecomposition
MatrixFactorizer<FloatType>::directEigenvalueDecomposition(const Matrix & A, const Matrix & Q, double tol){

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

template<typename FloatType>
typename MatrixFactorizer<FloatType>::EigenvalueDecomposition
MatrixFactorizer<FloatType>::EigenvalueDecompositionViaRowExtraction(const Matrix & A, const Matrix & Q, double tol){

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

template<typename FloatType>
typename MatrixFactorizer<FloatType>::EigenvalueDecomposition
MatrixFactorizer<FloatType>::EigenvalueDecompositionViaNystromMethod(const Matrix & A, const Matrix & Q, double tol){
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

template<typename FloatType>
typename MatrixFactorizer<FloatType>::EigenvalueDecomposition
MatrixFactorizer<FloatType>::EigenvalueDecompositionInOnePass(const Matrix & A, const Matrix & Q, const Matrix & Omega, double tol){

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

} // namespace randla::algorithms

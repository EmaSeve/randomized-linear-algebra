#pragma once

#include <stdexcept>
#include <Eigen/Dense>
#include <randla/metrics/error_estimators.hpp>

namespace randla::algorithms {

template<typename FloatType>
typename MatrixFactorizer<FloatType>::DirectSVDResult
MatrixFactorizer<FloatType>::directSVD(const Matrix & A, const Matrix & Q, double tol) {
	// reuse error computation from RandomizedRangeFinder (static dispatch)
	double error = ErrorEstimators<FloatType>::realError(A, Q);
	if (error > tol) {
		throw std::runtime_error("MatrixFactorizer::directSVD: residual norm exceeds tolerance");
	}

	Matrix B = Q.transpose() * A; // k x n

	Eigen::JacobiSVD<Matrix> svd(B, Eigen::ComputeThinU | Eigen::ComputeThinV);
	Matrix U_tilde = svd.matrixU();
	Vector S = svd.singularValues();
	Matrix V = svd.matrixV();

	Matrix U = Q * U_tilde; // m x k

	return DirectSVDResult{std::move(U), std::move(S), std::move(V)};
}

} // namespace randla::algorithms

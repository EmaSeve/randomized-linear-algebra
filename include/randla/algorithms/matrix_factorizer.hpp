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
	using typename randla::Types<FloatType>::Vector;
	using typename randla::Types<FloatType>::DirectSVDResult;

	/**
	 * @brief Direct SVD factorization (Algorithm: project then SVD) A ≈ U S V^T
	 * @param A  Input matrix (m x n)
	 * @param Q  Orthonormal basis (m x k) approximating range(A)
	 * @param tol If ||A - QQ^T A|| > tol throw (ensures Q captures A to desired precision)
	 * @return DirectSVDResult containing U (m x k), singular values (k), V (n x k)
	 */
	static DirectSVDResult directSVD(const Matrix & A, const Matrix & Q, double tol);
};

} // namespace randla::algorithms

#include "matrix_factorizer_impl.hpp"

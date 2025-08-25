#pragma once

#include <Eigen/Core>
#include <thread>


namespace randla::threading {

/**
 * @brief C interface for setting and getting the number of threads in OpenBLAS.
 */
extern "C" {
    void openblas_set_num_threads(int);
    int  openblas_get_num_threads(void);
}

/**
 * @brief Set the number of threads used by Eigen or BLAS backends.
 * 
 * Sets the number of threads for Eigen (if compiled with OpenMP support)
 * or for OpenBLAS (if using BLAS backend).
 * 
 * @param num_threads Number of threads to use.
 */
inline void setThreads(int num_threads) {
#ifdef EIGEN_USE_OPENMP
    Eigen::setNbThreads(num_threads);
#elif defined(EIGEN_USE_BLAS)
    openblas_set_num_threads(num_threads);
#endif
}

/**
 * @brief Get the number of threads used by Eigen or BLAS backends.
 * 
 * Returns the number of threads currently set for Eigen (if compiled with OpenMP support)
 * or for OpenBLAS (if using BLAS backend). Defaults to 1 if neither is available.
 * 
 * @return Number of threads in use.
 */
inline int getThreads() {
#ifdef EIGEN_USE_OPENMP
    return Eigen::nbThreads();
#elif defined(EIGEN_USE_BLAS)
    return openblas_get_num_threads();
#else
    return 1;
#endif
}

} // namespace randla::threading

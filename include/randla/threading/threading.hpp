#pragma once

#include <Eigen/Core>
#include <thread>


namespace randla::threading {

extern "C" {
    void openblas_set_num_threads(int);
    int  openblas_get_num_threads(void);
}

inline void setThreads(int num_threads) {
#ifdef EIGEN_USE_OPENMP
    Eigen::setNbThreads(num_threads);
#elif defined(EIGEN_USE_BLAS)
    openblas_set_num_threads(num_threads);
#endif
}

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

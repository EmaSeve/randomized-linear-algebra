#pragma once

#include <Eigen/Core>

#ifdef RRF_USE_OPENMP
#include <omp.h>
#endif

namespace randla::threading {

inline void setThreads(int num_threads) {
    // Set Eigen threads
    Eigen::setNbThreads(num_threads);
    
#ifdef RRF_USE_OPENMP
    // Set OpenMP threads
    omp_set_num_threads(num_threads);
#endif
}

inline int getMaxThreads() {
#ifdef RRF_USE_OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

inline int getCurrentThreads() {
    return Eigen::nbThreads();
}

inline bool isOpenMPEnabled() {
#ifdef RRF_USE_OPENMP
    return true;
#else
    return false;
#endif
}

} // namespace randla::threading

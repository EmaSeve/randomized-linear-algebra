#pragma once

#include <Eigen/Core>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace randla {
namespace threading {

inline void setThreads(int n) {
#ifdef _OPENMP
    omp_set_num_threads(n);
    // omp_set_nested(0);
#endif
    Eigen::setNbThreads(n);
}

inline int getThreads() {
    return Eigen::nbThreads();
}

} // namespace threading
} // namespace randla

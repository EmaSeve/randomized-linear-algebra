#pragma once

#include <Eigen/Core>
#include <thread>

namespace randla::threading {

inline void setThreads(int num_threads) {
    // Control threading only via Eigen
    Eigen::setNbThreads(num_threads);
}

} // namespace randla::threading

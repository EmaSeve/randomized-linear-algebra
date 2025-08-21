#pragma once

#include <Eigen/Core>
#include <thread>

namespace randla::threading {

inline void setThreads(int num_threads) {
    Eigen::setNbThreads(num_threads);
}

} // namespace randla::threading

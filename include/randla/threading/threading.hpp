namespace randla {
namespace threading {

inline void setThreads(int n) {
    #ifdef _OPENMP
    omp_set_num_threads(n);
    #endif
    Eigen::setNbThreads(n);
    Eigen::setNestedParallelism(false);
}

inline int getThreads() {
    return Eigen::nbThreads();
}

} // namespace threading
} // namespace randla

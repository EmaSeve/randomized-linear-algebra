cmake_minimum_required(VERSION 3.15)
project(randomized-linear-algebra VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ========================
# Eigen3 (header-only)
# ========================
# Prefer the modern CMake package (provided by Eigen installation)
find_package(Eigen3 QUIET NO_MODULE)
if(NOT TARGET Eigen3::Eigen)
    # Fallback: header-only discovery via include path
    find_path(EIGEN3_INCLUDE_DIR
        NAMES Eigen/Core
        HINTS ENV CPATH
        PATH_SUFFIXES Eigen eigen3
    )
    if(NOT EIGEN3_INCLUDE_DIR)
        message(FATAL_ERROR "Impossibile trovare le intestazioni Eigen3. Imposta EIGEN3_INCLUDE_DIR o installa Eigen (es. libeigen3-dev).")
    endif()
    add_library(Eigen3::Eigen INTERFACE IMPORTED)
    target_include_directories(Eigen3::Eigen INTERFACE ${EIGEN3_INCLUDE_DIR})
endif()

# ========================
# FFTW (double precision)
# ========================
# Try pkg-config first (common on Linux)
find_package(PkgConfig QUIET)
set(_have_fftw_target OFF)
if(PKG_CONFIG_FOUND)
    pkg_check_modules(FFTW3 QUIET IMPORTED_TARGET fftw3)
    if(TARGET PkgConfig::FFTW3)
        add_library(FFTW::Double INTERFACE IMPORTED)
        target_link_libraries(FFTW::Double INTERFACE PkgConfig::FFTW3)
        set(_have_fftw_target ON)
    endif()
endif()

if(NOT _have_fftw_target)
    # Fallback: manual discovery via include/library paths (double precision)
    find_path(FFTW3_INCLUDE_DIR
        NAMES fftw3.h
        HINTS ENV CPATH
    )
    find_library(FFTW3_LIB
        NAMES fftw3
        HINTS ENV LIBRARY_PATH ENV LD_LIBRARY_PATH
    )
    if(NOT FFTW3_INCLUDE_DIR OR NOT FFTW3_LIB)
        message(FATAL_ERROR "Impossibile trovare FFTW3 (double). Installa libfftw3-dev oppure imposta FFTW3_INCLUDE_DIR e FFTW3_LIB.")
    endif()
    add_library(FFTW::Double INTERFACE IMPORTED)
    target_include_directories(FFTW::Double INTERFACE ${FFTW3_INCLUDE_DIR})
    target_link_libraries(FFTW::Double INTERFACE ${FFTW3_LIB})
endif()

# ========================
# OpenMP (optional)
# ========================
option(ENABLE_OPENMP "Enable OpenMP parallelism" ON)
set(HAVE_OPENMP OFF)
if(ENABLE_OPENMP)
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        set(HAVE_OPENMP ON)
    else()
        message(STATUS "OpenMP non trovato: compilazione senza parallelismo OpenMP")
    endif()
endif()

# ========================
# randomized-linear-algebra (header-only)
# ========================
add_library(randomized-linear-algebra INTERFACE)
target_include_directories(randomized-linear-algebra INTERFACE include)
target_link_libraries(randomized-linear-algebra
    INTERFACE
        Eigen3::Eigen
        FFTW::Double
)
if(HAVE_OPENMP)
    target_link_libraries(randomized-linear-algebra INTERFACE OpenMP::OpenMP_CXX)
    target_compile_definitions(randomized-linear-algebra INTERFACE RLA_USE_OPENMP=1)
endif()
target_compile_features(randomized-linear-algebra INTERFACE cxx_std_17)

# ========================
# Tests
# ========================
enable_testing()

add_executable(test_fixed_rank tests/test_fixed_rank.cpp)
target_link_libraries(test_fixed_rank PRIVATE randomized-linear-algebra)
add_test(NAME FixedRankTest COMMAND test_fixed_rank)

add_executable(test_fixed_precision tests/test_fixed_precision.cpp)
target_link_libraries(test_fixed_precision PRIVATE randomized-linear-algebra)
add_test(NAME FixedPrecisionTest COMMAND test_fixed_precision)

# ========================
# Benchmark (manual run)
# ========================
add_executable(rla_benchmark benchmark/benchmark.cpp)
target_link_libraries(rla_benchmark PRIVATE randomized-linear-algebra)
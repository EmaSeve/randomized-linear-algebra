# ========================
# Threading/BLAS selection
# ========================
set(THREADING_MODE "openmp" CACHE STRING "Threading mode: openmp, blas, or single")
set(ENABLE_OPENMP OFF)
set(ENABLE_OPENBLAS OFF)
if(THREADING_MODE STREQUAL "openmp")
    set(ENABLE_OPENMP ON)
    set(ENABLE_OPENBLAS OFF)
elseif(THREADING_MODE STREQUAL "blas")
    set(ENABLE_OPENMP OFF)
    set(ENABLE_OPENBLAS ON)
elseif(THREADING_MODE STREQUAL "single")
    set(ENABLE_OPENMP OFF)
    set(ENABLE_OPENBLAS OFF)
else()
    message(FATAL_ERROR "THREADING_MODE deve essere uno tra: openmp, blas, single")
endif()

cmake_minimum_required(VERSION 3.15)
project(randomized-linear-algebra VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ========================
# Eigen3 (header-only)
# ========================
find_package(Eigen3 QUIET NO_MODULE)
if(NOT TARGET Eigen3::Eigen)
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

# ========================
# OpenMP (optional)
# ========================
option(BUILD_TESTS "Build tests" ON)
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
# OpenBLAS (optional, per BLAS in Eigen)
# ========================
set(HAVE_OPENBLAS OFF)
if(ENABLE_OPENBLAS)
    find_package(PkgConfig QUIET)
    if(PKG_CONFIG_FOUND)
        pkg_check_modules(OPENBLAS QUIET IMPORTED_TARGET openblas)
        if(TARGET PkgConfig::OPENBLAS)
            add_library(OpenBLAS::OpenBLAS INTERFACE IMPORTED)
            target_link_libraries(OpenBLAS::OpenBLAS INTERFACE PkgConfig::OPENBLAS)
            set(HAVE_OPENBLAS ON)
        endif()
    endif()
    if(NOT HAVE_OPENBLAS)
        message(STATUS "OpenBLAS non trovato: procedo senza backend BLAS per Eigen")
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
    target_compile_definitions(randomized-linear-algebra INTERFACE EIGEN_USE_OPENMP=1)
endif()

# Abilita l'uso della BLAS in Eigen se OpenBLAS è disponibile
if(HAVE_OPENBLAS)
    target_link_libraries(randomized-linear-algebra INTERFACE OpenBLAS::OpenBLAS)
    target_compile_definitions(randomized-linear-algebra INTERFACE EIGEN_USE_BLAS=1)
endif()

target_compile_features(randomized-linear-algebra INTERFACE cxx_std_17)

# ========================
# Tests (GoogleTest + CTest)
# ========================
if(BUILD_TESTS)
    include(CTest)
    enable_testing()

    include(FetchContent)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.17.0
    )
    set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
    set(BUILD_GTEST ON  CACHE BOOL "" FORCE)
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)

    include(GoogleTest)

    add_executable(test_fixed_precision tests/test_fixed_precision.cpp)
    target_link_libraries(test_fixed_precision PRIVATE randomized-linear-algebra GTest::gtest_main)
    gtest_discover_tests(test_fixed_precision
        PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=1"
        DISCOVERY_TIMEOUT 60)

    add_executable(test_fixed_rank tests/test_fixed_rank.cpp)
    target_link_libraries(test_fixed_rank PRIVATE randomized-linear-algebra GTest::gtest_main)
    gtest_discover_tests(test_fixed_rank
        PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=1"
        DISCOVERY_TIMEOUT 60)

    add_executable(test_matrix_factorizer_google tests/test_matrix_factorizer_google.cpp)
    target_link_libraries(test_matrix_factorizer_google PRIVATE randomized-linear-algebra GTest::gtest_main)
    gtest_discover_tests(test_matrix_factorizer_google
        PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=1")
endif()


# TEST DI EMANUELE
if(BUILD_TESTS)
    enable_testing()
    add_executable(test_matrix_factorizer tests/test_matrix_factorizer.cpp)
    target_link_libraries(test_matrix_factorizer PRIVATE randomized-linear-algebra)
    add_test(NAME MatrixFactorizerTest COMMAND test_matrix_factorizer)
endif()

# ========================
# Benchmark
# ========================
if(EXISTS ${CMAKE_SOURCE_DIR}/benchmark/benchmark_fixed_rank_A.cpp)
    add_executable(benchmark_fixed_rank_A benchmark/benchmark_fixed_rank_A.cpp)
    target_link_libraries(benchmark_fixed_rank_A PRIVATE randomized-linear-algebra)
endif()

if(EXISTS ${CMAKE_SOURCE_DIR}/benchmark/benchmark_fixed_precision_A.cpp)
    add_executable(benchmark_fixed_precision_A benchmark/benchmark_fixed_precision_A.cpp)
    target_link_libraries(benchmark_fixed_precision_A PRIVATE randomized-linear-algebra)
endif()

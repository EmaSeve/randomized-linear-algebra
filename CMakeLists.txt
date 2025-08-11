cmake_minimum_required(VERSION 3.15)
project(StochasticLA VERSION 1.0.0 LANGUAGES CXX)

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ========================
# Eigen3 (header-only)
# ========================
find_path(EIGEN3_INCLUDE_DIR
    NAMES Eigen/Core
    HINTS ENV CPATH
    PATH_SUFFIXES Eigen eigen3
)
if(NOT EIGEN3_INCLUDE_DIR)
    message(FATAL_ERROR "Eigen3 headers not found. Carica il modulo eigen/3.4.0 o imposta EIGEN3_INCLUDE_DIR.")
endif()

add_library(Eigen3::Eigen INTERFACE IMPORTED)
target_include_directories(Eigen3::Eigen INTERFACE ${EIGEN3_INCLUDE_DIR})

# ========================
# FFTW (double precision)
# ========================
# ===== FFTW (double precision) =====
find_path(FFTW3_INCLUDE_DIR
    NAMES fftw3.h
    HINTS ENV CPATH
          /u/sw/toolchains/gcc-glibc/11.2.0/pkgs/fftw/3.3.9/include
)
if(NOT FFTW3_INCLUDE_DIR)
    message(FATAL_ERROR "FFTW headers not found. Carica il modulo fftw/3.3.9 o imposta FFTW3_INCLUDE_DIR.")
endif()

find_library(FFTW3_LIB
    NAMES fftw3
    HINTS ENV LIBRARY_PATH ENV LD_LIBRARY_PATH
          /u/sw/toolchains/gcc-glibc/11.2.0/pkgs/fftw/3.3.9/lib
)
if(NOT FFTW3_LIB)
    message(FATAL_ERROR "FFTW library not found. Carica il modulo fftw/3.3.9 o imposta FFTW3_LIB.")
endif()

add_library(FFTW::Double INTERFACE IMPORTED)
target_include_directories(FFTW::Double INTERFACE ${FFTW3_INCLUDE_DIR})
target_link_libraries(FFTW::Double INTERFACE ${FFTW3_LIB})


# ========================
# StochasticLA library
# ========================
add_library(StochasticLA INTERFACE)
target_include_directories(StochasticLA INTERFACE include)
target_link_libraries(StochasticLA INTERFACE Eigen3::Eigen FFTW::Double)
target_compile_features(StochasticLA INTERFACE cxx_std_17)

# ========================
# Tests
# ========================
enable_testing()

file(COPY ${CMAKE_SOURCE_DIR}/tests/data DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

add_executable(test_matrices tests/test_matrices.cpp)
target_link_libraries(test_matrices PRIVATE StochasticLA)
add_test(NAME MatricesTest COMMAND test_matrices)

add_executable(test_spectrum_comparison tests/test_spectrum_comparison.cpp)
target_link_libraries(test_spectrum_comparison PRIVATE StochasticLA)
add_test(NAME SpectrumComparisonTest COMMAND test_spectrum_comparison)

add_executable(test_adaptive_algorithms tests/test_adaptive_algorithms.cpp)
target_link_libraries(test_adaptive_algorithms PRIVATE StochasticLA)
add_test(NAME AdaptiveAlgorithmsTest COMMAND test_adaptive_algorithms)

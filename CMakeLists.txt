cmake_minimum_required(VERSION 3.15)
project(randomized-linear-algebra VERSION 1.0.0 LANGUAGES CXX)

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
# randomized-linear-algebra library
# ========================
add_library(randomized-linear-algebra INTERFACE)
target_include_directories(randomized-linear-algebra INTERFACE include)
target_link_libraries(randomized-linear-algebra INTERFACE Eigen3::Eigen FFTW::Double)
target_compile_features(randomized-linear-algebra INTERFACE cxx_std_17)

# ========================
# Tests
# ========================
enable_testing()

# file(COPY ${CMAKE_SOURCE_DIR}/tests/data DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

add_executable(test_fixed_rank tests/test_fixed_rank.cpp)
target_link_libraries(test_fixed_rank PRIVATE randomized-linear-algebra)
add_test(NAME FixedRankTest COMMAND test_fixed_rank)


add_executable(test_fixed_precision tests/test_fixed_precision.cpp)
target_link_libraries(test_fixed_precision PRIVATE randomized-linear-algebra)
add_test(NAME FixedPrecisionTest COMMAND test_fixed_precision)
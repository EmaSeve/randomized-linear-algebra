cmake_minimum_required(VERSION 3.15)
project(StochasticLA VERSION 1.0.0 LANGUAGES CXX)

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Eigen3
find_package(Eigen3 3.3 QUIET)
if(NOT Eigen3_FOUND)
    find_path(EIGEN3_INCLUDE_DIR Eigen/Dense HINTS ${CMAKE_PREFIX_PATH})
    if(EIGEN3_INCLUDE_DIR)
        add_library(Eigen3::Eigen INTERFACE IMPORTED)
        target_include_directories(Eigen3::Eigen INTERFACE ${EIGEN3_INCLUDE_DIR})
    else()
        message(FATAL_ERROR "Eigen3 not found!")
    endif()
endif()

# Create library
add_library(StochasticLA INTERFACE)
target_include_directories(StochasticLA INTERFACE include)
target_link_libraries(StochasticLA INTERFACE Eigen3::Eigen)
target_compile_features(StochasticLA INTERFACE cxx_std_17)

# Tests
enable_testing()
add_executable(test_basic tests/test_basic.cpp)
target_link_libraries(test_basic StochasticLA)
add_test(NAME BasicTest COMMAND test_basic)

cmake_minimum_required(VERSION 3.20)
project(GlacierML_Benchmarks LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Default to Release build for benchmarks if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose Release build for benchmarks" FORCE)
endif()

# --------------------------------------------------
# Optimization & Eigen Performance Flags
# --------------------------------------------------
if(MSVC)
    add_compile_options(/O2 /Ob2)
else()
    add_compile_options(-O3 -march=native -fopenmp)
endif()

# Disable debug assertions (standard & Eigen)
add_compile_definitions(NDEBUG EIGEN_NO_DEBUG)

# --------------------------------------------------
# Dependencies
# --------------------------------------------------
find_package(OpenMP REQUIRED)
find_package(BLAS REQUIRED)
find_package(Eigen3 REQUIRED)

# Repository root directory path
set(REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/..")

# --------------------------------------------------
# Library: glacier_logr
# --------------------------------------------------
add_library(glacier_logr STATIC
        ${REPO_ROOT}/src/Models/LogR/LogR_impl.cpp
        ${REPO_ROOT}/src/Models/LogR/core/LogRCore_impl.cpp
        ${REPO_ROOT}/src/Utils/utilities.cpp
)

target_include_directories(glacier_logr PUBLIC
        ${REPO_ROOT}/include
        ${REPO_ROOT}/src
)

target_link_libraries(glacier_logr PUBLIC
        Eigen3::Eigen
        BLAS::BLAS
        OpenMP::OpenMP_CXX
)

# --------------------------------------------------
# Logistic Regression Benchmark Executable
# --------------------------------------------------
add_executable(benchmark_logr classifiers/LogisticRegressionDriver.cpp)
target_link_libraries(benchmark_logr PRIVATE glacier_logr)

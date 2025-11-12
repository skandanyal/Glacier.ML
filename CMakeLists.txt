cmake_minimum_required(VERSION 3.28.3)
project(Glacier_ML LANGUAGES CXX)

# Use modern C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Optimization flags for SIMD and fast math
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -ffast-math -fopt-info-vec-optimized")

# Include directories
set(EIGEN_INCLUDE_DIR /home/skandan-c-y/cpp_libraries/eigen)
set(BOOST_INCLUDE_DIR /home/skandan-c-y/cpp_libraries/boost_1_88_0)

include_directories(${EIGEN_INCLUDE_DIR})
include_directories(${BOOST_INCLUDE_DIR})
include_directories(${CMAKE_SOURCE_DIR}/Glacier)
include_directories(${CMAKE_SOURCE_DIR}/Utils)

# Find OpenMP and BLAS
find_package(OpenMP REQUIRED)
find_package(BLAS REQUIRED)

# Header-only library
add_library(Glacier INTERFACE)
target_include_directories(Glacier INTERFACE
        ${EIGEN_INCLUDE_DIR}
        ${BOOST_INCLUDE_DIR}
        ${CMAKE_SOURCE_DIR}/Glacier
        ${CMAKE_SOURCE_DIR}/Utils
)

# Link OpenMP and BLAS for targets that use Glacier
if(OpenMP_CXX_FOUND)
    target_link_libraries(Glacier INTERFACE OpenMP::OpenMP_CXX)
endif()

if(BLAS_FOUND)
    target_link_libraries(Glacier INTERFACE BLAS::BLAS)
endif()

################################################################################################

# Include the executable name here!!
add_executable(KNNC Benchmarking_results/Classifiers/KNNClassifier/KNNClassifierExample.cpp)
add_executable(KNNR Examples/Regression_Models/KNNRegressorExample.cpp)
add_executable(LogR Benchmarking_results/Classifiers/LogisticRegression/LogisticRegressionExample.cpp)
add_executable(MLR Examples/Regression_Models/MultipleLinearRegressionExample.cpp)
add_executable(SLR Examples/Regression_Models/SimpleLinearRegressionExample.cpp)
add_executable(SVMC Benchmarking_results/Classifiers/SVMClassifier/SVMClassifierExample.cpp)
add_executable(SVMR Examples/Regression_Models/SVMRegressorExample.cpp)

target_link_libraries(KNNC PRIVATE Glacier)
target_link_libraries(KNNR PRIVATE Glacier)
target_link_libraries(LogR PRIVATE Glacier)
target_link_libraries(MLR PRIVATE Glacier)
target_link_libraries(SLR PRIVATE Glacier)
target_link_libraries(SVMC PRIVATE Glacier)
target_link_libraries(SVMR PRIVATE Glacier)

target_compile_definitions(KNNC PRIVATE DEBUG_MODE=1)
target_compile_definitions(KNNR PRIVATE DEBUG_MODE=1)
target_compile_definitions(LogR PRIVATE DEBUG_MODE=1)
target_compile_definitions(MLR PRIVATE DEBUG_MODE=1)
target_compile_definitions(SLR PRIVATE DEBUG_MODE=1)
target_compile_definitions(SVMC PRIVATE DEBUG_MODE=1)
target_compile_definitions(SVMR PRIVATE DEBUG_MODE=1)


################################################################################################
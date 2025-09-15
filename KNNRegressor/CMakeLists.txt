cmake_minimum_required(VERSION 3.28.3)
project(__project_name__)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0")

set(EIGEN_INCLUDE_DIR __path_to_Eigen__)                                      # eg: "D:/EigenLibrary/eigen-3.4.0"
set(BOOST_INCLUDE_DIR __path_to_boost__)

# path until folder which contains bin
include_directories(${EIGEN_INCLUDE_DIR})
include_directories(${BOOST_INCLUDE_DIR})
include_directories(${CMAKE_SOURCE_DIR}/Utils)

# Enable high optimization for SIMD and fast math
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -ffast-math -fopt-info-vec-optimized")

# Find OpenMP, BLAS
find_package(OpenMP REQUIRED)
find_package(BLAS REQUIRED)

add_executable(KNNR
        KNNRegressor/KNNRegressorExample.cpp
        KNNRegressor/KNNRegressorFlow.h
        KNNRegressor/KNNRegressorImplementation.cpp
        Utils/utilities.h
        Utils/utilities.cpp
)
target_compile_definitions(KNNR PRIVATE DEBUG_MODE=1)

# Link BLAS to the targets
if(BLAS_FOUND)
    target_link_libraries(KNNR PUBLIC BLAS::BLAS)
endif()

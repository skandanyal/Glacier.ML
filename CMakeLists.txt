cmake_minimum_required(VERSION 3.20)
project(Glacier_ML LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(Boost_NO_SYSTEM_PATHS ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -ffast-math -fopt-info-vec-optimized")

include_directories(BEFORE /home/skandan-c-y/cpp_libraries/boost_1_88_0)
set(EIGEN_INCLUDE_DIR /home/skandan-c-y/cpp_libraries/eigen)
set(BOOST_INCLUDE_DIR /home/skandan-c-y/cpp_libraries/boost_1_88_0)

set(Boost_NO_SYSTEM_PATHS ON)
include_directories(BEFORE ${EIGEN_INCLUDE_DIR})
include_directories(BEFORE ${BOOST_INCLUDE_DIR})

include_directories(${CMAKE_SOURCE_DIR}/Glacier)
include_directories(${CMAKE_SOURCE_DIR}/Utils)

find_package(OpenMP REQUIRED)
find_package(BLAS REQUIRED)
find_package(Boost REQUIRED)
find_package(eigen REQUIRED)

add_library(Glacier INTERFACE)
target_include_directories(Glacier INTERFACE
        BEFORE
        ${EIGEN_INCLUDE_DIR}
        ${BOOST_INCLUDE_DIR}
        ${CMAKE_SOURCE_DIR}/Glacier
        ${CMAKE_SOURCE_DIR}/Utils
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(Glacier INTERFACE OpenMP::OpenMP_CXX)
endif()

if(BLAS_FOUND)
    target_link_libraries(Glacier INTERFACE BLAS::BLAS)
endif()

add_executable(KNNC Benchmark_drivers/Classification_Models/KNNClassifierDriver.cpp)
add_executable(KNNR Benchmark_drivers/Regression_Models/KNNRegressorDriver.cpp)
add_executable(LogR Benchmark_drivers/Classification_Models/LogisticRegressionDriver.cpp)
add_executable(MLR Benchmark_drivers/Regression_Models/MultipleLinearRegressionDriver.cpp)
add_executable(SLR Benchmark_drivers/Regression_Models/SimpleLinearRegressionDriver.cpp)
add_executable(SVMC Benchmark_drivers/Classification_Models/SVMClassifierDriver.cpp)
add_executable(SVMR Benchmark_drivers/Regression_Models/SVMRegressorDriver.cpp)
add_executable(all_models_classifiers Benchmark_drivers/Classification_Models/all_models_classifiers.cpp)

target_link_libraries(KNNC PRIVATE Glacier)
target_link_libraries(KNNR PRIVATE Glacier)
target_link_libraries(LogR PRIVATE Glacier)
target_link_libraries(MLR PRIVATE Glacier)
target_link_libraries(SLR PRIVATE Glacier)
target_link_libraries(SVMC PRIVATE Glacier)
target_link_libraries(SVMR PRIVATE Glacier)
target_link_libraries(all_models_classifiers PRIVATE Glacier)

target_compile_definitions(KNNC PRIVATE DEBUG_MODE=1)
target_compile_definitions(KNNR PRIVATE DEBUG_MODE=1)
target_compile_definitions(LogR PRIVATE DEBUG_MODE=1)
target_compile_definitions(MLR PRIVATE DEBUG_MODE=1)
target_compile_definitions(SLR PRIVATE DEBUG_MODE=1)
target_compile_definitions(SVMC PRIVATE DEBUG_MODE=1)
target_compile_definitions(SVMR PRIVATE DEBUG_MODE=1)
target_compile_definitions(all_models_classifiers PRIVATE DEBUG_MODE=1)

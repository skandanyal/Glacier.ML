cmake_minimum_required(VERSION 3.28)
project(__name_of_your_project__)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# enable high optimization for SIMD and fast math
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -ffast-math -fopt-info-vec-optimized")

set(EIGEN_INCLUDE_DIR __path_to_Eigen__)
set(BOOST_INCLUDE_DIR __path_to_Boost__)

# path until folder which contains bin
include_directories(${EIGEN_INCLUDE_DIR})
include_directories(${BOOST_INCLUDE_DIR})
include_directories(${CMAKE_SOURCE_DIR}/Utils)

# find OpenMP
find_package(OpenMP REQUIRED)

add_executable(KNNC
        KNNClassifier/KNNClassifierExample.cpp
        KNNClassifier/KNNRegressorFlow.h
        KNNClassifier/KNNRegressorImplementation.cpp
        KNNClassifierREADME.md
        Utils/utilities.h
        Utils/utilities.cpp
)
target_compile_definitions(KNNC PRIVATE DEBUG=MODE=0)

# linking OpenMP to the targets
if(OpenMP__CXX_FOUND)
        target_link_libraries(KNNC PUBLIC OpenMP::OpenMP_CXX)
endif()

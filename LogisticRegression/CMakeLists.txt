cmake_minimum_required(VERSION 3.28)
project(__name_of_your_project__)

set(CMAKE_CXX_STANDARD 20)
set(EIGEN_INCLUDE_DIR __path_to_Eigen__)
set(BOOST_INCLUDE_DIR __path_to_Boost__)

# path until folder which contains bin
include_directories(${EIGEN_INCLUDE_DIR})
include_directories(${BOOST_INCLUDE_DIR})
include_directories(${CMAKE_SOURCE_DIR}/Utils)

add_executable(LogR
        LogisticRegression/LogisticRegressionExample.cpp
        LogisticRegression/LogisticRegressionFlow.h
        LogisticRegression/LogisticRegressionImplementation.cpp
        LogisticRegression/LogisticRegressionREADME.md
        Utils/utilities.h
        Utils/utilities.cpp
)
target_compile_definitions(LogR PRIVATE DEBUG_MODE=0)

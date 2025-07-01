cmake_minimum_required(VERSION 3.28)
project(__name_of_your_project__)

set(CMAKE_CXX_STANDARD 20)
set(EIGEN_INCLUDE_DIR __path_to_Eigen__)                                         # eg: "D:/EigenLibrary/eigen-3.4.0"
set(BOOST_INCLUDE_DIR __path_to_Boost__)

# path until folder which contains bin
include_directories(${EIGEN_INCLUDE_DIR})
include_directories(${BOOST_INCLUDE_DIR})
include_directories(${CMAKE_SOURCE_DIR}/Utils)

#add_definitions(-DDEBUG_MODE)

add_executable(MLR
        Multiple_Linear_Regression/MultipleLinearRegression.h
        Multiple_Linear_Regression/MultipleLinearRegressionImplementation.cpp
        Multiple_Linear_Regression/MultipleLinearRegressionExample.cpp
        Utils/utilities.h
        Utils/utilities.cpp
)
target_compile_definition(MLR PRIVATE DEBUG_MODE=1)

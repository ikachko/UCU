cmake_minimum_required(VERSION 3.15)
project(HW1_Vectorization CXX)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -lcblas")

include_directories(/usr/local/opt/openblas/include)
add_executable(HW1_Vectorization main.cpp)

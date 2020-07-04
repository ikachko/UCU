cmake_minimum_required(VERSION 3.15)

set(PROJECT_NAME hw3_multithreading)
project(${PROJECT_NAME})

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


if(APPLE)
    set(OpenCV_DIR /Users/admin/opencv/build_opencv)
endif()

add_executable(${PROJECT_NAME} src/main.cpp)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

target_include_directories(${PROJECT_NAME} PUBLIC src)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})


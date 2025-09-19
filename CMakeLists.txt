cmake_minimum_required(VERSION 3.20.0)
project(opencv_project)

find_package(OpenCV REQUIRED)
add_executable(opencv_project main.cpp)

target_link_libraries(opencv_project ${OpenCV_LIBS})

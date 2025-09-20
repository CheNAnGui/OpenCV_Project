cmake_minimum_required(VERSION 3.10.0)
project(opencv_project)

find_package(OpenCV REQUIRED)
add_executable(opencv_project main.cpp)
add_executable(armor_detector armor_detector.cpp)

target_link_libraries(opencv_project ${OpenCV_LIBS})
target_link_libraries(armor_detector ${OpenCV_LIBS})

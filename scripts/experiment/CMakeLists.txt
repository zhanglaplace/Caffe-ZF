cmake_minimum_required(VERSION 2.9)
project(generate_image)

set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV)

set(SOURCE_FILES facial_experiment.cpp)
add_executable(generate_image ${SOURCE_FILES})

target_link_libraries(generate_image ${OpenCV_LIBS} )

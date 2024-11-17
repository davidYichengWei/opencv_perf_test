cmake_minimum_required(VERSION 3.10)

# For compatibility with CUDA
set(CMAKE_C_COMPILER "/usr/bin/gcc-11")
set(CMAKE_CXX_COMPILER "/usr/bin/g++-11")
set(CMAKE_CXX_STANDARD 14)

project(opencv_demo)

set(OpenCV_DIR "/usr/local/lib/cmake/opencv4")

# To supress the warning about CMP0146
if(POLICY CMP0146)
    cmake_policy(SET CMP0146 OLD)
endif()
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(opencv_demo src/main.cpp)
target_link_libraries(opencv_demo ${OpenCV_LIBS})
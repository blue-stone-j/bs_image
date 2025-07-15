cmake_minimum_required(VERSION 3.16.3)
project(image_analysis)

set(CMAKE_BUILD_TYPE "Release")
add_compile_options(-std=c++17)
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

find_package(OpenCV REQUIRED QUIET)
include_directories(${OpenCV_INCLUDE_DIRS})

set(TARGET_LIBRARIES "")
list(APPEND TARGET_LIBRARIES ${OpenCV_LIBRARIES})

add_executable(brightness src/brightness.cpp)
target_link_libraries(brightness ${TARGET_LIBRARIES})

add_executable(enhance_gray_image src/enhance_gray_image.cpp)
target_link_libraries(enhance_gray_image ${TARGET_LIBRARIES})

add_executable(sharpen_gray_image src/sharpen_gray_image.cpp)
target_link_libraries(sharpen_gray_image ${TARGET_LIBRARIES})
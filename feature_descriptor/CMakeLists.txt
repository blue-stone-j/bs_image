cmake_minimum_required(VERSION 3.16.3)
project(feature_descriptor)

set(CMAKE_BUILD_TYPE "Release")
add_compile_options(-std=c++17)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -pthread")

enable_testing()

find_package(OpenCV REQUIRED QUIET)
include_directories(${OpenCV_INCLUDE_DIRS})

include_directories(include)
set(module_lib "")
list(APPEND module_lib ${GLOG_LIBRARIES})
list(APPEND module_lib ${OpenCV_LIBRARIES})

set(function_names "")
file(GLOB cpp_files "${CMAKE_CURRENT_SOURCE_DIR}/src/*")

add_library(${PROJECT_NAME} SHARED ${cpp_files})
target_link_libraries(${PROJECT_NAME} ${module_lib})
target_include_directories(${PROJECT_NAME} PUBLIC ./include  ${module_include})

if(BUILD_TEST)
    foreach(cpp_file ${cpp_files})
        message(STATUS "Found source file: ${cpp_file}")
        get_filename_component(function_name "${cpp_file}" NAME_WLE)

        add_executable(${function_name}_test ${CMAKE_CURRENT_SOURCE_DIR}/test/${function_name}_test.cpp ${CMAKE_CURRENT_SOURCE_DIR}/src/${function_name}.cpp)
        target_link_libraries(${function_name}_test ${module_lib} gtest gtest_main)
        target_include_directories(${function_name}_test PUBLIC ./include ${module_include})

        add_test(NAME ${function_name}Test COMMAND ${function_name}_test)
    endforeach()
endif()
cmake_minimum_required(VERSION 3.12)
project(dbscan)

add_definitions("-std=c++11")

add_subdirectory(test)

if (NOT APPLE)

    include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
    conan_basic_setup()
    add_executable(dbscan src/dbscan.cpp)
    target_include_directories(dbscan PRIVATE "${PROJECT_SOURCE_DIR}/include")
endif()

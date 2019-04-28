cmake_minimum_required(VERSION 3.12)
project(dbscan)

add_definitions("-std=c++11")

if (NOT APPLE)
    add_executable(dbscan src/dbscan.cpp)
    target_include_directories(dbscan PRIVATE "${PROJECT_SOURCE_DIR}/include")
endif()

add_subdirectory(test)
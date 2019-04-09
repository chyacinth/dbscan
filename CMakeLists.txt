cmake_minimum_required(VERSION 2.8.12)
project(dbscan)

add_definitions("-std=c++11")

add_subdirectory(test)

if (NOT APPLE)
    add_executable(dbscan src/dbscan.cpp)
    target_include_directories(dbscan PRIVATE "${PROJECT_SOURCE_DIR}/include")
endif()
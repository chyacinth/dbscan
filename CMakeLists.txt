cmake_minimum_required(VERSION 3.12)
project(dbscan)

set(default_build_type "Release")
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
  set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE
      STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
    "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

add_definitions("-std=c++11")

if (APPLE)
    add_subdirectory(test)
endif()

if (NOT APPLE)

    include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
    conan_basic_setup()
    add_executable(dbscan src/dbscan.cpp)
    target_include_directories(dbscan PRIVATE "${PROJECT_SOURCE_DIR}/include")

    target_compile_options(dbscan PRIVATE "$<$<CONFIG:Release>:-O3>")

    target_compile_options(dbscan PRIVATE -Wall -Werror -Wextra -pedantic)

endif()

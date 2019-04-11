set(default_build_type "Release")
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
  set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE
      STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
    "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

add_executable(testSingleLinkageTree test.cpp)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)    
    target_link_libraries(testSingleLinkageTree PUBLIC OpenMP::OpenMP_CXX)
endif()

target_include_directories(testSingleLinkageTree PRIVATE "${PROJECT_SOURCE_DIR}/include")

target_compile_options(testSingleLinkageTree PRIVATE "$<$<CONFIG:Release>:-O3>")

target_compile_options(testSingleLinkageTree PRIVATE -Wall -Werror -Wextra -pedantic)
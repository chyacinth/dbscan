add_executable(testSingleLinkageTree test.cpp)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)    
    target_link_libraries(testSingleLinkageTree PUBLIC OpenMP::OpenMP_CXX)
endif()

target_include_directories(testSingleLinkageTree PRIVATE "${PROJECT_SOURCE_DIR}/include")

target_compile_options(testSingleLinkageTree PRIVATE "$<$<CONFIG:Release>:-O3>")

target_compile_options(testSingleLinkageTree PRIVATE -Wall -Werror -Wextra -pedantic)
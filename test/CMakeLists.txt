add_executable(testSingleLinkageTree test.cpp)

target_include_directories(testSingleLinkageTree PRIVATE "${PROJECT_SOURCE_DIR}/include")

target_compile_options(testSingleLinkageTree PRIVATE "$<$<CONFIG:Release>:-O3>")

target_compile_options(testSingleLinkageTree PRIVATE -Wall -Werror -Wextra -pedantic)
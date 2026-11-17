cmake_minimum_required(VERSION 3.16)
project(Learn_OpenCL LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(OpenCL REQUIRED)

# Standalone learning example (no framework).
add_executable(basic_vectoradd src/basic_vectoradd.cpp)
target_link_libraries(basic_vectoradd PRIVATE OpenCL::OpenCL)

# Benchmark harness. Any .cpp dropped into src/algorithms/ is picked up automatically.
file(GLOB ALGORITHM_SOURCES CONFIGURE_DEPENDS src/algorithms/*.cpp)
add_executable(bench
    src/main.cpp
    src/algorithm_driver.cpp
    src/cl_common.cpp
    ${ALGORITHM_SOURCES}
)
target_include_directories(bench PRIVATE include)
target_link_libraries(bench PRIVATE OpenCL::OpenCL)
target_compile_options(bench PRIVATE -Wall -Wextra)
# Kernels are read from the source tree at runtime: edit a .cl file and just re-run, no rebuild needed.
target_compile_definitions(bench PRIVATE KERNEL_DIR="${CMAKE_SOURCE_DIR}/kernels")

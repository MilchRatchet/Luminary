cmake_minimum_required (VERSION 3.9)

project ("Luminary" LANGUAGES CUDA C)

set(CMAKE_CUDA_ARCHITECTURES 61)
set(CMAKE_C_STANDARD 17)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/Luminary/CMake" ${CMAKE_MODULE_PATH})

message(STATUS "SOURCE_DIR: ${CMAKE_SOURCE_DIR}")

find_package(OptiX REQUIRED)

include_directories("${OptiX_INCLUDE}")

file(GLOB cpu_source_files "${CMAKE_SOURCE_DIR}/Luminary/*.c" "${CMAKE_SOURCE_DIR}/Luminary/lib/*.c" "${CMAKE_SOURCE_DIR}/Luminary/lib/zlib/*.c")
file(GLOB gpu_source_files "${CMAKE_SOURCE_DIR}/Luminary/lib/*.cu")

message(STATUS "cpu_source_files: ${cpu_source_files}")
message(STATUS "gpu_source_files: ${gpu_source_files}")

set(SDL2_DIR "${PROJECT_SOURCE_DIR}/Luminary/lib/SDL")

include_directories(${SDL2_DIR})
link_directories(${SDL2_DIR})

add_executable(Luminary ${cpu_source_files} ${gpu_source_files})

target_link_libraries(Luminary SDL2main SDL2)

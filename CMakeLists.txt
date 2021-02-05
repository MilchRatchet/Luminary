cmake_minimum_required (VERSION 3.9)

project ("Luminary" LANGUAGES CUDA C)

set(CMAKE_CUDA_ARCHITECTURES 61)

message(STATUS "SOURCE_DIR: ${CMAKE_SOURCE_DIR}")

file(GLOB cpu_source_files "${CMAKE_SOURCE_DIR}/Luminary/*.c" "${CMAKE_SOURCE_DIR}/Luminary/lib/*.c" "${CMAKE_SOURCE_DIR}/Luminary/lib/zlib/*.c")
file(GLOB gpu_source_files "${CMAKE_SOURCE_DIR}/Luminary/lib/*.cu")

message(STATUS "cpu_source_files: ${cpu_source_files}")
message(STATUS "gpu_source_files: ${gpu_source_files}")

add_executable(Luminary ${cpu_source_files} ${gpu_source_files})

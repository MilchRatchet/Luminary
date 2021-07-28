cmake_minimum_required (VERSION 3.9)

project ("Luminary" LANGUAGES CUDA C)

set(CMAKE_CUDA_ARCHITECTURES 86)
set(CMAKE_C_STANDARD 17)
set(CMAKE_CXX_STANDARD 11)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/Luminary/CMake" ${CMAKE_MODULE_PATH})

find_package(CUDAToolkit 11.3 REQUIRED)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math --ptxas-options=-v --ptxas-options=-warn-double-usage --extra-device-vectorization --generate-line-info --source-in-ptx --compiler-options -O2 -fp:fast -arch:AVX2 -g")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O2 -march=native")

message(STATUS "SOURCE_DIR: ${CMAKE_SOURCE_DIR}")

find_package(OptiX REQUIRED)

include_directories("${OptiX_INCLUDE}")

file(GLOB cpu_source_files "${CMAKE_SOURCE_DIR}/Luminary/*.c" "${CMAKE_SOURCE_DIR}/Luminary/lib/*.c" "${CMAKE_SOURCE_DIR}/Luminary/lib/zlib/*.c" "${CMAKE_SOURCE_DIR}/Luminary/lib/UI/*.c")
file(GLOB gpu_source_files "${CMAKE_SOURCE_DIR}/Luminary/lib/*.cu")

message(STATUS "cpu_source_files: ${cpu_source_files}")
message(STATUS "gpu_source_files: ${gpu_source_files}")

set(SDL2_DIR "${PROJECT_SOURCE_DIR}/Luminary/lib/SDL")

include_directories(${SDL2_DIR})
link_directories(${SDL2_DIR})

add_executable(Luminary ${cpu_source_files} ${gpu_source_files})

target_link_libraries(Luminary SDL2main SDL2)

add_custom_command(TARGET Luminary POST_BUILD
               COMMAND ${CMAKE_COMMAND} -E copy_if_different
               "${SDL2_DIR}/SDL2.dll"
                ${CMAKE_BINARY_DIR})

cmake_minimum_required (VERSION 3.9)

project ("Luminary" LANGUAGES CUDA C)


set(CMAKE_C_STANDARD 17)

option(DEBUG "Debug Mode" OFF)
option(NATIVE_CUDA_ARCH "Use Native CUDA Architecture" ON)

set(CMAKE_BUILD_TYPE "Release")

if (DEBUG)
    set(CMAKE_BUILD_TYPE "Debug")
endif()

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/CMake" ${CMAKE_MODULE_PATH})

#=============================================================
# Configure this path
#=============================================================
find_package(CUDAToolkit 11.6 REQUIRED QUIET)
find_package(OptiX 7.4 REQUIRED QUIET)

#=============================================================
# Set files to compile
#=============================================================
file(GLOB cpu_source_files "${CMAKE_SOURCE_DIR}/src/*.c" "${CMAKE_SOURCE_DIR}/src/zlib/*.c" "${CMAKE_SOURCE_DIR}/src/UI/*.c")
file(GLOB gpu_source_files "${CMAKE_SOURCE_DIR}/src/*.cu")

#=============================================================
# Set include directories
#=============================================================
include_directories("${CMAKE_SOURCE_DIR}/include")
include_directories("${CMAKE_SOURCE_DIR}/include/UI")
include_directories("${CMAKE_SOURCE_DIR}/src/cuda")
include_directories("${OptiX_INCLUDE}")

#=============================================================
# SDL2
#=============================================================
if (DEFINED SDL2_DIR)
    set(SDL2_INCLUDE_DIR "${SDL2_DIR}/include")
    set(SDL2_LIBRARIES_DIR "${SDL2_DIR}/lib/x64")
else()
    find_package(SDL2 QUIET)
    if (SDL2_FOUND)
    else()
        message(FATAL_ERROR "Could not find SDL2. Specify the directory using -DSDL2_DIR")
    endif()
endif()

if (DEFINED SDL2_TTF_DIR)
    set(SDL2_TTF_INCLUDE_DIR "${SDL2_TTF_DIR}/include")
    set(SDL2_TTF_LIBRARIES_DIR "${SDL2_TTF_DIR}/lib/x64")
else()
    message(FATAL_ERROR "Could not find SDL2_ttf.")
endif()

include_directories(${SDL2_INCLUDE_DIR})
link_directories(${SDL2_LIBRARIES_DIR})
include_directories(${SDL2_TTF_INCLUDE_DIR})
link_directories(${SDL2_TTF_LIBRARIES_DIR})
set(SDL2_LIBRARIES "SDL2")
set(SDL2_TTF_LIBRARIES "SDL2_ttf")


#=============================================================
# Set CUDA flags
#=============================================================
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --generate-line-info --source-in-ptx --ptxas-options=-warn-double-usage --ptxas-options=-v --use_fast_math")
if(WIN32)
    set(CMAKE_CUDA_FLAGS_RELEASE "--extra-device-vectorization --compiler-options -O2 -fp:fast -arch:AVX2")
    set(CMAKE_CUDA_FLAGS_DEBUG "--compiler-options -Od -fp:fast -arch:AVX2 -g")
else()
    set(CMAKE_CUDA_FLAGS_RELEASE "--extra-device-vectorization --compiler-options -O3 -march=native")
    set(CMAKE_CUDA_FLAGS_DEBUG "--compiler-options -Og -g")
endif()

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    if (NATIVE_CUDA_ARCH)
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=native")
        set(CMAKE_CUDA_ARCHITECTURES OFF)
    else()
        message(SEND_ERROR "No CUDA Architecture selected. Building for all major archs.")
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=all-major")
        set(CMAKE_CUDA_ARCHITECTURES OFF)
    endif()
else()
    set(NATIVE_CUDA_ARCH OFF)
endif()

#=============================================================
# Set C flags
#=============================================================
if (WIN32)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -D_CRT_SECURE_NO_WARNINGS -D_CRT_SECURE_NO_DEPRECATE -D_CRT_NONSTDC_NO_WARNINGS")
endif()

if(CMAKE_C_COMPILER_ID STREQUAL "Clang")
    if (CMAKE_C_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
        set(CMAKE_C_FLAGS_RELEASE "-O2 -march=native")
        set(CMAKE_C_FLAGS_DEBUG "-Od -g")
    elseif(CMAKE_C_COMPILER_FRONTEND_VARIANT STREQUAL "GNU")
        set(CMAKE_C_FLAGS_RELEASE "-O3 -march=native")
        set(CMAKE_C_FLAGS_DEBUG "-Og -g")
    endif()
elseif (CMAKE_C_COMPILER_ID STREQUAL "GNU")
    set(CMAKE_C_FLAGS_RELEASE "-O3 -march=native")
    set(CMAKE_C_FLAGS_DEBUG "-Og -g")
else()
    message(SEND_ERROR "No flags available for CUDA Compiler ${CMAKE_C_COMPILER_ID}.")
endif()

#=============================================================
# Print directories
#=============================================================
message("")
message("============================= DIRECTORIES =============================")
message("CUDA Toolkit Include:        ${CUDAToolkit_INCLUDE_DIR}")
message("CUDA Toolkit Libraries:      ${CUDAToolkit_LIBRARY_DIR}")
message("CUDA Toolkit NVCC:           ${CUDAToolkit_NVCC_EXECUTABLE}")
message("Optix Include:               ${OptiX_INCLUDE}")
message("SDL2 Include:                ${SDL2_INCLUDE_DIR}")
message("SDL2 Libraries:              ${SDL2_LIBRARIES_DIR}")
message("SDL2_TTF Include:            ${SDL2_TTF_INCLUDE_DIR}")
message("SDL2_TTF Libraries:          ${SDL2_TTF_LIBRARIES_DIR}")
message("=======================================================================")
#=============================================================
# Print compile flags
#=============================================================
message("")
message("=============================== SUMMARY ===============================")
message("Debug Mode                     ${DEBUG}")
message("Use Native CUDA Architecture   ${NATIVE_CUDA_ARCH}")

message("C Compiler:                    ${CMAKE_C_COMPILER_ID} ${CMAKE_C_COMPILER_VERSION}")
message("CUDA Compiler:                 ${CMAKE_CUDA_COMPILER_ID} ${CMAKE_CUDA_COMPILER_VERSION}")

message("C Flags:                       ${CMAKE_C_FLAGS}")
message("CUDA Flags:                    ${CMAKE_CUDA_FLAGS}")

if (DEBUG)
    message("C Debug Flags:                 ${CMAKE_C_FLAGS_DEBUG}")
    message("CUDA Debug Flags:              ${CMAKE_CUDA_FLAGS_DEBUG}")
else()
    message("C Release Flags:               ${CMAKE_C_FLAGS_RELEASE}")
    message("CUDA Release Flags:            ${CMAKE_CUDA_FLAGS_RELEASE}")
endif()


message("=======================================================================")
message("")

#=============================================================
# Executable
#=============================================================
add_executable(Luminary ${cpu_source_files} ${gpu_source_files})
target_link_libraries(Luminary ${SDL2_LIBRARIES} ${SDL2_TTF_LIBRARIES})

if(WIN32)
    add_custom_command(TARGET Luminary POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${SDL2_LIBRARIES_DIR}/SDL2.dll"
        "${SDL2_TTF_LIBRARIES_DIR}/SDL2_ttf.dll"
        "${CMAKE_SOURCE_DIR}/LuminaryFont.ttf"
        ${CMAKE_BINARY_DIR})
else()
    add_custom_command(TARGET Luminary POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_SOURCE_DIR}/LuminaryFont.ttf"
        ${CMAKE_BINARY_DIR})
endif()

#=============================================================
# Configure build information for executable
#=============================================================
set(GIT_COMMIT_DATE "Unknown")
set(GIT_BRANCH_NAME "Unknown")
set(GIT_COMMIT_HASH "Unknown")

find_package(Git QUIET)
if(GIT_FOUND)
    execute_process(
        COMMAND ${GIT_EXECUTABLE} log -1 --format=%cd
        RESULT_VARIABLE RESULT
        OUTPUT_VARIABLE GIT_COMMIT_DATE)

    string (REGEX REPLACE "\n" ""  GIT_COMMIT_DATE ${GIT_COMMIT_DATE})

    execute_process(
        COMMAND ${GIT_EXECUTABLE} rev-parse --abbrev-ref HEAD
        RESULT_VARIABLE RESULT
        OUTPUT_VARIABLE GIT_BRANCH_NAME)

    string (REGEX REPLACE "\n" ""  GIT_BRANCH_NAME ${GIT_BRANCH_NAME})

    execute_process(
        COMMAND ${GIT_EXECUTABLE} rev-parse --short HEAD
        RESULT_VARIABLE RESULT
        OUTPUT_VARIABLE GIT_COMMIT_HASH)

    string (REGEX REPLACE "\n" ""  GIT_COMMIT_HASH ${GIT_COMMIT_HASH})
else()
    message(SEND_ERROR "Failed to find git.")
endif()

configure_file("${CMAKE_SOURCE_DIR}/config.h.in" "${CMAKE_SOURCE_DIR}/include/config.h")

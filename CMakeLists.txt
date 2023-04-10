cmake_minimum_required (VERSION 3.25)

IF(WIN32)
  set(CMAKE_TRY_COMPILE_TARGET_TYPE "STATIC_LIBRARY")
ENDIF()

project ("Luminary" LANGUAGES C CUDA)

set(CMAKE_C_STANDARD 23)

option(DEBUG "Debug Mode" OFF)
option(NATIVE_CUDA_ARCH "Use Native CUDA Architecture" ON)
option(SHOW_KERNEL_STATS "Show CUDA Kernel Stats at Compilation" OFF)

set(CMAKE_BUILD_TYPE "Release")

if (DEBUG)
    set(CMAKE_BUILD_TYPE "Debug")
endif()

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/CMake" ${CMAKE_MODULE_PATH})

#=============================================================
# Link MSVC and Windows SDK libraries
#=============================================================
if(WIN32)
  if(DEFINED WIN_LIB_DIR)
    link_directories("${WIN_LIB_DIR}/um/x64")
    link_directories("${WIN_LIB_DIR}/ucrt/x64")
  endif()

  if(DEFINED MSVC_LIB_DIR)
    link_directories(${MSVC_LIB_DIR})
  endif()
endif()

#=============================================================
# Find Nvidia packages
#=============================================================
find_package(CUDAToolkit 11.6 REQUIRED QUIET)
find_package(Optix 7.4 REQUIRED QUIET)

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
include_directories("${CUDAToolkit_INCLUDE_DIR}")

#=============================================================
# SDL2
#=============================================================
if(WIN32)
    if (DEFINED SDL2_DIR)
        set(SDL2_INCLUDE_DIRS "${SDL2_DIR}/include")
        set(SDL2_LIBRARIES_DIR "${SDL2_DIR}/lib/x64")
    else()
        find_package(SDL2 QUIET)
        if (SDL2_FOUND)
        else()
            message(FATAL_ERROR "Could not find SDL2. Specify the directory using -DSDL2_DIR")
        endif()
    endif()
else()
    find_package(SDL2 REQUIRED QUIET)
endif()

if(WIN32)
    if(DEFINED SDL2_TTF_DIR)
        set(SDL2_TTF_INCLUDE_DIRS "${SDL2_TTF_DIR}/include")
        set(SDL2_TTF_LIBRARIES_DIR "${SDL2_TTF_DIR}/lib/x64")
    else()
        message(FATAL_ERROR "Could not find SDL2_ttf. Specify the directory using -DSDL2_TTF_DIR")
    endif()
elseif(UNIX)
    set(SDL2_TTF_INCLUDE_DIRS "${SDL2_INCLUDE_DIRS}")
    set(SDL2_TTF_LIBRARIES -lSDL2_ttf)
else()
    message(FATAL_ERROR "No handling for SDL2_ttf for the current OS.")
endif()

include_directories(${SDL2_INCLUDE_DIRS})
include_directories(${SDL2_TTF_INCLUDE_DIRS})

if (WIN32)
    link_directories(${SDL2_LIBRARIES_DIR})
    link_directories(${SDL2_TTF_LIBRARIES_DIR})
    set(SDL2_LIBRARIES "SDL2")
    set(SDL2_TTF_LIBRARIES "SDL2_ttf")
endif()


#=============================================================
# Set CUDA flags
#=============================================================
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --generate-line-info --source-in-ptx --ptxas-options=-warn-double-usage --ptxas-options=-warn-spills --use_fast_math --restrict")
if(WIN32)
    set(CMAKE_CUDA_FLAGS_RELEASE "--extra-device-vectorization --ptxas-options=-O3 --ptxas-options=-allow-expensive-optimizations=true --compiler-options -O2 -fp:fast -arch:AVX2")
    set(CMAKE_CUDA_FLAGS_DEBUG "--compiler-options -Od -fp:fast -arch:AVX2 -g")
else()
    set(CMAKE_CUDA_FLAGS_RELEASE "--extra-device-vectorization --ptxas-options=-O3 --ptxas-options=-allow-expensive-optimizations=true --compiler-options -O3 -march=native")
    set(CMAKE_CUDA_FLAGS_DEBUG "--compiler-options -Og -g")
endif()

if(SHOW_KERNEL_STATS)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --ptxas-options=-v")
endif()

if(NATIVE_CUDA_ARCH)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=native")
    set(CMAKE_CUDA_ARCHITECTURES OFF)
elseif(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    message(SEND_ERROR "No CUDA Architecture selected. Building for all major archs.")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=all-major")
    set(CMAKE_CUDA_ARCHITECTURES OFF)
endif()


#=============================================================
# Set C flags
#=============================================================
if (WIN32)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -D_CRT_SECURE_NO_WARNINGS -D_CRT_SECURE_NO_DEPRECATE -D_CRT_NONSTDC_NO_WARNINGS")
elseif(UNIX)
    if(CMAKE_C_COMPILER_ID STREQUAL "Clang")
        if(CMAKE_C_COMPILER_FRONTEND_VARIANT STREQUAL "GNU")
            set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIE")
        endif()
    elseif (CMAKE_C_COMPILER_ID STREQUAL "GNU")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIE")
    endif()
endif()

if(CMAKE_C_COMPILER_ID STREQUAL "Clang")
    if (CMAKE_C_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=native /W4")
        set(CMAKE_C_FLAGS_RELEASE "/O2")
        set(CMAKE_C_FLAGS_DEBUG "/Od /DEBUG")
    elseif(CMAKE_C_COMPILER_FRONTEND_VARIANT STREQUAL "GNU")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=native -Wall -Wno-unused-result -Wno-format -Wno-uninitialized -Wno-incompatible-pointer-types -Wno-strict-aliasing")
        set(CMAKE_C_FLAGS_RELEASE "-O3")
        set(CMAKE_C_FLAGS_DEBUG "-Og -g")
    endif()
elseif (CMAKE_C_COMPILER_ID STREQUAL "GNU")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=native -Wall -Wno-unused-result -Wno-format -Wno-uninitialized -Wno-incompatible-pointer-types -Wno-strict-aliasing")
    set(CMAKE_C_FLAGS_RELEASE "-O3")
    set(CMAKE_C_FLAGS_DEBUG "-Og -g")
else()
    message(SEND_ERROR "No flags available for C Compiler ${CMAKE_C_COMPILER_ID}.")
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
if (WIN32)
    message("SDL2 Include:                ${SDL2_INCLUDE_DIRS}")
    message("SDL2 Libraries:              ${SDL2_LIBRARIES_DIR}")
    message("SDL2_TTF Include:            ${SDL2_TTF_INCLUDE_DIRS}")
    message("SDL2_TTF Libraries:          ${SDL2_TTF_LIBRARIES_DIR}")
else()
    message("SDL2 Include:                ${SDL2_INCLUDE_DIRS}")
    message("SDL2 Libraries:              ${SDL2_LIBRARIES} ${SDL2_TTF_LIBRARIES}")
endif()
message("=======================================================================")
#=============================================================
# Print compile flags
#=============================================================
message("")
message("=============================== SUMMARY ===============================")
message("Debug Mode                     ${DEBUG}")
message("Use Native CUDA Architecture   ${NATIVE_CUDA_ARCH}")
message("Show CUDA Kernel Stats         ${SHOW_KERNEL_STATS}")

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
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
add_executable(Luminary ${cpu_source_files} ${gpu_source_files})
target_link_libraries(Luminary ${SDL2_LIBRARIES} ${SDL2_TTF_LIBRARIES})
target_link_libraries(Luminary CUDA::cudart)
target_link_libraries(Luminary CUDA::cuda_driver)

if(WIN32)
    add_custom_command(TARGET Luminary POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${SDL2_LIBRARIES_DIR}/SDL2.dll"
        "${SDL2_TTF_LIBRARIES_DIR}/SDL2_ttf.dll"
        "${CMAKE_SOURCE_DIR}/LuminaryFont.ttf"
        ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
else()
    add_custom_command(TARGET Luminary POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_SOURCE_DIR}/LuminaryFont.ttf"
        ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
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

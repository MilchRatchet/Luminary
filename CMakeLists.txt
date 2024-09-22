cmake_minimum_required(VERSION 3.25)

IF(WIN32)
  set(CMAKE_TRY_COMPILE_TARGET_TYPE "STATIC_LIBRARY")
ENDIF()

project("Luminary" LANGUAGES C CUDA)

set(CMAKE_C_STANDARD 11)

option(DEBUG "Debug Mode" OFF)
option(NATIVE_CUDA_ARCH "Use Native CUDA Architecture" ON)
option(SHOW_KERNEL_STATS "Show CUDA Kernel Stats at Compilation" OFF)

set(CMAKE_BUILD_TYPE "Release")

if (DEBUG)
    set(CMAKE_BUILD_TYPE "Debug")
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/CMake")

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
# Set CUDA flags
#=============================================================

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --generate-line-info --Wreorder --ptxas-options=-warn-double-usage --ptxas-options=-warn-spills --ptxas-options=-warn-lmem-usage --use_fast_math --split-compile=0")
if(WIN32)
    set(CMAKE_CUDA_FLAGS_RELEASE "--extra-device-vectorization --ptxas-options=-O3 --ptxas-options=-allow-expensive-optimizations=true --compiler-options -O2 -fp:fast -arch:AVX2 -W4")
    set(CMAKE_CUDA_FLAGS_DEBUG "--compiler-options -Od -fp:fast -arch:AVX2 -g")
else()
    set(CMAKE_CUDA_FLAGS_RELEASE "--extra-device-vectorization --ptxas-options=-O3 --ptxas-options=-allow-expensive-optimizations=true --compiler-options -O3 -march=native -Wextra")
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
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=native /W4 -Wno-static-in-inline -Wno-deprecated-non-prototype -Wextra")
        set(CMAKE_C_FLAGS_RELEASE "/O2")
        set(CMAKE_C_FLAGS_DEBUG "/Od /DEBUG")
    elseif(CMAKE_C_COMPILER_FRONTEND_VARIANT STREQUAL "GNU")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=native -Wextra -Wno-unused-result -Wno-format -Wno-uninitialized -Wno-incompatible-pointer-types -Wno-deprecated-non-prototype")
        set(CMAKE_C_FLAGS_RELEASE "-O3")
        set(CMAKE_C_FLAGS_DEBUG "-Og -g")
    endif()
elseif (CMAKE_C_COMPILER_ID STREQUAL "GNU")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=native -Wextra -Wno-unused-result -Wno-format -Wno-uninitialized -Wno-incompatible-pointer-types -Wno-deprecated-non-prototype")
    set(CMAKE_C_FLAGS_RELEASE "-O3")
    set(CMAKE_C_FLAGS_DEBUG "-Og -g")
else()
    message(SEND_ERROR "No flags available for C Compiler ${CMAKE_C_COMPILER_ID}.")
endif()

if(USE_ADDRESS_SANITIZER)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /fsanitize=address")
  link_libraries(clang_rt.asan_dynamic-x86_64.lib)
  link_libraries(clang_rt.asan_dynamic_runtime_thunk-x86_64.lib)
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
message("SDL2                         ${SDL2_DIR}")
message("SDL2_ttf                     ${SDL2_ttf_DIR}")
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
# Set output directory
#=============================================================

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

#=============================================================
# Compile modules
#=============================================================

add_subdirectory(src)

#=============================================================
# Configure build information for executable
#=============================================================

#set(GIT_COMMIT_DATE "Unknown")
#set(GIT_BRANCH_NAME "Unknown")
#set(GIT_COMMIT_HASH "Unknown")
#
#find_package(Git QUIET)
#if(GIT_FOUND)
#    execute_process(
#        COMMAND ${GIT_EXECUTABLE} log -1 --format=%cd
#        RESULT_VARIABLE RESULT
#        OUTPUT_VARIABLE GIT_COMMIT_DATE)
#
#    string (REGEX REPLACE "\n" ""  GIT_COMMIT_DATE ${GIT_COMMIT_DATE})
#
#    execute_process(
#        COMMAND ${GIT_EXECUTABLE} rev-parse --abbrev-ref HEAD
#        RESULT_VARIABLE RESULT
#        OUTPUT_VARIABLE GIT_BRANCH_NAME)
#
#    string (REGEX REPLACE "\n" ""  GIT_BRANCH_NAME ${GIT_BRANCH_NAME})
#
#    execute_process(
#        COMMAND ${GIT_EXECUTABLE} rev-parse --short HEAD
#        RESULT_VARIABLE RESULT
#        OUTPUT_VARIABLE GIT_COMMIT_HASH)
#
#    string (REGEX REPLACE "\n" ""  GIT_COMMIT_HASH ${GIT_COMMIT_HASH})
#else()
#    message(SEND_ERROR "Failed to find git.")
#endif()
#
#configure_file("${CMAKE_SOURCE_DIR}/config.h.in" "${CMAKE_SOURCE_DIR}/include/config.h")

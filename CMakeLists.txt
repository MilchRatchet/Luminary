cmake_minimum_required(VERSION 3.25)

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
# Find Nvidia packages
#=============================================================
find_package(CUDAToolkit 12.1 REQUIRED QUIET)
find_package(Optix 8.0 REQUIRED QUIET)

if(NOT OptiX_VERSION)
  set(OptiX_VERSION "8.0")
endif()

#=============================================================
# Set files to compile
#=============================================================
file(GLOB cpu_source_files "${CMAKE_SOURCE_DIR}/src/*.c" "${CMAKE_SOURCE_DIR}/src/zlib/*.c" "${CMAKE_SOURCE_DIR}/src/UI/*.c")
file(GLOB gpu_source_files "${CMAKE_SOURCE_DIR}/src/*.cu")
file(GLOB gpu_header_files "${CMAKE_SOURCE_DIR}/src/cuda/*.cuh")

list(REMOVE_ITEM gpu_source_files
  "${CMAKE_CURRENT_SOURCE_DIR}/src/optix_kernels.cu"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/optix_kernels_trace_particle.cu"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/optix_kernels_geometry.cu"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/optix_kernels_toy.cu"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/optix_kernels_volume.cu"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/optix_kernels_particle.cu"
)

#=============================================================
# SDL2
#=============================================================
find_package(SDL2 REQUIRED CONFIG)
find_package(SDL2_ttf REQUIRED CONFIG)

#=============================================================
# Set CUDA flags
#=============================================================
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --generate-line-info --Wreorder --ptxas-options=-warn-double-usage --ptxas-options=-warn-spills --ptxas-options=-warn-lmem-usage --use_fast_math --split-compile=0")
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

#=============================================================
# Add Ceb
#=============================================================

add_subdirectory(Ceb)
include_directories(${CEB_INCLUDE_DIR})

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
# Executable
#=============================================================

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
add_executable(Luminary)

#=============================================================
# Create PTX files
#=============================================================

function(compile_ptx kernel_name)
  # TODO: Change this into a proper target so that changes to cuh files are automatically tracked. Currently, any change to cuh files will cause a rebuild of this.
  # Warning: Do not use --optix-ir, I have tried it multiple times over the months now but it is always beyond broken.
  add_custom_command(
    OUTPUT
      "${CMAKE_BINARY_DIR}/${kernel_name}.ptx"
    COMMAND
      ${CUDAToolkit_NVCC_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/src/${kernel_name}.cu"
      --ptx --use_fast_math --generate-line-info --Wreorder --ptxas-options=-warn-double-usage --ptxas-options=-warn-spills --ptxas-options=-warn-lmem-usage --extra-device-vectorization --ptxas-options=-O3 --ptxas-options=-allow-expensive-optimizations=true
      -o "${kernel_name}.ptx"
      -I"${CMAKE_SOURCE_DIR}/include" -I"${CMAKE_SOURCE_DIR}/src/cuda" -I"${OptiX_INCLUDE}"
    DEPENDS
      "${CMAKE_CURRENT_SOURCE_DIR}/src/${kernel_name}.cu"
      "${gpu_header_files}"
    COMMENT
      "Compiling ${kernel_name}.ptx"
  )

  add_custom_target(
    ${kernel_name}
    DEPENDS
      "${CMAKE_BINARY_DIR}/${kernel_name}.ptx"
    COMMENT
      "Target for compiled OptiX kernel."
  )

  add_dependencies(Luminary ${kernel_name})
endfunction()

compile_ptx(optix_kernels)
compile_ptx(optix_kernels_trace_particle)
compile_ptx(optix_kernels_geometry)
compile_ptx(optix_kernels_volume)

#=============================================================
# Embed files
#=============================================================

set(EMBED_FILE_NAME "embed_file")
include(ceb_embed_files)
ceb_embed_files(
  PROJECT Luminary
  OUTPUT ${EMBED_FILE_NAME}
  INPUT
  ${CMAKE_BINARY_DIR}/optix_kernels.ptx
  ${CMAKE_BINARY_DIR}/optix_kernels_trace_particle.ptx
  ${CMAKE_BINARY_DIR}/optix_kernels_geometry.ptx
  ${CMAKE_BINARY_DIR}/optix_kernels_volume.ptx
  ${CMAKE_CURRENT_SOURCE_DIR}/data/LuminaryFont.ttf
  ${CMAKE_CURRENT_SOURCE_DIR}/data/moon/moon_albedo.png
  ${CMAKE_CURRENT_SOURCE_DIR}/data/moon/moon_normal.png
  ${CMAKE_CURRENT_SOURCE_DIR}/data/bluenoise/bluenoise_1D.bin
  ${CMAKE_CURRENT_SOURCE_DIR}/data/bluenoise/bluenoise_2D.bin
)

#=============================================================
# Executable sources
#=============================================================

target_sources(Luminary PRIVATE ${cpu_source_files} ${gpu_source_files} ${CEB_OUTPUT_FILE_${EMBED_FILE_NAME}})

target_link_libraries(Luminary PRIVATE SDL2::SDL2)
target_link_libraries(Luminary PRIVATE SDL2_ttf::SDL2_ttf)
target_link_libraries(Luminary PRIVATE CUDA::cudart)
target_link_libraries(Luminary PRIVATE CUDA::cuda_driver)

target_include_directories(Luminary PRIVATE "${CMAKE_SOURCE_DIR}/include")
target_include_directories(Luminary PRIVATE "${CMAKE_SOURCE_DIR}/include/UI")
target_include_directories(Luminary PRIVATE "${CMAKE_SOURCE_DIR}/src/cuda")
target_include_directories(Luminary PRIVATE SDL2::Headers)
target_include_directories(Luminary PRIVATE SDL2_ttf::SDL2_ttf)
target_include_directories(Luminary PRIVATE "${OptiX_INCLUDE}")
target_include_directories(Luminary PRIVATE CUDA::toolkit)

#=============================================================
# File copying
#=============================================================
if(WIN32)
    add_custom_command(TARGET Luminary POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "$<TARGET_FILE:SDL2::SDL2>"
        "$<TARGET_FILE:SDL2_ttf::SDL2_ttf>"
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

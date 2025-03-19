cmake_minimum_required(VERSION 3.25)

IF(WIN32)
  set(CMAKE_TRY_COMPILE_TARGET_TYPE "STATIC_LIBRARY")
ENDIF()

project("Luminary" LANGUAGES C)

set(CMAKE_C_STANDARD 11)

option(DEBUG "Debug Mode" OFF)
option(LUMINARY_MEMORY_DEBUG "Verbose memory checking" OFF)
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
  if (NOT CMAKE_C_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
    message(FATAL_ERROR "Address Sanitizer is only implemented for MSVC frontends atm.")
  endif()

  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /fsanitize=address /Zi")
  add_link_options("/DEBUG")
  add_link_options("/OPT:REF")
  add_link_options("/OPT:ICF")
  set(CMAKE_C_FLAGS_RELEASE "/Od")
  link_libraries(clang_rt.asan_dynamic-x86_64.lib)
  link_libraries(clang_rt.asan_dynamic_runtime_thunk-x86_64.lib)
endif()

#=============================================================
# Print compile flags
#=============================================================

message("============================= Overall =============================")
message("Debug Mode                     ||  ${DEBUG}")

message("C Compiler:                    ||  ${CMAKE_C_COMPILER_ID} ${CMAKE_C_COMPILER_VERSION}")

message("C Flags:                       ||  ${CMAKE_C_FLAGS}")

if (DEBUG)
    message("C Debug Flags:                 ||  ${CMAKE_C_FLAGS_DEBUG}")
else()
    message("C Release Flags:               ||  ${CMAKE_C_FLAGS_RELEASE}")
endif()

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

configure_file("${CMAKE_SOURCE_DIR}/config.h.in" "${CMAKE_SOURCE_DIR}/src/mandarin_duck/config.h")

message("===================================================================")

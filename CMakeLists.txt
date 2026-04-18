cmake_minimum_required(VERSION 3.25)

IF(WIN32)
  set(CMAKE_TRY_COMPILE_TARGET_TYPE "STATIC_LIBRARY")
ENDIF()

project("Luminary" LANGUAGES C VERSION 1.2.0)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

# Common options
option(LUMINARY_MEMORY_DEBUG "Verbose memory checking" OFF)
option(LUMINARY_KERNEL_ARCHS "Architecture to build" "")
option(LUMINARY_FAST_KERNEL_COMPILE "Use fast Kernel compilation" OFF)
option(LUMINARY_SHOW_KERNEL_STATS "Show CUDA Kernel Stats at Compilation" OFF)
option(LUMINARY_USE_ADDRESS_SANITIZER "Enable Address Sanitizer" OFF)

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
if (CMAKE_C_COMPILER_ID STREQUAL "MSVC")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /D_CRT_SECURE_NO_WARNINGS /D_CRT_SECURE_NO_DEPRECATE /D_CRT_NONSTDC_NO_WARNINGS /MT /Zi")
  set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} /O2 /DNDEBUG")
  set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} /Od /DLUM_DEBUG")
  add_link_options("/DEBUG")
elseif (CMAKE_C_COMPILER_ID STREQUAL "Clang")
  if (CMAKE_C_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /D_CRT_SECURE_NO_WARNINGS /D_CRT_SECURE_NO_DEPRECATE /D_CRT_NONSTDC_NO_WARNINGS -march=native /W4 -Wno-static-in-inline -Wno-deprecated-non-prototype -Wextra /Zi -g -fuse-ld=lld -Z7 /DEBUG")
    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} /O2 /DNDEBUG")
    set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} /Od /DLUM_DEBUG")
    add_link_options("/DEBUG")
  else()
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=native -fPIE -Wextra -Wno-unused-result -Wno-format -Wno-uninitialized -Wno-incompatible-pointer-types -Wno-deprecated-non-prototype")
    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -O3 -DNDEBUG")
    set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -Og -g -DLUM_DEBUG")
  endif()
elseif (CMAKE_C_COMPILER_ID STREQUAL "GNU")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=native -fPIE -Wextra -Wno-unused-result -Wno-format -Wno-uninitialized -Wno-incompatible-pointer-types -Wno-deprecated-non-prototype")
  set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -O3 -DNDEBUG")
  set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -Og -g -DLUM_DEBUG")
else()
    message(SEND_ERROR "No flags available for C Compiler ${CMAKE_C_COMPILER_ID}.")
endif()

if(LUMINARY_USE_ADDRESS_SANITIZER)
  if(CMAKE_C_COMPILER_ID STREQUAL "MSVC" OR (CMAKE_C_COMPILER_ID STREQUAL "Clang" AND CMAKE_C_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC"))
    message(STATUS "Enabling AddressSanitizer flags for MSVC frontends; ensure runtimes are available")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /fsanitize=address")
    add_link_options("/OPT:REF" "/OPT:ICF")
    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} /Od")
    link_libraries(clang_rt.asan_dynamic-x86_64.lib)
    link_libraries(clang_rt.asan_dynamic_runtime_thunk-x86_64.lib)
  else()
    message(STATUS "Enabling AddressSanitizer flags for GCC/Clang")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fsanitize=address")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=address")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -fsanitize=address")
  endif()
endif()

#=============================================================
# Print compile flags
#=============================================================

message("============================= Overall =============================")
message("Build Type                     ||  ${CMAKE_BUILD_TYPE}")
message("C Compiler:                    ||  ${CMAKE_C_COMPILER_ID} ${CMAKE_C_COMPILER_VERSION}")
message("C Flags:                       ||  ${CMAKE_C_FLAGS}")

if (${CMAKE_BUILD_TYPE} STREQUAL "Debug")
  message("C Debug Flags:                 ||  ${CMAKE_C_FLAGS_DEBUG}")
else()
  message("C Release Flags:               ||  ${CMAKE_C_FLAGS_RELEASE}")
endif()

#=============================================================
# Print options
#=============================================================

message("============================= Options =============================")
message("LUMINARY_MEMORY_DEBUG          ||  ${LUMINARY_MEMORY_DEBUG}")
message("LUMINARY_KERNEL_ARCHS          ||  ${LUMINARY_KERNEL_ARCHS}")
message("LUMINARY_FAST_KERNEL_COMPILE   ||  ${LUMINARY_FAST_KERNEL_COMPILE}")
message("LUMINARY_SHOW_KERNEL_STATS     ||  ${LUMINARY_SHOW_KERNEL_STATS}")
message("LUMINARY_USE_ADDRESS_SANITIZER ||  ${LUMINARY_USE_ADDRESS_SANITIZER}")

#=============================================================
# Set output directory
#=============================================================

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
if(CMAKE_CONFIGURATION_TYPES)
  foreach(cfg ${CMAKE_CONFIGURATION_TYPES})
    string(TOUPPER ${cfg} cfg_u)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${cfg_u} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
  endforeach()
else()
  foreach(cfg Debug Release RelWithDebInfo MinSizeRel)
    string(TOUPPER ${cfg} cfg_u)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${cfg_u} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
  endforeach()
endif()

#=============================================================
# Configure build information for executable
#=============================================================

ADD_CUSTOM_COMMAND(
  OUTPUT ${CMAKE_SOURCE_DIR}/git_state.h
         ${CMAKE_SOURCE_DIR}/git_state_cmake_dummy.h # Never actually produced which causes this to always run on every build
  COMMAND ${CMAKE_COMMAND} -DGIT_STATE_TARGET_DIR="${CMAKE_SOURCE_DIR}" -P ${CMAKE_SOURCE_DIR}/cmake/QueryGitState.cmake
)

add_custom_target(
  GIT_STATE_HEADER
  DEPENDS
    ${CMAKE_SOURCE_DIR}/git_state.h
    ${CMAKE_SOURCE_DIR}/git_state_cmake_dummy.h
  COMMENT
    "Target for Git State Header."
)

#=============================================================
# Compile modules
#=============================================================

add_subdirectory(src)
message("===================================================================")

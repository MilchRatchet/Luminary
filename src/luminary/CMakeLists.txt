set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)

#=============================================================
# Find packages
#=============================================================

set(LUMINARY_CUDA_VERSION 12.9 CACHE INTERNAL "CUDA Version")
set(LUMINARY_OPTIX_VERSION 9.0 CACHE INTERNAL "OptiX Version")

find_package(CUDAToolkit ${CUDA_VERSION} REQUIRED QUIET)

# OptiX is a header only library, the headers are provided through a git submodule.
set(OptiX_INCLUDE "${CMAKE_CURRENT_SOURCE_DIR}/optix-dev/include")

#=============================================================
# Get source files
#=============================================================

file(GLOB luminary_cpu_source_files "${CMAKE_CURRENT_SOURCE_DIR}/*.c" "${CMAKE_CURRENT_SOURCE_DIR}/host/*.c" "${CMAKE_CURRENT_SOURCE_DIR}/host/lum/*.c" "${CMAKE_CURRENT_SOURCE_DIR}/device/*.c" "${CMAKE_CURRENT_SOURCE_DIR}/zlib/*.c")
file(GLOB luminary_cpu_header_files "${CMAKE_SOURCE_DIR}/include/luminary/*.h" "${CMAKE_CURRENT_SOURCE_DIR}/*.h" "${CMAKE_CURRENT_SOURCE_DIR}/host/*.h" "${CMAKE_CURRENT_SOURCE_DIR}/host/lum/*.h" "${CMAKE_CURRENT_SOURCE_DIR}/device/*.h")
file(GLOB luminary_gpu_header_files "${CMAKE_CURRENT_SOURCE_DIR}/device/cuda/*.cuh")
file(GLOB luminary_api_header_files "${CMAKE_SOURCE_DIR}/include/luminary/*.h")

#=============================================================
# Targets
#=============================================================

add_library(LuminaryHeaders INTERFACE)
add_library(Luminary::Headers ALIAS LuminaryHeaders)
set_target_properties(LuminaryHeaders PROPERTIES EXPORT_NAME "Headers")
target_include_directories(LuminaryHeaders INTERFACE "${CMAKE_SOURCE_DIR}/include")

add_library(Luminary STATIC)
add_library(Luminary::Library ALIAS Luminary)
set_target_properties(Luminary PROPERTIES EXPORT_NAME "luminary" POSITION_INDEPENDENT_CODE TRUE)

#=============================================================
# Add Ceb
#=============================================================

add_subdirectory(Ceb)
include_directories(${CEB_INCLUDE_DIR})

#=============================================================
# Gather flags
#=============================================================

set(CUDA_KERNEL_FLAGS --generate-line-info --Wreorder --ptxas-options=-warn-double-usage --ptxas-options=-warn-spills --ptxas-options=-warn-lmem-usage --use_fast_math --extra-device-vectorization --ptxas-options=-allow-expensive-optimizations=true)

if(SHOW_KERNEL_STATS)
  set(CUDA_KERNEL_FLAGS ${CUDA_KERNEL_FLAGS} --ptxas-options=-v)
endif()

if(LUMINARY_FAST_KERNEL_COMPILE)
  set(CUDA_KERNEL_FLAGS ${CUDA_KERNEL_FLAGS} --Ofast-compile=min)
else()
  set(CUDA_KERNEL_FLAGS ${CUDA_KERNEL_FLAGS} --ptxas-options=-O3)
endif()

#=============================================================
# Validate architectures
#=============================================================

if (NOT LUMINARY_KERNEL_ARCHS)
  message(FATAL_ERROR "No kernel architectures were specified. Use -DLUMINARY_KERNEL_ARCHS=sm_XY to build for device architecture X.Y.")
endif()

string(REPLACE "," ";" user_specified_archs ${LUMINARY_KERNEL_ARCHS})

set(kernel_archs)
foreach(user_specified_arch ${user_specified_archs})
  if(${user_specified_arch} MATCHES "^sm_[0-9]+a?")
    list(APPEND kernel_archs ${user_specified_arch})
  else()
    message(WARNING "${user_specified_arch} is not a valid CUDA architecture input. The input must be in the form of sm_120,sm90a,...")
  endif()
endforeach()

if(NOT kernel_archs)
  message(FATAL_ERROR "No kernel architectures specified.")
endif()

#=============================================================
# Create CUBIN files
#=============================================================

function(compile_cubin kernel_name kernel_arch out_cubin_file)
  # TODO: Change this into a proper target so that changes to cuh files are automatically tracked. Currently, any change to cuh files will cause a rebuild of this.
  set(cubin_file "${CMAKE_BINARY_DIR}/cubin/${kernel_name}_${kernel_arch}.cubin")
  add_custom_command(
    OUTPUT
      ${cubin_file}
    COMMAND
      ${CUDAToolkit_NVCC_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/device/cuda/${kernel_name}.cu"
      --cubin ${CUDA_KERNEL_FLAGS} --gpu-architecture=${kernel_arch}
      -o ${cubin_file}
      -I"${CMAKE_SOURCE_DIR}/include" -I"${CMAKE_CURRENT_SOURCE_DIR}" -I"${CMAKE_CURRENT_SOURCE_DIR}/device/cuda" -I"${OptiX_INCLUDE}"
    DEPENDS
      "${CMAKE_CURRENT_SOURCE_DIR}/device/cuda/${kernel_name}.cu"
      "${luminary_gpu_header_files}"
      "${luminary_cpu_header_files}"
    COMMENT
      "Compiling ${kernel_name}_${kernel_arch}.cubin"
  )

  add_custom_target(
    ${kernel_name}_${kernel_arch}
    DEPENDS
      ${cubin_file}
    COMMENT
      "Target for compiled CUBIN kernel."
  )

  add_dependencies(Luminary ${kernel_name}_${kernel_arch})

  set(${out_cubin_file} ${cubin_file} PARENT_SCOPE)
endfunction()

set(cubin_files)

foreach(kernel_arch ${kernel_archs})
  set(cubin_file)
  compile_cubin(cuda_kernels ${kernel_arch} cubin_file)

  list(APPEND cubin_files ${cubin_file})
endforeach()


#=============================================================
# Create PTX files
#=============================================================

function(compile_ptx cuda_file_name kernel_name kernel_arch out_ptx_file)
  # TODO: Change this into a proper target so that changes to cuh files are automatically tracked. Currently, any change to cuh files will cause a rebuild of this.
  # Warning: Do not use --optix-ir, I have tried it multiple times over the months now but it is always beyond broken.
  set(ptx_file "${CMAKE_BINARY_DIR}/ptx/${kernel_name}_${kernel_arch}.ptx")
  add_custom_command(
    OUTPUT
      ${ptx_file}
    COMMAND
      ${CUDAToolkit_NVCC_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/device/optix/${cuda_file_name}.cu"
      --ptx ${CUDA_KERNEL_FLAGS} --gpu-architecture=${kernel_arch}
      -D${kernel_name}
      -o ${ptx_file}
      -I"${CMAKE_SOURCE_DIR}/include" -I"${CMAKE_CURRENT_SOURCE_DIR}" -I"${CMAKE_CURRENT_SOURCE_DIR}/device/cuda" -I"${OptiX_INCLUDE}"
    DEPENDS
      "${CMAKE_CURRENT_SOURCE_DIR}/device/optix/${cuda_file_name}.cu"
      "${luminary_gpu_header_files}"
      "${luminary_cpu_header_files}"
    COMMENT
      "Compiling ${kernel_name}_${kernel_arch}.ptx"
  )

  add_custom_target(
    ${kernel_name}_${kernel_arch}
    DEPENDS
      ${ptx_file}
    COMMENT
      "Target for compiled OptiX kernel."
  )

  add_dependencies(Luminary ${kernel_name}_${kernel_arch})

  set(${out_ptx_file} ${ptx_file} PARENT_SCOPE)
endfunction()

set(ptx_files)

foreach(kernel_arch ${kernel_archs})
  set(ptx_file)

  compile_ptx(optix_kernel_raytrace optix_kernel_raytrace ${kernel_arch} ptx_file)
  list(APPEND ptx_files ${ptx_file})

  compile_ptx(optix_kernel_shading_geometry optix_kernel_shading_geometry_geo ${kernel_arch} ptx_file)
  list(APPEND ptx_files ${ptx_file})

  compile_ptx(optix_kernel_shading_geometry optix_kernel_shading_geometry_sky ${kernel_arch} ptx_file)
  list(APPEND ptx_files ${ptx_file})

  compile_ptx(optix_kernel_shading_volume optix_kernel_shading_volume_geo ${kernel_arch} ptx_file)
  list(APPEND ptx_files ${ptx_file})

  compile_ptx(optix_kernel_shading_volume optix_kernel_shading_volume_sky ${kernel_arch} ptx_file)
  list(APPEND ptx_files ${ptx_file})

  compile_ptx(optix_kernel_shading_particles optix_kernel_shading_particles_geo ${kernel_arch} ptx_file)
  list(APPEND ptx_files ${ptx_file})

  compile_ptx(optix_kernel_shading_particles optix_kernel_shading_particles_sky ${kernel_arch} ptx_file)
  list(APPEND ptx_files ${ptx_file})
endforeach()

#=============================================================
# Embed files
#=============================================================

set(EMBED_FILE_NAME "embed_file")
include(ceb_embed_files)
ceb_embed_files(
  PROJECT Luminary
  OUTPUT ${EMBED_FILE_NAME}
  INPUT
  ${cubin_files}
  ${ptx_files}
  ${CMAKE_SOURCE_DIR}/data/LuminaryFont.ttf
  ${CMAKE_SOURCE_DIR}/data/LuminaryFontBold.ttf
  ${CMAKE_SOURCE_DIR}/data/MaterialSymbols.ttf
  ${CMAKE_SOURCE_DIR}/data/moon/moon_albedo.png
  ${CMAKE_SOURCE_DIR}/data/moon/moon_normal.png
  ${CMAKE_SOURCE_DIR}/data/spectral/cie1931_cdf.bin
  ${CMAKE_SOURCE_DIR}/data/spectral/cie1931_xy_lut.bin
  ${CMAKE_SOURCE_DIR}/data/spectral/cie1931_z_lut.bin
  ${CMAKE_SOURCE_DIR}/data/bluenoise/bluenoise_1D.bin
  ${CMAKE_SOURCE_DIR}/data/bluenoise/bluenoise_2D.bin
  ${CMAKE_SOURCE_DIR}/data/bridge/bridge_lut.bin
  ${CMAKE_SOURCE_DIR}/data/icons/SplashScreen.bmp
)

#=============================================================
# Sources
#=============================================================

target_sources(Luminary PRIVATE ${luminary_cpu_source_files} ${CEB_OUTPUT_FILE_${EMBED_FILE_NAME}})

#=============================================================
# Include
#=============================================================

target_include_directories(Luminary PRIVATE "${OptiX_INCLUDE}")
target_include_directories(Luminary PRIVATE CUDA::toolkit)
target_include_directories(Luminary PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
target_include_directories(Luminary PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/device/cuda")

#=============================================================
# Linking
#=============================================================

target_link_libraries(Luminary PRIVATE CUDA::cuda_driver)
target_link_libraries(Luminary PUBLIC $<TARGET_NAME:Luminary::Headers>)

#=============================================================
# Defines
#=============================================================

if(LUMINARY_MEMORY_DEBUG)
  target_compile_definitions(Luminary PRIVATE "LUMINARY_MEMORY_DEBUG")
endif()

#=============================================================
# Configure
#=============================================================

add_dependencies(Luminary GIT_STATE_HEADER)

configure_file("${CMAKE_SOURCE_DIR}/src/luminary/config.h.in" "${CMAKE_SOURCE_DIR}/src/luminary/config.h")

#=============================================================
# Print summary
#=============================================================

message("============================= Luminary ============================")
message("CUDA Kernel Architectures      ||  ${kernel_archs}")
message("Show CUDA Kernel Stats         ||  ${SHOW_KERNEL_STATS}")
message("Verbose Memory Checking        ||  ${LUMINARY_MEMORY_DEBUG}")
message("CUDA Toolkit Version:          ||  ${LUMINARY_CUDA_VERSION}")
message("CUDA Toolkit Libraries:        ||  ${CUDAToolkit_LIBRARY_DIR}")
message("CUDA Toolkit NVCC:             ||  ${CUDAToolkit_NVCC_EXECUTABLE}")
message("OptiX Version:                 ||  ${LUMINARY_OPTIX_VERSION}")
message("OptiX Include:                 ||  ${OptiX_INCLUDE}")
message("CUDA Flags:                    ||  ${CUDA_KERNEL_FLAGS}")

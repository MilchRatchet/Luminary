set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)

#=============================================================
# Find packages
#=============================================================

find_package(CUDAToolkit 12.6 REQUIRED QUIET)
find_package(Optix 8.0 REQUIRED QUIET)

if(NOT OptiX_VERSION)
  set(OptiX_VERSION "8.0")
endif()

#=============================================================
# Get source files
#=============================================================

file(GLOB luminary_cpu_source_files "${CMAKE_CURRENT_SOURCE_DIR}/*.c" "${CMAKE_CURRENT_SOURCE_DIR}/host/*.c" "${CMAKE_CURRENT_SOURCE_DIR}/device/*.c" "${CMAKE_CURRENT_SOURCE_DIR}/zlib/*.c")
file(GLOB luminary_cpu_header_files "${CMAKE_SOURCE_DIR}/include/luminary/*.h" "${CMAKE_CURRENT_SOURCE_DIR}/*.h" "${CMAKE_CURRENT_SOURCE_DIR}/host/*.h" "${CMAKE_CURRENT_SOURCE_DIR}/device/*.h")
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
# Create CUBIN files
#=============================================================

function(compile_cubin kernel_name)
  # TODO: Change this into a proper target so that changes to cuh files are automatically tracked. Currently, any change to cuh files will cause a rebuild of this.
  # TODO: This is currently always building for native. Add some logic that ties in with the normal CUDA kernel arch settings.
  # Warning: Do not use --optix-ir, I have tried it multiple times over the months now but it is always beyond broken.
  add_custom_command(
    OUTPUT
      "${CMAKE_BINARY_DIR}/cubin/${kernel_name}.cubin"
    COMMAND
      ${CUDAToolkit_NVCC_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/device/cuda/${kernel_name}.cu"
      -arch=native --cubin --use_fast_math --generate-line-info --Wreorder --ptxas-options=-warn-double-usage --ptxas-options=-warn-spills --ptxas-options=-warn-lmem-usage --extra-device-vectorization --ptxas-options=-O3 --ptxas-options=-allow-expensive-optimizations=true
      -o "${CMAKE_BINARY_DIR}/cubin/${kernel_name}.cubin"
      -I"${CMAKE_SOURCE_DIR}/include" -I"${CMAKE_CURRENT_SOURCE_DIR}" -I"${CMAKE_CURRENT_SOURCE_DIR}/device/cuda" -I"${OptiX_INCLUDE}"
    DEPENDS
      "${CMAKE_CURRENT_SOURCE_DIR}/device/cuda/${kernel_name}.cu"
      "${luminary_gpu_header_files}"
      "${luminary_cpu_header_files}"
    COMMENT
      "Compiling ${kernel_name}.cubin"
  )

  add_custom_target(
    ${kernel_name}
    DEPENDS
      "${CMAKE_BINARY_DIR}/cubin/${kernel_name}.cubin"
    COMMENT
      "Target for compiled CUBIN kernel."
  )

  add_dependencies(Luminary ${kernel_name})
endfunction()

compile_cubin(cuda_kernels)

#=============================================================
# Create OptiX-IR files
#=============================================================

function(compile_optixir kernel_name)
  # TODO: Change this into a proper target so that changes to cuh files are automatically tracked. Currently, any change to cuh files will cause a rebuild of this.
  # TODO: This is currently always building for native. Add some logic that ties in with the normal CUDA kernel arch settings.
  # Warning: Do not use --optix-ir, I have tried it multiple times over the months now but it is always beyond broken.
  add_custom_command(
    OUTPUT
      "${CMAKE_BINARY_DIR}/optixir/${kernel_name}.optixir"
    COMMAND
      ${CUDAToolkit_NVCC_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/device/optix/${kernel_name}.cu"
      -arch=native --optix-ir --use_fast_math --generate-line-info --Wreorder --ptxas-options=-warn-double-usage --ptxas-options=-warn-spills --ptxas-options=-warn-lmem-usage --extra-device-vectorization --ptxas-options=-O3 --ptxas-options=-allow-expensive-optimizations=true
      -o "${CMAKE_BINARY_DIR}/optixir/${kernel_name}.optixir"
      -I"${CMAKE_SOURCE_DIR}/include" -I"${CMAKE_CURRENT_SOURCE_DIR}" -I"${CMAKE_CURRENT_SOURCE_DIR}/device/cuda" -I"${OptiX_INCLUDE}"
    DEPENDS
      "${CMAKE_CURRENT_SOURCE_DIR}/device/optix/${kernel_name}.cu"
      "${luminary_gpu_header_files}"
      "${luminary_cpu_header_files}"
    COMMENT
      "Compiling ${kernel_name}.optixir"
  )

  add_custom_target(
    ${kernel_name}
    DEPENDS
      "${CMAKE_BINARY_DIR}/optixir/${kernel_name}.optixir"
    COMMENT
      "Target for compiled OptiX kernel."
  )

  add_dependencies(Luminary ${kernel_name})
endfunction()

compile_optixir(optix_kernel_raytrace_geometry)
compile_optixir(optix_kernel_raytrace_particles)
compile_optixir(optix_kernel_shading_geometry)
compile_optixir(optix_kernel_shading_volume)
compile_optixir(optix_kernel_shading_particles)
compile_optixir(optix_kernel_shading_volume_bridges)

#=============================================================
# Embed files
#=============================================================

set(EMBED_FILE_NAME "embed_file")
include(ceb_embed_files)
ceb_embed_files(
  PROJECT Luminary
  OUTPUT ${EMBED_FILE_NAME}
  INPUT
  ${CMAKE_BINARY_DIR}/cubin/cuda_kernels.cubin
  ${CMAKE_BINARY_DIR}/optixir/optix_kernel_raytrace_geometry.optixir
  ${CMAKE_BINARY_DIR}/optixir/optix_kernel_raytrace_particles.optixir
  ${CMAKE_BINARY_DIR}/optixir/optix_kernel_shading_geometry.optixir
  ${CMAKE_BINARY_DIR}/optixir/optix_kernel_shading_volume.optixir
  ${CMAKE_BINARY_DIR}/optixir/optix_kernel_shading_particles.optixir
  ${CMAKE_BINARY_DIR}/optixir/optix_kernel_shading_volume_bridges.optixir
  ${CMAKE_SOURCE_DIR}/data/LuminaryFont.ttf
  ${CMAKE_SOURCE_DIR}/data/LuminaryFontBold.ttf
  ${CMAKE_SOURCE_DIR}/data/MaterialSymbols.ttf
  ${CMAKE_SOURCE_DIR}/data/moon/moon_albedo.png
  ${CMAKE_SOURCE_DIR}/data/moon/moon_normal.png
  ${CMAKE_SOURCE_DIR}/data/bluenoise/bluenoise_1D.bin
  ${CMAKE_SOURCE_DIR}/data/bluenoise/bluenoise_2D.bin
  ${CMAKE_SOURCE_DIR}/data/bridge/bridge_lut.bin
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

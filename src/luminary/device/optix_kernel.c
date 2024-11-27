#include "optix_kernel.h"

#include <optix.h>
#include <optix_stubs.h>
#include <stdio.h>
#include <stdlib.h>

#include "ceb.h"
#include "device.h"
#include "device_memory.h"
#include "internal_error.h"
#include "utils.h"

struct OptixKernelConfig {
  const char* name;
  int num_payloads;
} typedef OptixKernelConfig;

// TODO: Rename the ptx to the same as the types.
static const OptixKernelConfig optix_kernel_configs[OPTIX_KERNEL_TYPE_COUNT] = {
  [OPTIX_KERNEL_TYPE_RAYTRACE_GEOMETRY]      = {.name = "optix_kernels.ptx", .num_payloads = 3},
  [OPTIX_KERNEL_TYPE_RAYTRACE_PARTICLES]     = {.name = "optix_kernels_trace_particle.ptx", .num_payloads = 2},
  [OPTIX_KERNEL_TYPE_SHADING_GEOMETRY]       = {.name = "optix_kernels_geometry.ptx", .num_payloads = 5},
  [OPTIX_KERNEL_TYPE_SHADING_VOLUME]         = {.name = "optix_kernels_volume.ptx", .num_payloads = 5},
  [OPTIX_KERNEL_TYPE_SHADING_PARTICLES]      = {.name = "optix_kernels_particle.ptx", .num_payloads = 5},
  [OPTIX_KERNEL_TYPE_SHADING_VOLUME_BRIDGES] = {.name = "optix_kernels_volume_bridges.ptx", .num_payloads = 5}};

LuminaryResult optix_kernel_create(OptixKernel** kernel, Device* device, OptixKernelType type) {
  __CHECK_NULL_ARGUMENT(kernel);
  __CHECK_NULL_ARGUMENT(device);

  log_message("Compiling kernel %s for %s.", optix_kernel_configs[type].name, device->properties.name);

  __FAILURE_HANDLE(host_malloc(kernel, sizeof(OptixKernel)));

  ////////////////////////////////////////////////////////////////////
  // Get PTX
  ////////////////////////////////////////////////////////////////////

  int64_t ptx_length;
  char* ptx;
  uint64_t ptx_info;

  ceb_access(optix_kernel_configs[type].name, (void**) &ptx, &ptx_length, &ptx_info);

  if (ptx_info || !ptx_length) {
    __RETURN_ERROR(
      LUMINARY_ERROR_API_EXCEPTION, "Failed to load OptiX kernels %s. Ceb Error Code: %zu.", optix_kernel_configs[type].name, ptx_info);
  }

  ////////////////////////////////////////////////////////////////////
  // Module Compilation
  ////////////////////////////////////////////////////////////////////

  OptixModuleCompileOptions module_compile_options;
  memset(&module_compile_options, 0, sizeof(OptixModuleCompileOptions));

  module_compile_options.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
  module_compile_options.optLevel         = OPTIX_COMPILE_OPTIMIZATION_LEVEL_3;
  module_compile_options.debugLevel       = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

  OptixPipelineCompileOptions pipeline_compile_options;
  memset(&pipeline_compile_options, 0, sizeof(OptixPipelineCompileOptions));

  pipeline_compile_options.usesMotionBlur                   = 0;
  pipeline_compile_options.traversableGraphFlags            = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY;
  pipeline_compile_options.numPayloadValues                 = optix_kernel_configs[type].num_payloads;
  pipeline_compile_options.numAttributeValues               = 2;
  pipeline_compile_options.exceptionFlags                   = OPTIX_EXCEPTION_FLAG_NONE;
  pipeline_compile_options.pipelineLaunchParamsVariableName = "device";
  pipeline_compile_options.usesPrimitiveTypeFlags           = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;

  // TODO: Handle OMMs and DMMs.
#if 0
  if (settings.use_opacity_micromaps)
    pipeline_compile_options.allowOpacityMicromaps = 1;

  if (settings.use_displacement_micromaps)
    pipeline_compile_options.usesPrimitiveTypeFlags |= OPTIX_PRIMITIVE_TYPE_FLAGS_DISPLACED_MICROMESH_TRIANGLE;
#endif

  char log[4096];
  size_t log_size = sizeof(log);

  OPTIX_FAILURE_HANDLE_LOG(
    optixModuleCreate(
      device->optix_ctx, &module_compile_options, &pipeline_compile_options, ptx, ptx_length, log, &log_size, &(*kernel)->module),
    log);

  ////////////////////////////////////////////////////////////////////
  // Group Creation
  ////////////////////////////////////////////////////////////////////

  OptixProgramGroupOptions group_options;
  memset(&group_options, 0, sizeof(OptixProgramGroupOptions));

  OptixProgramGroupDesc group_desc[OPTIXRT_NUM_GROUPS];
  memset(group_desc, 0, OPTIXRT_NUM_GROUPS * sizeof(OptixProgramGroupDesc));

  group_desc[0].kind                     = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
  group_desc[0].raygen.module            = (*kernel)->module;
  group_desc[0].raygen.entryFunctionName = "__raygen__optix";

  group_desc[1].kind = OPTIX_PROGRAM_GROUP_KIND_MISS;

  group_desc[2].kind                         = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
  group_desc[2].hitgroup.moduleAH            = (*kernel)->module;
  group_desc[2].hitgroup.entryFunctionNameAH = "__anyhit__optix";
  group_desc[2].hitgroup.moduleCH            = (*kernel)->module;
  group_desc[2].hitgroup.entryFunctionNameCH = "__closesthit__optix";

  OPTIX_FAILURE_HANDLE_LOG(
    optixProgramGroupCreate(device->optix_ctx, group_desc, OPTIXRT_NUM_GROUPS, &group_options, log, &log_size, (*kernel)->groups), log);

  ////////////////////////////////////////////////////////////////////
  // Pipeline Creation
  ////////////////////////////////////////////////////////////////////

  OptixPipelineLinkOptions pipeline_link_options;
  pipeline_link_options.maxTraceDepth = 1;

  OPTIX_FAILURE_HANDLE_LOG(
    optixPipelineCreate(
      device->optix_ctx, &pipeline_compile_options, &pipeline_link_options, (*kernel)->groups, OPTIXRT_NUM_GROUPS, log, &log_size,
      &(*kernel)->pipeline),
    log);

  ////////////////////////////////////////////////////////////////////
  // Shader Binding Table Creation
  ////////////////////////////////////////////////////////////////////

  __FAILURE_HANDLE(device_malloc(&(*kernel)->records, OPTIXRT_NUM_GROUPS * OPTIX_SBT_RECORD_HEADER_SIZE));

  char host_records[OPTIXRT_NUM_GROUPS * OPTIX_SBT_RECORD_HEADER_SIZE];

  for (uint32_t i = 0; i < OPTIXRT_NUM_GROUPS; i++) {
    OPTIX_FAILURE_HANDLE(optixSbtRecordPackHeader((*kernel)->groups[i], host_records + i * OPTIX_SBT_RECORD_HEADER_SIZE));
  }

  __FAILURE_HANDLE(
    device_upload((*kernel)->records, host_records, 0, OPTIXRT_NUM_GROUPS * OPTIX_SBT_RECORD_HEADER_SIZE, device->stream_main));

  memset(&(*kernel)->shaders, 0, sizeof(OptixShaderBindingTable));

  (*kernel)->shaders.raygenRecord       = DEVICE_CUPTR((*kernel)->records) + 0 * OPTIX_SBT_RECORD_HEADER_SIZE;
  (*kernel)->shaders.missRecordBase     = DEVICE_CUPTR((*kernel)->records) + 1 * OPTIX_SBT_RECORD_HEADER_SIZE;
  (*kernel)->shaders.hitgroupRecordBase = DEVICE_CUPTR((*kernel)->records) + 2 * OPTIX_SBT_RECORD_HEADER_SIZE;

  (*kernel)->shaders.missRecordStrideInBytes = OPTIX_SBT_RECORD_HEADER_SIZE;
  (*kernel)->shaders.missRecordCount         = 1;

  (*kernel)->shaders.hitgroupRecordStrideInBytes = OPTIX_SBT_RECORD_HEADER_SIZE;
  (*kernel)->shaders.hitgroupRecordCount         = 1;

  ////////////////////////////////////////////////////////////////////
  // Params Creation
  ////////////////////////////////////////////////////////////////////

  __FAILURE_HANDLE(device_malloc(&((*kernel)->params), sizeof(DeviceConstantMemory)));

  return LUMINARY_SUCCESS;
}

LuminaryResult optix_kernel_execute(OptixKernel* kernel, Device* device) {
  __CHECK_NULL_ARGUMENT(kernel);
  __CHECK_NULL_ARGUMENT(device);

  CUDA_FAILURE_HANDLE(cuMemcpyDtoDAsync_v2(
    DEVICE_CUPTR(kernel->params), device->cuda_device_const_memory, sizeof(DeviceConstantMemory), device->stream_main));

  OPTIX_FAILURE_HANDLE(optixLaunch(
    kernel->pipeline, device->stream_main, DEVICE_CUPTR(kernel->params), sizeof(DeviceConstantMemory), &kernel->shaders, THREADS_PER_BLOCK,
    BLOCKS_PER_GRID, 1));

  return LUMINARY_SUCCESS;
}

LuminaryResult optix_kernel_destroy(OptixKernel** kernel) {
  __CHECK_NULL_ARGUMENT(kernel);

  __FAILURE_HANDLE(device_free(&(*kernel)->records));

  OPTIX_FAILURE_HANDLE(optixPipelineDestroy((*kernel)->pipeline));

  for (uint32_t group_id = 0; group_id < OPTIXRT_NUM_GROUPS; group_id++) {
    OPTIX_FAILURE_HANDLE(optixProgramGroupDestroy((*kernel)->groups[group_id]));
  }

  OPTIX_FAILURE_HANDLE(optixModuleDestroy((*kernel)->module))

  __FAILURE_HANDLE(device_free(&(*kernel)->params));

  __FAILURE_HANDLE(host_free(kernel));

  return LUMINARY_SUCCESS;
}

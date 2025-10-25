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

#define OPTIX_LOG_SIZE (1u << 16)

struct OptixKernelConfig {
  const char* name;
  uint32_t register_count;
  bool allow_gas;
} typedef OptixKernelConfig;

// TODO: Make register count architecture dependent.
static const OptixKernelConfig optix_kernel_configs[OPTIX_KERNEL_TYPE_COUNT] = {
  [OPTIX_KERNEL_TYPE_RAYTRACE]      = {.name = "optix_kernel_raytrace", .register_count = 40, .allow_gas = false},
  [OPTIX_KERNEL_TYPE_SHADOW]        = {.name = "optix_kernel_shadow", .register_count = 40, .allow_gas = false},
  [OPTIX_KERNEL_TYPE_SHADOW_VOLUME] = {.name = "optix_kernel_shadow_volume", .register_count = 40, .allow_gas = false}};

static const char* optix_anyhit_function_names[OPTIX_KERNEL_FUNCTION_COUNT] = {
  "__anyhit__geometry_trace", "__anyhit__particle_trace", "__anyhit__shadow_trace", "__anyhit__shadow_sun_trace"};

static const char* optix_closesthit_function_names[OPTIX_KERNEL_FUNCTION_COUNT] = {
  "__closesthit__geometry_trace", "__closesthit__particle_trace", "__closesthit__shadow_trace", "__closesthit__shadow_sun_trace"};

static const uint32_t optix_kernel_function_payload_semantics_geometry_trace[OPTIX_KERNEL_FUNCTION_GEOMETRY_TRACE_PAYLOAD_VALUE_COUNT] = {
  [OPTIX_KERNEL_FUNCTION_GEOMETRY_TRACE_PAYLOAD_VALUE_DEPTH] =
    OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ_WRITE | OPTIX_PAYLOAD_SEMANTICS_CH_WRITE | OPTIX_PAYLOAD_SEMANTICS_AH_NONE,
  [OPTIX_KERNEL_FUNCTION_GEOMETRY_TRACE_PAYLOAD_VALUE_TRIANGLE_HANDLE] =
    OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ_WRITE | OPTIX_PAYLOAD_SEMANTICS_CH_WRITE | OPTIX_PAYLOAD_SEMANTICS_AH_READ,
  [OPTIX_KERNEL_FUNCTION_GEOMETRY_TRACE_PAYLOAD_VALUE_TRIANGLE_HANDLE2] =
    OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ_WRITE | OPTIX_PAYLOAD_SEMANTICS_CH_WRITE | OPTIX_PAYLOAD_SEMANTICS_AH_READ};

static const uint32_t optix_kernel_function_payload_semantics_particle_trace[OPTIX_KERNEL_FUNCTION_PARTICLE_TRACE_PAYLOAD_VALUE_COUNT] = {
  [OPTIX_KERNEL_FUNCTION_PARTICLE_TRACE_PAYLOAD_VALUE_DEPTH] =
    OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ_WRITE | OPTIX_PAYLOAD_SEMANTICS_CH_WRITE | OPTIX_PAYLOAD_SEMANTICS_AH_NONE,
  [OPTIX_KERNEL_FUNCTION_PARTICLE_TRACE_PAYLOAD_VALUE_INSTANCE_ID] =
    OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ_WRITE | OPTIX_PAYLOAD_SEMANTICS_CH_WRITE | OPTIX_PAYLOAD_SEMANTICS_AH_NONE};

static const uint32_t optix_kernel_function_payload_semantics_shadow_trace[OPTIX_KERNEL_FUNCTION_SHADOW_TRACE_PAYLOAD_VALUE_COUNT] = {
  [OPTIX_KERNEL_FUNCTION_SHADOW_TRACE_PAYLOAD_VALUE_TRIANGLE_HANDLE] =
    OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ_WRITE | OPTIX_PAYLOAD_SEMANTICS_CH_WRITE | OPTIX_PAYLOAD_SEMANTICS_AH_READ,
  [OPTIX_KERNEL_FUNCTION_SHADOW_TRACE_PAYLOAD_VALUE_TRIANGLE_HANDLE2] =
    OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_WRITE | OPTIX_PAYLOAD_SEMANTICS_CH_NONE | OPTIX_PAYLOAD_SEMANTICS_AH_READ,
  [OPTIX_KERNEL_FUNCTION_SHADOW_TRACE_PAYLOAD_VALUE_TRIANGLE_HANDLE3] =
    OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_WRITE | OPTIX_PAYLOAD_SEMANTICS_CH_NONE | OPTIX_PAYLOAD_SEMANTICS_AH_READ,
  [OPTIX_KERNEL_FUNCTION_SHADOW_TRACE_PAYLOAD_VALUE_TRIANGLE_HANDLE4] =
    OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_WRITE | OPTIX_PAYLOAD_SEMANTICS_CH_NONE | OPTIX_PAYLOAD_SEMANTICS_AH_READ,
  [OPTIX_KERNEL_FUNCTION_SHADOW_TRACE_PAYLOAD_VALUE_THROUGHPUT] =
    OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ_WRITE | OPTIX_PAYLOAD_SEMANTICS_CH_NONE | OPTIX_PAYLOAD_SEMANTICS_AH_READ_WRITE,
  [OPTIX_KERNEL_FUNCTION_SHADOW_TRACE_PAYLOAD_VALUE_THROUGHPUT2] =
    OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ_WRITE | OPTIX_PAYLOAD_SEMANTICS_CH_NONE | OPTIX_PAYLOAD_SEMANTICS_AH_READ_WRITE,
  [OPTIX_KERNEL_FUNCTION_SHADOW_TRACE_PAYLOAD_VALUE_THROUGHPUT3] =
    OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ_WRITE | OPTIX_PAYLOAD_SEMANTICS_CH_NONE | OPTIX_PAYLOAD_SEMANTICS_AH_READ_WRITE};

static const uint32_t optix_kernel_function_payload_semantics_shadow_sun_trace[OPTIX_KERNEL_FUNCTION_SHADOW_SUN_TRACE_PAYLOAD_VALUE_COUNT] =
  {[OPTIX_KERNEL_FUNCTION_SHADOW_SUN_TRACE_PAYLOAD_VALUE_TRIANGLE_HANDLE] =
     OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_WRITE | OPTIX_PAYLOAD_SEMANTICS_CH_NONE | OPTIX_PAYLOAD_SEMANTICS_AH_READ,
   [OPTIX_KERNEL_FUNCTION_SHADOW_SUN_TRACE_PAYLOAD_VALUE_TRIANGLE_HANDLE2] =
     OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_WRITE | OPTIX_PAYLOAD_SEMANTICS_CH_NONE | OPTIX_PAYLOAD_SEMANTICS_AH_READ,
   [OPTIX_KERNEL_FUNCTION_SHADOW_SUN_TRACE_PAYLOAD_VALUE_THROUGHPUT] =
     OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ_WRITE | OPTIX_PAYLOAD_SEMANTICS_CH_WRITE | OPTIX_PAYLOAD_SEMANTICS_AH_READ_WRITE,
   [OPTIX_KERNEL_FUNCTION_SHADOW_SUN_TRACE_PAYLOAD_VALUE_THROUGHPUT2] =
     OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ_WRITE | OPTIX_PAYLOAD_SEMANTICS_CH_WRITE | OPTIX_PAYLOAD_SEMANTICS_AH_READ_WRITE,
   [OPTIX_KERNEL_FUNCTION_SHADOW_SUN_TRACE_PAYLOAD_VALUE_THROUGHPUT3] =
     OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ_WRITE | OPTIX_PAYLOAD_SEMANTICS_CH_WRITE | OPTIX_PAYLOAD_SEMANTICS_AH_READ_WRITE};

static const OptixPayloadType optix_kernel_function_payload_types[OPTIX_KERNEL_FUNCTION_COUNT] = {
  [OPTIX_KERNEL_FUNCTION_GEOMETRY_TRACE] =
    {.numPayloadValues = OPTIX_KERNEL_FUNCTION_GEOMETRY_TRACE_PAYLOAD_VALUE_COUNT,
     .payloadSemantics = optix_kernel_function_payload_semantics_geometry_trace},
  [OPTIX_KERNEL_FUNCTION_PARTICLE_TRACE] =
    {.numPayloadValues = OPTIX_KERNEL_FUNCTION_PARTICLE_TRACE_PAYLOAD_VALUE_COUNT,
     .payloadSemantics = optix_kernel_function_payload_semantics_particle_trace},
  [OPTIX_KERNEL_FUNCTION_SHADOW_TRACE] =
    {.numPayloadValues = OPTIX_KERNEL_FUNCTION_SHADOW_TRACE_PAYLOAD_VALUE_COUNT,
     .payloadSemantics = optix_kernel_function_payload_semantics_shadow_trace},
  [OPTIX_KERNEL_FUNCTION_SHADOW_SUN_TRACE] = {
    .numPayloadValues = OPTIX_KERNEL_FUNCTION_SHADOW_SUN_TRACE_PAYLOAD_VALUE_COUNT,
    .payloadSemantics = optix_kernel_function_payload_semantics_shadow_sun_trace}};

LuminaryResult optix_kernel_create(OptixKernel** kernel, Device* device, OptixKernelType type) {
  __CHECK_NULL_ARGUMENT(kernel);
  __CHECK_NULL_ARGUMENT(device);

  log_message("Compiling kernel %s for %s.", optix_kernel_configs[type].name, device->properties.name);

  __FAILURE_HANDLE(host_malloc(kernel, sizeof(OptixKernel)));
  memset(*kernel, 0, sizeof(OptixKernel));

  ////////////////////////////////////////////////////////////////////
  // Get OptiX-IR
  ////////////////////////////////////////////////////////////////////

  char optixir_name[4096];
  sprintf(optixir_name, "%s_sm_%u%u.optix-ir", optix_kernel_configs[type].name, device->properties.major, device->properties.minor);

  int64_t optixir_length;
  char* optixir;
  uint64_t optixir_info;

  ceb_access(optixir_name, (void**) &optixir, &optixir_length, &optixir_info);

  if (optixir_info || !optixir_length) {
    warn_message(
      "Failed to load OptiX kernel %s for CUDA architecture sm_%u%u. All devices of this architecture will be unavailable. Recompile "
      "Luminary "
      "with this architecture enabled.",
      optix_kernel_configs[type].name, device->properties.major, device->properties.minor);

    return LUMINARY_SUCCESS;
  }

  ////////////////////////////////////////////////////////////////////
  // Module Compilation
  ////////////////////////////////////////////////////////////////////

  OptixModuleCompileOptions module_compile_options;
  memset(&module_compile_options, 0, sizeof(OptixModuleCompileOptions));

  module_compile_options.maxRegisterCount = optix_kernel_configs[type].register_count;
  module_compile_options.optLevel         = OPTIX_COMPILE_OPTIMIZATION_LEVEL_3;
  module_compile_options.debugLevel       = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
  module_compile_options.numPayloadTypes  = OPTIX_KERNEL_FUNCTION_COUNT;
  module_compile_options.payloadTypes     = optix_kernel_function_payload_types;

  OptixPipelineCompileOptions pipeline_compile_options;
  memset(&pipeline_compile_options, 0, sizeof(OptixPipelineCompileOptions));

  pipeline_compile_options.usesMotionBlur                   = 0;
  pipeline_compile_options.traversableGraphFlags            = (optix_kernel_configs[type].allow_gas)
                                                                ? OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY
                                                                : OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
  pipeline_compile_options.numPayloadValues                 = 0;
  pipeline_compile_options.numAttributeValues               = 2;
  pipeline_compile_options.exceptionFlags                   = OPTIX_EXCEPTION_FLAG_NONE;
  pipeline_compile_options.pipelineLaunchParamsVariableName = "device";
  pipeline_compile_options.usesPrimitiveTypeFlags           = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;
  pipeline_compile_options.allowOpacityMicromaps            = true;

  char* log;
  __FAILURE_HANDLE(host_malloc(&log, OPTIX_LOG_SIZE));
  memset(log, 0, OPTIX_LOG_SIZE);

  size_t log_size = OPTIX_LOG_SIZE;

  OPTIX_FAILURE_HANDLE_LOG(
    optixModuleCreate(
      device->optix_ctx, &module_compile_options, &pipeline_compile_options, optixir, optixir_length, log, &log_size, &(*kernel)->module),
    log, log_size);

  ////////////////////////////////////////////////////////////////////
  // Group Creation
  ////////////////////////////////////////////////////////////////////

  OptixProgramGroupOptions group_options[OPTIX_KERNEL_NUM_GROUPS];
  memset(&group_options, 0, OPTIX_KERNEL_NUM_GROUPS * sizeof(OptixProgramGroupOptions));

  OptixProgramGroupDesc group_desc[OPTIX_KERNEL_NUM_GROUPS];
  memset(group_desc, 0, OPTIX_KERNEL_NUM_GROUPS * sizeof(OptixProgramGroupDesc));

  group_desc[0].kind                     = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
  group_desc[0].raygen.module            = (*kernel)->module;
  group_desc[0].raygen.entryFunctionName = "__raygen__optix";

  group_desc[1].kind = OPTIX_PROGRAM_GROUP_KIND_MISS;

  for (uint32_t function_id = 0; function_id < OPTIX_KERNEL_FUNCTION_COUNT; function_id++) {
    group_desc[2 + function_id].kind                         = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    group_desc[2 + function_id].hitgroup.moduleAH            = (*kernel)->module;
    group_desc[2 + function_id].hitgroup.entryFunctionNameAH = optix_anyhit_function_names[function_id];
    group_desc[2 + function_id].hitgroup.moduleCH            = (*kernel)->module;
    group_desc[2 + function_id].hitgroup.entryFunctionNameCH = optix_closesthit_function_names[function_id];
  }

  OPTIX_FAILURE_HANDLE_LOG(
    optixProgramGroupCreate(device->optix_ctx, group_desc, OPTIX_KERNEL_NUM_GROUPS, group_options, log, &log_size, (*kernel)->groups), log,
    log_size);

  ////////////////////////////////////////////////////////////////////
  // Pipeline Creation
  ////////////////////////////////////////////////////////////////////

  OptixPipelineLinkOptions pipeline_link_options;
  pipeline_link_options.maxTraceDepth = 1;

  OPTIX_FAILURE_HANDLE_LOG(
    optixPipelineCreate(
      device->optix_ctx, &pipeline_compile_options, &pipeline_link_options, (*kernel)->groups, OPTIX_KERNEL_NUM_GROUPS, log, &log_size,
      &(*kernel)->pipeline),
    log, log_size);

  __FAILURE_HANDLE(host_free(&log));

  ////////////////////////////////////////////////////////////////////
  // Shader Binding Table Creation
  ////////////////////////////////////////////////////////////////////
  __FAILURE_HANDLE(device_malloc(&(*kernel)->records, OPTIX_KERNEL_NUM_GROUPS * OPTIX_SBT_RECORD_HEADER_SIZE));

  char* host_records;
  __FAILURE_HANDLE(host_malloc(&host_records, OPTIX_KERNEL_NUM_GROUPS * OPTIX_SBT_RECORD_HEADER_SIZE + OPTIX_SBT_RECORD_ALIGNMENT));

  // The SBT record must be aligned on the host for some reason, hardcoded aligning here, assert for if that alignment ever changes.
  _STATIC_ASSERT(OPTIX_SBT_RECORD_ALIGNMENT == 16);
  char* records_dst = host_records + (((~((uint64_t) host_records)) + 1) & (OPTIX_SBT_RECORD_ALIGNMENT - 1));

  for (uint32_t i = 0; i < OPTIX_KERNEL_NUM_GROUPS; i++) {
    OPTIX_FAILURE_HANDLE(optixSbtRecordPackHeader((*kernel)->groups[i], records_dst + i * OPTIX_SBT_RECORD_HEADER_SIZE));
  }

  __FAILURE_HANDLE(
    device_upload((*kernel)->records, records_dst, 0, OPTIX_KERNEL_NUM_GROUPS * OPTIX_SBT_RECORD_HEADER_SIZE, device->stream_main));

  __FAILURE_HANDLE(host_free(&host_records));

  memset(&(*kernel)->shaders, 0, sizeof(OptixShaderBindingTable));

  (*kernel)->shaders.raygenRecord       = DEVICE_CUPTR_OFFSET((*kernel)->records, 0 * OPTIX_SBT_RECORD_HEADER_SIZE);
  (*kernel)->shaders.missRecordBase     = DEVICE_CUPTR_OFFSET((*kernel)->records, 1 * OPTIX_SBT_RECORD_HEADER_SIZE);
  (*kernel)->shaders.hitgroupRecordBase = DEVICE_CUPTR_OFFSET((*kernel)->records, 2 * OPTIX_SBT_RECORD_HEADER_SIZE);

  (*kernel)->shaders.missRecordStrideInBytes = OPTIX_SBT_RECORD_HEADER_SIZE;
  (*kernel)->shaders.missRecordCount         = 1;

  (*kernel)->shaders.hitgroupRecordStrideInBytes = OPTIX_SBT_RECORD_HEADER_SIZE;
  (*kernel)->shaders.hitgroupRecordCount         = OPTIX_KERNEL_FUNCTION_COUNT;

  (*kernel)->available = true;

  return LUMINARY_SUCCESS;
}

LuminaryResult optix_kernel_execute(OptixKernel* kernel, Device* device) {
  __CHECK_NULL_ARGUMENT(kernel);
  __CHECK_NULL_ARGUMENT(device);

  uint32_t pixels_per_thread;
  __FAILURE_HANDLE(device_get_current_pixels_per_thread(device, &pixels_per_thread));

  const uint32_t tasks_per_thread = device->constant_memory->config.num_tasks_per_thread;

  const uint32_t max_current_resident_tasks_per_thread = min(tasks_per_thread, pixels_per_thread);

  OPTIX_FAILURE_HANDLE(optixLaunch(
    kernel->pipeline, device->stream_main, device->cuda_device_const_memory, sizeof(DeviceConstantMemory), &kernel->shaders,
    THREADS_PER_BLOCK, device->properties.optimal_block_count, max_current_resident_tasks_per_thread));

  return LUMINARY_SUCCESS;
}

LuminaryResult optix_kernel_destroy(OptixKernel** kernel) {
  __CHECK_NULL_ARGUMENT(kernel);

  __FAILURE_HANDLE(device_free(&(*kernel)->records));

  OPTIX_FAILURE_HANDLE(optixPipelineDestroy((*kernel)->pipeline));

  for (uint32_t group_id = 0; group_id < OPTIX_KERNEL_NUM_GROUPS; group_id++) {
    OPTIX_FAILURE_HANDLE(optixProgramGroupDestroy((*kernel)->groups[group_id]));
  }

  OPTIX_FAILURE_HANDLE(optixModuleDestroy((*kernel)->module))

  __FAILURE_HANDLE(host_free(kernel));

  return LUMINARY_SUCCESS;
}

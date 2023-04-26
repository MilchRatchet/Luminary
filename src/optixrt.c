#include "optixrt.h"

#include <optix.h>
#include <optix_stubs.h>
#include <stdio.h>
#include <stdlib.h>

#define UTILS_NO_DEVICE_TABLE
#define UTILS_NO_DEVICE_FUNCTIONS

#include "bench.h"
#include "buffer.h"
#include "ceb.h"
#include "device.h"
#include "utils.cuh"
#include "utils.h"

#define NUM_HISTOGRAMS 1

#define OPTIX_CHECK_LOGS(call, log)                                                                      \
  {                                                                                                      \
    OptixResult res = call;                                                                              \
                                                                                                         \
    if (res != OPTIX_SUCCESS) {                                                                          \
      error_message("Optix returned message: %s", log);                                                  \
      crash_message("Optix returned error \"%s\"(%d) in call (%s)", optixGetErrorName(res), res, #call); \
    }                                                                                                    \
  }

void optixrt_init(RaytraceInstance* instance) {
  bench_tic();

  ////////////////////////////////////////////////////////////////////
  // Opacity Micromaps Building
  ////////////////////////////////////////////////////////////////////

  OptixBuildInputOpacityMicromap omm = micromap_opacity_build(instance);

  ////////////////////////////////////////////////////////////////////
  // BVH Building
  ////////////////////////////////////////////////////////////////////

  OptixAccelBuildOptions build_options;
  memset(&build_options, 0, sizeof(OptixAccelBuildOptions));
  build_options.operation             = OPTIX_BUILD_OPERATION_BUILD;
  build_options.motionOptions.flags   = OPTIX_MOTION_FLAG_NONE;
  build_options.motionOptions.numKeys = 1;
  build_options.buildFlags            = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE | OPTIX_BUILD_FLAG_ALLOW_COMPACTION;

  OptixBuildInput build_inputs;
  memset(&build_inputs, 0, sizeof(OptixBuildInput));
  build_inputs.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

  build_inputs.triangleArray.vertexFormat        = OPTIX_VERTEX_FORMAT_FLOAT3;
  build_inputs.triangleArray.vertexStrideInBytes = 16;
  build_inputs.triangleArray.numVertices         = instance->scene.triangle_data.vertex_count;
  build_inputs.triangleArray.vertexBuffers       = (CUdeviceptr*) &(instance->scene.triangle_data.vertex_buffer);

  build_inputs.triangleArray.indexFormat        = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
  build_inputs.triangleArray.indexStrideInBytes = 16;
  build_inputs.triangleArray.numIndexTriplets   = instance->scene.triangle_data.triangle_count;
  build_inputs.triangleArray.indexBuffer        = (CUdeviceptr) instance->scene.triangle_data.index_buffer;

  unsigned int inputFlags[1] = {OPTIX_GEOMETRY_FLAG_DISABLE_TRIANGLE_FACE_CULLING};

  build_inputs.triangleArray.flags           = inputFlags;
  build_inputs.triangleArray.opacityMicromap = omm;
  build_inputs.triangleArray.numSbtRecords   = 1;

  OptixAccelBufferSizes buffer_sizes;

  OPTIX_CHECK(optixAccelComputeMemoryUsage(instance->optix_ctx, &build_options, &build_inputs, 1, &buffer_sizes));

  void* temp_buffer;
  device_malloc(&temp_buffer, buffer_sizes.tempSizeInBytes);

  void* output_buffer;
  device_malloc(&output_buffer, buffer_sizes.outputSizeInBytes);

  OptixTraversableHandle traversable;

  OptixAccelEmitDesc accel_emit;
  memset(&accel_emit, 0, sizeof(OptixAccelEmitDesc));

  device_malloc((void**) &accel_emit.result, sizeof(size_t));
  accel_emit.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;

  OPTIX_CHECK(optixAccelBuild(
    instance->optix_ctx, 0, &build_options, &build_inputs, 1, (CUdeviceptr) temp_buffer, buffer_sizes.tempSizeInBytes,
    (CUdeviceptr) output_buffer, buffer_sizes.outputSizeInBytes, &traversable, &accel_emit, 1));

  size_t compact_size;

  device_download(&compact_size, (void*) accel_emit.result, sizeof(size_t));

  device_free((void*) accel_emit.result, sizeof(size_t));
  device_free(temp_buffer, buffer_sizes.tempSizeInBytes);

  if (compact_size < buffer_sizes.outputSizeInBytes) {
    log_message("OptiX BVH is being compacted from size %zu to size %zu", buffer_sizes.outputSizeInBytes, compact_size);
    void* output_buffer_compact;
    device_malloc(&output_buffer_compact, compact_size);

    OPTIX_CHECK(optixAccelCompact(instance->optix_ctx, 0, traversable, (CUdeviceptr) output_buffer_compact, compact_size, &traversable));

    device_free(output_buffer, buffer_sizes.outputSizeInBytes);

    output_buffer                  = output_buffer_compact;
    buffer_sizes.outputSizeInBytes = compact_size;
  }

  instance->optix_bvh.bvh_data    = output_buffer;
  instance->optix_bvh.traversable = traversable;

  micromap_opacity_free(omm);

  ////////////////////////////////////////////////////////////////////
  // Module Compilation
  ////////////////////////////////////////////////////////////////////

  int64_t ptx_length;
  char* ptx;
  uint64_t ptx_info;

  ceb_access("optix_kernels.ptx", (void**) &ptx, &ptx_length, &ptx_info);

  if (ptx_info || !ptx_length) {
    crash_message("Failed to load OptiX kernels. Ceb Error Code: %zu", ptx_info);
  }

  OptixModuleCompileOptions module_compile_options;
  memset(&module_compile_options, 0, sizeof(OptixModuleCompileOptions));

  module_compile_options.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
  module_compile_options.optLevel         = OPTIX_COMPILE_OPTIMIZATION_LEVEL_3;
  module_compile_options.debugLevel       = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

  OptixPipelineCompileOptions pipeline_compile_options;
  memset(&pipeline_compile_options, 0, sizeof(OptixPipelineCompileOptions));

  pipeline_compile_options.usesMotionBlur                   = 0;
  pipeline_compile_options.traversableGraphFlags            = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
  pipeline_compile_options.numPayloadValues                 = 3;
  pipeline_compile_options.numAttributeValues               = 2;
  pipeline_compile_options.exceptionFlags                   = OPTIX_EXCEPTION_FLAG_NONE;
  pipeline_compile_options.pipelineLaunchParamsVariableName = "device";
  pipeline_compile_options.usesPrimitiveTypeFlags           = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;
  pipeline_compile_options.allowOpacityMicromaps            = 1;

  char log[4096];
  size_t log_size = sizeof(log);

  OptixModule module;

  OPTIX_CHECK_LOGS(
    optixModuleCreate(instance->optix_ctx, &module_compile_options, &pipeline_compile_options, ptx, ptx_length, log, &log_size, &module),
    log);

  ////////////////////////////////////////////////////////////////////
  // Group Creation
  ////////////////////////////////////////////////////////////////////

  OptixProgramGroupOptions group_options;
  memset(&group_options, 0, sizeof(OptixProgramGroupOptions));

  OptixProgramGroupDesc group_desc[OPTIXRT_NUM_GROUPS];
  memset(group_desc, 0, OPTIXRT_NUM_GROUPS * sizeof(OptixProgramGroupDesc));

  group_desc[0].kind                     = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
  group_desc[0].raygen.module            = module;
  group_desc[0].raygen.entryFunctionName = "__raygen__optix";

  group_desc[1].kind = OPTIX_PROGRAM_GROUP_KIND_MISS;

  group_desc[2].kind                         = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
  group_desc[2].hitgroup.moduleAH            = module;
  group_desc[2].hitgroup.entryFunctionNameAH = "__anyhit__optix";
  group_desc[2].hitgroup.moduleCH            = module;
  group_desc[2].hitgroup.entryFunctionNameCH = "__closesthit__optix";

  OptixProgramGroup groups[OPTIXRT_NUM_GROUPS];

  OPTIX_CHECK_LOGS(
    optixProgramGroupCreate(instance->optix_ctx, group_desc, OPTIXRT_NUM_GROUPS, &group_options, log, &log_size, groups), log);

  ////////////////////////////////////////////////////////////////////
  // Pipeline Creation
  ////////////////////////////////////////////////////////////////////

  OptixPipelineLinkOptions pipeline_link_options;
  pipeline_link_options.maxTraceDepth = 1;

  OPTIX_CHECK_LOGS(
    optixPipelineCreate(
      instance->optix_ctx, &pipeline_compile_options, &pipeline_link_options, groups, OPTIXRT_NUM_GROUPS, log, &log_size,
      &instance->optix_bvh.pipeline),
    log);

  ////////////////////////////////////////////////////////////////////
  // Shader Binding Table Creation
  ////////////////////////////////////////////////////////////////////

  char* records;
  device_malloc((void**) &records, OPTIXRT_NUM_GROUPS * OPTIX_SBT_RECORD_HEADER_SIZE);

  char host_records[OPTIXRT_NUM_GROUPS * OPTIX_SBT_RECORD_HEADER_SIZE];

  for (int i = 0; i < OPTIXRT_NUM_GROUPS; i++) {
    OPTIX_CHECK(optixSbtRecordPackHeader(groups[i], host_records + i * OPTIX_SBT_RECORD_HEADER_SIZE));
  }

  gpuErrchk(cudaMemcpy(records, host_records, OPTIXRT_NUM_GROUPS * OPTIX_SBT_RECORD_HEADER_SIZE, cudaMemcpyHostToDevice));

  instance->optix_bvh.shaders.raygenRecord       = (CUdeviceptr) (records + 0 * OPTIX_SBT_RECORD_HEADER_SIZE);
  instance->optix_bvh.shaders.missRecordBase     = (CUdeviceptr) (records + 1 * OPTIX_SBT_RECORD_HEADER_SIZE);
  instance->optix_bvh.shaders.hitgroupRecordBase = (CUdeviceptr) (records + 2 * OPTIX_SBT_RECORD_HEADER_SIZE);

  instance->optix_bvh.shaders.missRecordStrideInBytes = OPTIX_SBT_RECORD_HEADER_SIZE;
  instance->optix_bvh.shaders.missRecordCount         = 1;

  instance->optix_bvh.shaders.hitgroupRecordStrideInBytes = OPTIX_SBT_RECORD_HEADER_SIZE;
  instance->optix_bvh.shaders.hitgroupRecordCount         = 1;

  ////////////////////////////////////////////////////////////////////
  // Params Creation
  ////////////////////////////////////////////////////////////////////

  device_malloc((void**) &instance->optix_bvh.params, sizeof(DeviceConstantMemory));

  device_update_symbol(optix_bvh, instance->optix_bvh.traversable);

  instance->optix_bvh.initialized = 1;

  bench_toc("BVH Setup (OptiX)");
}

void optixrt_update_params(RaytraceInstance* instance) {
  // Since this is device -> device, the cost of this is negligble.
  device_gather_device_table(instance->optix_bvh.params, cudaMemcpyDeviceToDevice);
}

void optixrt_execute(RaytraceInstance* instance) {
  if (!instance->optix_bvh.initialized)
    return;

  optixrt_update_params(instance);
  OPTIX_CHECK(optixLaunch(
    instance->optix_bvh.pipeline, 0, (CUdeviceptr) instance->optix_bvh.params, sizeof(DeviceConstantMemory), &instance->optix_bvh.shaders,
    THREADS_PER_BLOCK, BLOCKS_PER_GRID, 1));
}

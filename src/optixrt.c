#include "optixrt.h"

#include <optix.h>
#include <optix_stubs.h>
#include <stdio.h>
#include <stdlib.h>

#define UTILS_NO_DEVICE_TABLE

#include "bench.h"
#include "buffer.h"
#include "ceb.h"
#include "device.h"
#include "utils.h"

#define OPTIX_CHECK_LOGS(call, log)                                                                      \
  {                                                                                                      \
    OptixResult res = call;                                                                              \
                                                                                                         \
    if (res != OPTIX_SUCCESS) {                                                                          \
      error_message("Optix returned message: %s", log);                                                  \
      crash_message("Optix returned error \"%s\"(%d) in call (%s)", optixGetErrorName(res), res, #call); \
    }                                                                                                    \
  }

void optixrt_compile_kernel(const OptixDeviceContext optix_ctx, const char* kernels_name, OptixKernel* kernel, CommandlineOptions options) {
  log_message("Compiling kernels: %s.", kernels_name);

  ////////////////////////////////////////////////////////////////////
  // Module Compilation
  ////////////////////////////////////////////////////////////////////

  int64_t ptx_length;
  char* ptx;
  uint64_t ptx_info;

  ceb_access(kernels_name, (void**) &ptx, &ptx_length, &ptx_info);

  if (ptx_info || !ptx_length) {
    crash_message("Failed to load OptiX kernels %s. Ceb Error Code: %zu", kernels_name, ptx_info);
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
  pipeline_compile_options.numPayloadValues                 = 4;
  pipeline_compile_options.numAttributeValues               = 2;
  pipeline_compile_options.exceptionFlags                   = OPTIX_EXCEPTION_FLAG_NONE;
  pipeline_compile_options.pipelineLaunchParamsVariableName = "device";
  pipeline_compile_options.usesPrimitiveTypeFlags           = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;

  if (options.omm_active)
    pipeline_compile_options.allowOpacityMicromaps = 1;

  if (options.dmm_active)
    pipeline_compile_options.usesPrimitiveTypeFlags |= OPTIX_PRIMITIVE_TYPE_FLAGS_DISPLACED_MICROMESH_TRIANGLE;

  char log[4096];
  size_t log_size = sizeof(log);

  OptixModule module;

  OPTIX_CHECK_LOGS(
    optixModuleCreate(optix_ctx, &module_compile_options, &pipeline_compile_options, ptx, ptx_length, log, &log_size, &module), log);

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

  OPTIX_CHECK_LOGS(optixProgramGroupCreate(optix_ctx, group_desc, OPTIXRT_NUM_GROUPS, &group_options, log, &log_size, groups), log);

  ////////////////////////////////////////////////////////////////////
  // Pipeline Creation
  ////////////////////////////////////////////////////////////////////

  OptixPipelineLinkOptions pipeline_link_options;
  pipeline_link_options.maxTraceDepth = 1;

  OPTIX_CHECK_LOGS(
    optixPipelineCreate(
      optix_ctx, &pipeline_compile_options, &pipeline_link_options, groups, OPTIXRT_NUM_GROUPS, log, &log_size, &kernel->pipeline),
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

  kernel->shaders.raygenRecord       = (CUdeviceptr) (records + 0 * OPTIX_SBT_RECORD_HEADER_SIZE);
  kernel->shaders.missRecordBase     = (CUdeviceptr) (records + 1 * OPTIX_SBT_RECORD_HEADER_SIZE);
  kernel->shaders.hitgroupRecordBase = (CUdeviceptr) (records + 2 * OPTIX_SBT_RECORD_HEADER_SIZE);

  kernel->shaders.missRecordStrideInBytes = OPTIX_SBT_RECORD_HEADER_SIZE;
  kernel->shaders.missRecordCount         = 1;

  kernel->shaders.hitgroupRecordStrideInBytes = OPTIX_SBT_RECORD_HEADER_SIZE;
  kernel->shaders.hitgroupRecordCount         = 1;

  ////////////////////////////////////////////////////////////////////
  // Params Creation
  ////////////////////////////////////////////////////////////////////

  device_malloc(&(kernel->params), sizeof(DeviceConstantMemory));
}

void optixrt_build_bvh(
  OptixDeviceContext optix_ctx, OptixBVH* bvh, const TriangleGeomData tri_data, const OptixBuildInputDisplacementMicromap* dmm,
  const OptixBuildInputOpacityMicromap* omm, const OptixRTBVHType type) {
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
  build_inputs.triangleArray.numVertices         = tri_data.vertex_count;
  build_inputs.triangleArray.vertexBuffers       = (CUdeviceptr*) &(tri_data.vertex_buffer);

  build_inputs.triangleArray.indexFormat        = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
  build_inputs.triangleArray.indexStrideInBytes = 16;
  build_inputs.triangleArray.numIndexTriplets   = tri_data.triangle_count;
  build_inputs.triangleArray.indexBuffer        = (CUdeviceptr) tri_data.index_buffer;

  // OptiX requires pointers to be null if there is no data.
  if (tri_data.vertex_count == 0) {
    build_inputs.triangleArray.vertexBuffers = (CUdeviceptr*) 0;
  }

  if (build_inputs.triangleArray.numIndexTriplets == 0) {
    build_inputs.triangleArray.indexBuffer = (CUdeviceptr) 0;
  }

  unsigned int inputFlags = 0;
  inputFlags |= OPTIX_GEOMETRY_FLAG_DISABLE_TRIANGLE_FACE_CULLING;
  inputFlags |= (type == OPTIX_RT_BVH_TYPE_SHADOW) ? OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL : 0;

  build_inputs.triangleArray.flags         = &inputFlags;
  build_inputs.triangleArray.numSbtRecords = 1;

  if (omm)
    build_inputs.triangleArray.opacityMicromap = *omm;

  if (dmm)
    build_inputs.triangleArray.displacementMicromap = *dmm;

  OptixAccelBufferSizes buffer_sizes;

  OPTIX_CHECK(optixAccelComputeMemoryUsage(optix_ctx, &build_options, &build_inputs, 1, &buffer_sizes));
  gpuErrchk(cudaDeviceSynchronize());

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
    optix_ctx, 0, &build_options, &build_inputs, 1, (CUdeviceptr) temp_buffer, buffer_sizes.tempSizeInBytes, (CUdeviceptr) output_buffer,
    buffer_sizes.outputSizeInBytes, &traversable, &accel_emit, 1));
  gpuErrchk(cudaDeviceSynchronize());

  size_t compact_size;

  device_download(&compact_size, (void*) accel_emit.result, sizeof(size_t));

  device_free((void*) accel_emit.result, sizeof(size_t));
  device_free(temp_buffer, buffer_sizes.tempSizeInBytes);

  if (compact_size < buffer_sizes.outputSizeInBytes) {
    log_message("OptiX BVH is being compacted from size %zu to size %zu", buffer_sizes.outputSizeInBytes, compact_size);
    void* output_buffer_compact;
    device_malloc(&output_buffer_compact, compact_size);

    OPTIX_CHECK(optixAccelCompact(optix_ctx, 0, traversable, (CUdeviceptr) output_buffer_compact, compact_size, &traversable));
    gpuErrchk(cudaDeviceSynchronize());

    device_free(output_buffer, buffer_sizes.outputSizeInBytes);

    output_buffer                  = output_buffer_compact;
    buffer_sizes.outputSizeInBytes = compact_size;
  }

  bvh->bvh_data     = output_buffer;
  bvh->bvh_mem_size = buffer_sizes.outputSizeInBytes;
  bvh->traversable  = traversable;
}

void optixrt_init(RaytraceInstance* instance, CommandlineOptions options) {
  bench_tic("Kernel Setup (OptiX)");

  optixrt_compile_kernel(instance->optix_ctx, (char*) "optix_kernels.ptx", &(instance->optix_kernel), options);
  optixrt_compile_kernel(instance->optix_ctx, (char*) "optix_kernels_trace_particle.ptx", &(instance->particles_instance.kernel), options);
  optixrt_compile_kernel(instance->optix_ctx, (char*) "optix_kernels_geometry.ptx", &(instance->optix_kernel_geometry), options);
  optixrt_compile_kernel(instance->optix_ctx, (char*) "optix_kernels_volume.ptx", &(instance->optix_kernel_volume), options);
  optixrt_compile_kernel(instance->optix_ctx, (char*) "optix_kernels_particle.ptx", &(instance->optix_kernel_particle), options);
  optixrt_compile_kernel(
    instance->optix_ctx, (char*) "optix_kernels_volume_bridges.ptx", &(instance->optix_kernel_volume_bridges), options);

  bench_toc();

  bench_tic("BVH Setup (OptiX)");

  ////////////////////////////////////////////////////////////////////
  // Displacement Micromaps Building
  ////////////////////////////////////////////////////////////////////

  OptixBuildInputDisplacementMicromap dmm;
  if ((instance->device_info.rt_core_version >= 1 && options.dmm_active) || instance->device_info.rt_core_version >= 3) {
    dmm = micromap_displacement_build(instance);
  }
  else {
    log_message("No DMM is built due to device constraints.");
    memset(&dmm, 0, sizeof(OptixBuildInputDisplacementMicromap));
  }

  ////////////////////////////////////////////////////////////////////
  // Opacity Micromaps Building
  ////////////////////////////////////////////////////////////////////

  OptixBuildInputOpacityMicromap omm;
  // OMM and DMM at the same time are only supported on RT core version 3.0 (Ada Lovelace)
  if (
    options.omm_active
    && (((instance->device_info.rt_core_version >= 1 && !dmm.displacementMicromapArray) || instance->device_info.rt_core_version >= 3))) {
    omm = micromap_opacity_build(instance);
  }
  else {
    log_message("No OMM is built due to device constraints or user preference.");
    memset(&omm, 0, sizeof(OptixBuildInputOpacityMicromap));
  }

  ////////////////////////////////////////////////////////////////////
  // BVH Building
  ////////////////////////////////////////////////////////////////////

  optixrt_build_bvh(instance->optix_ctx, &instance->optix_bvh, instance->scene.triangle_data, &dmm, &omm, OPTIX_RT_BVH_TYPE_DEFAULT);
  optixrt_build_bvh(instance->optix_ctx, &instance->optix_bvh_shadow, instance->scene.triangle_data, &dmm, &omm, OPTIX_RT_BVH_TYPE_SHADOW);
  optixrt_build_bvh(
    instance->optix_ctx, &instance->optix_bvh_light, instance->scene.triangle_lights_data, (void*) 0, (void*) 0, OPTIX_RT_BVH_TYPE_DEFAULT);

  micromap_opacity_free(omm);

  device_update_symbol(optix_bvh, instance->optix_bvh.traversable);
  device_update_symbol(optix_bvh_shadow, instance->optix_bvh_shadow.traversable);
  device_update_symbol(optix_bvh_light, instance->optix_bvh_light.traversable);

  instance->optix_bvh.initialized = 1;

  bench_toc();
}

void optixrt_update_params(OptixKernel kernel) {
  // Since this is device -> device, the cost of this is negligble.
  device_gather_device_table(kernel.params, cudaMemcpyDeviceToDevice);
}

void optixrt_execute(OptixKernel kernel) {
  optixrt_update_params(kernel);
  OPTIX_CHECK(optixLaunch(
    kernel.pipeline, 0, (CUdeviceptr) kernel.params, sizeof(DeviceConstantMemory), &kernel.shaders, THREADS_PER_BLOCK, BLOCKS_PER_GRID, 1));
}

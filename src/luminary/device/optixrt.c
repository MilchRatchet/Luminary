#include "optixrt.h"

#include <optix.h>
#include <optix_stubs.h>
#include <stdio.h>
#include <stdlib.h>

#define UTILS_NO_DEVICE_TABLE

#include "ceb.h"
#include "device.h"
#include "device_memory.h"
#include "internal_error.h"
#include "utils.h"

LuminaryResult optixrt_compile_kernel(
  const OptixDeviceContext optix_ctx, const char* kernels_name, OptixKernel* kernel, RendererSettings settings) {
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

  if (settings.use_opacity_micromaps)
    pipeline_compile_options.allowOpacityMicromaps = 1;

  if (settings.use_displacement_micromaps)
    pipeline_compile_options.usesPrimitiveTypeFlags |= OPTIX_PRIMITIVE_TYPE_FLAGS_DISPLACED_MICROMESH_TRIANGLE;

  char log[4096];
  size_t log_size = sizeof(log);

  OptixModule module;

  OPTIX_FAILURE_HANDLE_LOG(
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

  OPTIX_FAILURE_HANDLE_LOG(optixProgramGroupCreate(optix_ctx, group_desc, OPTIXRT_NUM_GROUPS, &group_options, log, &log_size, groups), log);

  ////////////////////////////////////////////////////////////////////
  // Pipeline Creation
  ////////////////////////////////////////////////////////////////////

  OptixPipelineLinkOptions pipeline_link_options;
  pipeline_link_options.maxTraceDepth = 1;

  OPTIX_FAILURE_HANDLE_LOG(
    optixPipelineCreate(
      optix_ctx, &pipeline_compile_options, &pipeline_link_options, groups, OPTIXRT_NUM_GROUPS, log, &log_size, &kernel->pipeline),
    log);

  ////////////////////////////////////////////////////////////////////
  // Shader Binding Table Creation
  ////////////////////////////////////////////////////////////////////

  DEVICE char* records;
  __FAILURE_HANDLE(device_malloc((void**) &records, OPTIXRT_NUM_GROUPS * OPTIX_SBT_RECORD_HEADER_SIZE));

  char host_records[OPTIXRT_NUM_GROUPS * OPTIX_SBT_RECORD_HEADER_SIZE];

  for (int i = 0; i < OPTIXRT_NUM_GROUPS; i++) {
    OPTIX_FAILURE_HANDLE(optixSbtRecordPackHeader(groups[i], host_records + i * OPTIX_SBT_RECORD_HEADER_SIZE));
  }

  __FAILURE_HANDLE(device_upload(records, host_records, 0, OPTIXRT_NUM_GROUPS * OPTIX_SBT_RECORD_HEADER_SIZE));

  kernel->shaders.raygenRecord       = DEVICE_PTR(records) + 0 * OPTIX_SBT_RECORD_HEADER_SIZE;
  kernel->shaders.missRecordBase     = DEVICE_PTR(records) + 1 * OPTIX_SBT_RECORD_HEADER_SIZE;
  kernel->shaders.hitgroupRecordBase = DEVICE_PTR(records) + 2 * OPTIX_SBT_RECORD_HEADER_SIZE;

  kernel->shaders.missRecordStrideInBytes = OPTIX_SBT_RECORD_HEADER_SIZE;
  kernel->shaders.missRecordCount         = 1;

  kernel->shaders.hitgroupRecordStrideInBytes = OPTIX_SBT_RECORD_HEADER_SIZE;
  kernel->shaders.hitgroupRecordCount         = 1;

  ////////////////////////////////////////////////////////////////////
  // Params Creation
  ////////////////////////////////////////////////////////////////////

  __FAILURE_HANDLE(device_malloc(&(kernel->params), sizeof(DeviceConstantMemory)));

  return LUMINARY_SUCCESS;
}

LuminaryResult optixrt_build_bvh(
  OptixDeviceContext optix_ctx, OptixBVH* bvh, const Mesh* mesh, const OptixBuildInputDisplacementMicromap* dmm,
  const OptixBuildInputOpacityMicromap* omm, const OptixRTBVHType type) {
  OptixAccelBuildOptions build_options;
  memset(&build_options, 0, sizeof(OptixAccelBuildOptions));
  build_options.operation             = OPTIX_BUILD_OPERATION_BUILD;
  build_options.motionOptions.flags   = OPTIX_MOTION_FLAG_NONE;
  build_options.motionOptions.numKeys = 1;
  build_options.buildFlags            = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE | OPTIX_BUILD_FLAG_ALLOW_COMPACTION;

  // TODO: I need to create a build input for each meshlets

  uint32_t meshlet_count;
  __FAILURE_HANDLE(array_get_num_elements(mesh->meshlets, &meshlet_count));

  DEVICE float* device_vertex_buffer;
  __FAILURE_HANDLE(device_malloc(&device_vertex_buffer, sizeof(float) * 4 * mesh->data->vertex_count));
  __FAILURE_HANDLE(device_upload(device_vertex_buffer, mesh->data->vertex_buffer, 0, sizeof(float) * 4 * mesh->data->vertex_count));

  CUdeviceptr device_vertex_buffer_ptr = DEVICE_PTR(device_vertex_buffer);

  DEVICE uint32_t** meshlet_device_index_buffers;
  __FAILURE_HANDLE(host_malloc(&meshlet_device_index_buffers, sizeof(DEVICE uint32_t*) * meshlet_count));

  OptixBuildInput* build_inputs;
  __FAILURE_HANDLE(host_malloc(&build_inputs, sizeof(OptixBuildInput) * meshlet_count));

  unsigned int inputFlags = 0;
  inputFlags |= OPTIX_GEOMETRY_FLAG_DISABLE_TRIANGLE_FACE_CULLING;
  inputFlags |= (type == OPTIX_RT_BVH_TYPE_SHADOW) ? OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL : 0;

  for (uint32_t meshlet_id = 0; meshlet_id < meshlet_count; meshlet_id++) {
    const Meshlet meshlet = mesh->meshlets[meshlet_id];

    OptixBuildInput build_input;
    memset(&build_input, 0, sizeof(OptixBuildInput));
    build_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

    build_input.triangleArray.vertexFormat        = OPTIX_VERTEX_FORMAT_FLOAT3;
    build_input.triangleArray.vertexStrideInBytes = 16;
    build_input.triangleArray.numVertices         = mesh->data->vertex_count;
    build_input.triangleArray.vertexBuffers       = &device_vertex_buffer_ptr;

    __FAILURE_HANDLE(device_malloc(&meshlet_device_index_buffers[meshlet_id], sizeof(uint32_t) * 4 * meshlet.triangle_count));
    __FAILURE_HANDLE(
      device_upload(meshlet_device_index_buffers[meshlet_id], meshlet.index_buffer, 0, sizeof(uint32_t) * 4 * meshlet.triangle_count));

    build_input.triangleArray.indexFormat        = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    build_input.triangleArray.indexStrideInBytes = 16;
    build_input.triangleArray.numIndexTriplets   = meshlet.triangle_count;
    build_input.triangleArray.indexBuffer        = DEVICE_PTR(meshlet_device_index_buffers[meshlet_id]);

    build_input.triangleArray.flags         = &inputFlags;
    build_input.triangleArray.numSbtRecords = 1;

    // Lower 12 bits is the triangle id inside the meshlet, upper bits are the meshlet id.
    build_input.triangleArray.primitiveIndexOffset = (meshlet_id << 12);

    build_inputs[meshlet_id] = build_input;
  }

  // OptiX requires pointers to be null if there is no data.
  // TODO: Make sure that we don't need that anymore by requiring that meshes are not empty and that the renderer can handle having no
  // geometry.
#if 0
  if (tri_data.vertex_count == 0) {
    build_inputs.triangleArray.vertexBuffers = (CUdeviceptr*) 0;
  }

  if (build_inputs.triangleArray.numIndexTriplets == 0) {
    build_inputs.triangleArray.indexBuffer = (CUdeviceptr) 0;
  }
#endif

  // TODO: Figure out how to do this now that we have meshlets
#if 0
  if (omm)
    build_inputs.triangleArray.opacityMicromap = *omm;

  if (dmm)
    build_inputs.triangleArray.displacementMicromap = *dmm;
#endif

  OptixAccelBufferSizes buffer_sizes;

  OPTIX_FAILURE_HANDLE(optixAccelComputeMemoryUsage(optix_ctx, &build_options, build_inputs, meshlet_count, &buffer_sizes));
  CUDA_FAILURE_HANDLE(cudaDeviceSynchronize());

  DEVICE void* temp_buffer;
  __FAILURE_HANDLE(device_malloc(&temp_buffer, buffer_sizes.tempSizeInBytes));

  DEVICE void* output_buffer;
  __FAILURE_HANDLE(device_malloc(&output_buffer, buffer_sizes.outputSizeInBytes));

  OptixTraversableHandle traversable;

  DEVICE size_t* accel_emit_buffer;
  __FAILURE_HANDLE(device_malloc(&accel_emit_buffer, sizeof(size_t)));

  OptixAccelEmitDesc accel_emit;
  memset(&accel_emit, 0, sizeof(OptixAccelEmitDesc));

  accel_emit.result = DEVICE_PTR(accel_emit_buffer);
  accel_emit.type   = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;

  OPTIX_FAILURE_HANDLE(optixAccelBuild(
    optix_ctx, 0, &build_options, build_inputs, meshlet_count, DEVICE_PTR(temp_buffer), buffer_sizes.tempSizeInBytes,
    DEVICE_PTR(output_buffer), buffer_sizes.outputSizeInBytes, &traversable, &accel_emit, 1));
  CUDA_FAILURE_HANDLE(cudaDeviceSynchronize());

  size_t compact_size;
  __FAILURE_HANDLE(device_download(&compact_size, accel_emit_buffer, 0, sizeof(size_t)));

  __FAILURE_HANDLE(device_free(&accel_emit_buffer));
  __FAILURE_HANDLE(device_free(&temp_buffer));

  if (compact_size < buffer_sizes.outputSizeInBytes) {
    log_message("OptiX BVH is being compacted from size %zu to size %zu", buffer_sizes.outputSizeInBytes, compact_size);
    DEVICE void* output_buffer_compact;
    __FAILURE_HANDLE(device_malloc(&output_buffer_compact, compact_size));

    OPTIX_FAILURE_HANDLE(optixAccelCompact(optix_ctx, 0, traversable, DEVICE_PTR(output_buffer_compact), compact_size, &traversable));
    CUDA_FAILURE_HANDLE(cudaDeviceSynchronize());

    __FAILURE_HANDLE(device_free(&output_buffer));

    output_buffer                  = output_buffer_compact;
    buffer_sizes.outputSizeInBytes = compact_size;
  }

  // Cleanup
  __FAILURE_HANDLE(host_free(&build_inputs));
  __FAILURE_HANDLE(device_free(&device_vertex_buffer));

  for (uint32_t meshlet_id = 0; meshlet_id < meshlet_count; meshlet_id++) {
    __FAILURE_HANDLE(device_free(&meshlet_device_index_buffers[meshlet_id]));
  }

  __FAILURE_HANDLE(host_free(&meshlet_device_index_buffers));

  bvh->bvh_data     = output_buffer;
  bvh->bvh_mem_size = buffer_sizes.outputSizeInBytes;
  bvh->traversable  = traversable;

  return LUMINARY_SUCCESS;
}

LuminaryResult optixrt_init(Device* device, RendererSettings settings) {
  optixrt_compile_kernel(device->optix_ctx, (char*) "optix_kernels.ptx", device->optix_kernel, settings);
  optixrt_compile_kernel(device->optix_ctx, (char*) "optix_kernels_trace_particle.ptx", device->particles_instance.kernel, settings);
  optixrt_compile_kernel(device->optix_ctx, (char*) "optix_kernels_geometry.ptx", device->optix_kernel_geometry, settings);
  optixrt_compile_kernel(device->optix_ctx, (char*) "optix_kernels_volume.ptx", device->optix_kernel_volume, settings);
  optixrt_compile_kernel(device->optix_ctx, (char*) "optix_kernels_particle.ptx", device->optix_kernel_particle, settings);
  optixrt_compile_kernel(device->optix_ctx, (char*) "optix_kernels_volume_bridges.ptx", device->optix_kernel_volume_bridges, settings);

  ////////////////////////////////////////////////////////////////////
  // Displacement Micromaps Building
  ////////////////////////////////////////////////////////////////////

  OptixBuildInputDisplacementMicromap dmm;
  if ((device->properties.rt_core_version >= 1 && settings.use_opacity_micromaps) || device->properties.rt_core_version >= 3) {
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
  // OMM and DMM at the same time are only supported on RT core version 3.0 or later (Ada+)
  if (
    settings.use_opacity_micromaps
    && (((device->properties.rt_core_version >= 1 && !dmm.displacementMicromapArray) || device->properties.rt_core_version >= 3))) {
    omm = micromap_opacity_build(instance);
  }
  else {
    log_message("No OMM is built due to device constraints or user preference.");
    memset(&omm, 0, sizeof(OptixBuildInputOpacityMicromap));
  }

  ////////////////////////////////////////////////////////////////////
  // BVH Building
  ////////////////////////////////////////////////////////////////////

  // TODO: This must be separate

  optixrt_build_bvh(instance->optix_ctx, &instance->optix_bvh, instance->scene.triangle_data, &dmm, &omm, OPTIX_RT_BVH_TYPE_DEFAULT);
  optixrt_build_bvh(instance->optix_ctx, &instance->optix_bvh_shadow, instance->scene.triangle_data, &dmm, &omm, OPTIX_RT_BVH_TYPE_SHADOW);
  optixrt_build_bvh(
    instance->optix_ctx, &instance->optix_bvh_light, instance->scene.triangle_lights_data, (void*) 0, (void*) 0, OPTIX_RT_BVH_TYPE_DEFAULT);

  micromap_opacity_free(omm);

  device_update_symbol(optix_bvh, instance->optix_bvh.traversable);
  device_update_symbol(optix_bvh_shadow, instance->optix_bvh_shadow.traversable);
  device_update_symbol(optix_bvh_light, instance->optix_bvh_light.traversable);

  instance->optix_bvh.initialized = 1;

  return LUMINARY_SUCCESS;
}

LuminaryResult optixrt_update_params(OptixKernel kernel) {
  // Since this is device -> device, the cost of this is negligble.
  // TODO: Only do this if dirty, we only need to update the sample slice.
  device_gather_device_table(kernel.params, cudaMemcpyDeviceToDevice);

  return LUMINARY_SUCCESS;
}

LuminaryResult optixrt_execute(OptixKernel kernel) {
  optixrt_update_params(kernel);
  OPTIX_FAILURE_HANDLE(optixLaunch(
    kernel.pipeline, 0, (CUdeviceptr) kernel.params, sizeof(DeviceConstantMemory), &kernel.shaders, THREADS_PER_BLOCK, BLOCKS_PER_GRID, 1));

  return LUMINARY_SUCCESS;
}

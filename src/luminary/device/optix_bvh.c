#include "optix_bvh.h"

#include <optix.h>
#include <optix_stubs.h>
#include <stdio.h>
#include <stdlib.h>

#include "ceb.h"
#include "device.h"
#include "device_memory.h"
#include "host_math.h"
#include "internal_error.h"
#include "utils.h"

static LuminaryResult _optix_bvh_compute_transform(Quaternion rotation, vec3 scale, vec3 offset, float transformation[12]) {
  const float x2    = rotation.x * rotation.x;
  const float y2    = rotation.y * rotation.y;
  const float z2    = rotation.z * rotation.z;
  const float w2    = rotation.w * rotation.w;
  const float xy    = rotation.x * rotation.y;
  const float xz    = rotation.x * rotation.z;
  const float xw    = rotation.x * rotation.w;
  const float yz    = rotation.y * rotation.z;
  const float yw    = rotation.y * rotation.w;
  const float zw    = rotation.z * rotation.w;
  const float denom = 2.0f / (x2 + y2 + z2 + w2);

  transformation[0]  = scale.x * (1.0f - denom * (y2 + z2));
  transformation[1]  = scale.y * (denom * (xy + zw));
  transformation[2]  = scale.z * (denom * (xz - yw));
  transformation[3]  = offset.x;
  transformation[4]  = scale.x * (denom * (xy - zw));
  transformation[5]  = scale.y * (1.0f - denom * (x2 + z2));
  transformation[6]  = scale.z * (denom * (yz + xw));
  transformation[7]  = offset.y;
  transformation[8]  = scale.x * (denom * (xz + yw));
  transformation[9]  = scale.y * (denom * (yz - xw));
  transformation[10] = scale.z * (1.0f - denom * (x2 + y2));
  transformation[11] = offset.z;

  return LUMINARY_SUCCESS;
}

static LuminaryResult _optix_bvh_get_optix_instance(
  const Device* device, const MeshInstance* instance, uint32_t instance_id, OptixInstance* optix_instance, OptixBVHType type) {
  __CHECK_NULL_ARGUMENT(device);
  __CHECK_NULL_ARGUMENT(device->meshes);
  __CHECK_NULL_ARGUMENT(instance);
  __CHECK_NULL_ARGUMENT(optix_instance);

  memset(optix_instance, 0, sizeof(OptixInstance));

  optix_instance->instanceId        = instance_id;
  optix_instance->sbtOffset         = 0;
  optix_instance->visibilityMask    = (instance->active) ? ((1u << (device->optix_properties.num_bits_instance_visibility_mask)) - 1) : 0;
  optix_instance->flags             = OPTIX_INSTANCE_FLAG_DISABLE_TRIANGLE_FACE_CULLING;
  optix_instance->traversableHandle = device->meshes[instance->mesh_id]->bvh->traversable[type];

  const Quaternion rotation = rotation_euler_angles_to_quaternion(instance->rotation);

  __FAILURE_HANDLE(_optix_bvh_compute_transform(rotation, instance->scale, instance->translation, optix_instance->transform));

  return LUMINARY_SUCCESS;
}

LuminaryResult optix_bvh_instance_cache_create(OptixBVHInstanceCache** cache, Device* device) {
  __CHECK_NULL_ARGUMENT(cache);
  __CHECK_NULL_ARGUMENT(device);

  __FAILURE_HANDLE(host_malloc(cache, sizeof(OptixBVHInstanceCache)));

  for (uint32_t type = 0; type < OPTIX_BVH_TYPE_COUNT; type++) {
    __FAILURE_HANDLE(device_malloc(&(*cache)->instances[type], 16 * sizeof(OptixInstance)));
  }

  (*cache)->num_instances           = 0;
  (*cache)->num_instances_allocated = 16;

  (*cache)->device = device;

  return LUMINARY_SUCCESS;
}

LuminaryResult optix_bvh_instance_cache_update(OptixBVHInstanceCache* cache, const ARRAY MeshInstanceUpdate* instance_updates) {
  __CHECK_NULL_ARGUMENT(cache);
  __CHECK_NULL_ARGUMENT(instance_updates);

  uint32_t num_updates;
  __FAILURE_HANDLE(array_get_num_elements(instance_updates, &num_updates));

  uint32_t num_instances_after = cache->num_instances;

  // Compute the number of instances we will have afterwards
  for (uint32_t update_id = 0; update_id < num_updates; update_id++) {
    const MeshInstanceUpdate* update = instance_updates + update_id;

    if (update->instance_id >= num_instances_after)
      num_instances_after = update->instance_id + 1;
  }

  // Reallocate the instances buffer if it is not large enough.
  if (num_instances_after > cache->num_instances_allocated) {
    cache->num_instances_allocated = num_instances_after * 2;

    for (uint32_t type = 0; type < OPTIX_BVH_TYPE_COUNT; type++) {
      DEVICE OptixInstance* new_instances;
      __FAILURE_HANDLE(device_malloc(&new_instances, sizeof(OptixInstance) * cache->num_instances_allocated));

      OptixInstance* direct_access_buffer;
      __FAILURE_HANDLE(device_staging_manager_register_direct_access(
        cache->device->staging_manager, new_instances, 0, sizeof(OptixInstance) * cache->num_instances, (void**) &direct_access_buffer));

      __FAILURE_HANDLE(device_download(
        direct_access_buffer, cache->instances[type], 0, sizeof(OptixInstance) * cache->num_instances, cache->device->stream_main));

      __FAILURE_HANDLE(device_free(&cache->instances[type]));

      cache->instances[type] = new_instances;
    }
  }

  cache->num_instances = num_instances_after;

  // Stage the new optix instances
  for (uint32_t update_id = 0; update_id < num_updates; update_id++) {
    const MeshInstanceUpdate* update = instance_updates + update_id;

    for (uint32_t type = 0; type < OPTIX_BVH_TYPE_COUNT; type++) {
      OptixInstance* direct_access_buffer;
      __FAILURE_HANDLE(device_staging_manager_register_direct_access(
        cache->device->staging_manager, cache->instances[type], sizeof(OptixInstance) * update->instance_id, sizeof(OptixInstance),
        (void**) &direct_access_buffer));

      __FAILURE_HANDLE(_optix_bvh_get_optix_instance(cache->device, &update->instance, update->instance_id, direct_access_buffer, type));
    }
  }

  return LUMINARY_SUCCESS;
}

LuminaryResult optix_bvh_instance_cache_destroy(OptixBVHInstanceCache** cache) {
  __CHECK_NULL_ARGUMENT(cache);
  __CHECK_NULL_ARGUMENT(*cache);

  for (uint32_t type = 0; type < OPTIX_BVH_TYPE_COUNT; type++) {
    __FAILURE_HANDLE(device_free(&(*cache)->instances[type]));
  }

  __FAILURE_HANDLE(host_free(cache));

  return LUMINARY_SUCCESS;
}

LuminaryResult optix_bvh_create(OptixBVH** bvh) {
  __CHECK_NULL_ARGUMENT(bvh);

  __FAILURE_HANDLE(host_malloc(bvh, sizeof(OptixBVH)));
  memset(*bvh, 0, sizeof(OptixBVH));

  for (uint32_t type = 0; type < OPTIX_BVH_TYPE_COUNT; type++) {
    (*bvh)->allocated_mask[type] = false;
  }

  (*bvh)->fast_trace = false;

  return LUMINARY_SUCCESS;
}

static LuminaryResult _optix_bvh_free(OptixBVH* bvh) {
  __CHECK_NULL_ARGUMENT(bvh);

  for (uint32_t type = 0; type < OPTIX_BVH_TYPE_COUNT; type++) {
    if (bvh->allocated_mask[type]) {
      __FAILURE_HANDLE(device_free(&bvh->bvh_data[type]));
      bvh->allocated_mask[type] = false;
    }
  }

  return LUMINARY_SUCCESS;
}

LuminaryResult optix_bvh_gas_build(OptixBVH* bvh, Device* device, const Mesh* mesh, const OpacityMicromap* omm) {
  __CHECK_NULL_ARGUMENT(bvh);
  __CHECK_NULL_ARGUMENT(device);
  __CHECK_NULL_ARGUMENT(mesh);
  __CHECK_NULL_ARGUMENT(omm);

  __FAILURE_HANDLE(_optix_bvh_free(bvh));

  ////////////////////////////////////////////////////////////////////
  // Setting up build input
  ////////////////////////////////////////////////////////////////////

  OptixAccelBuildOptions build_options;
  memset(&build_options, 0, sizeof(OptixAccelBuildOptions));
  build_options.operation             = OPTIX_BUILD_OPERATION_BUILD;
  build_options.motionOptions.flags   = OPTIX_MOTION_FLAG_NONE;
  build_options.motionOptions.numKeys = 1;
  build_options.buildFlags            = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE | OPTIX_BUILD_FLAG_ALLOW_COMPACTION;

  OptixBuildInput build_input;
  memset(&build_input, 0, sizeof(OptixBuildInput));
  build_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

  DEVICE float* device_vertex_buffer;
  __FAILURE_HANDLE(device_malloc(&device_vertex_buffer, sizeof(float) * 4 * mesh->data.vertex_count));
  __FAILURE_HANDLE(
    device_upload(device_vertex_buffer, mesh->data.vertex_buffer, 0, sizeof(float) * 4 * mesh->data.vertex_count, device->stream_main));

  CUdeviceptr device_vertex_buffer_ptr = DEVICE_CUPTR(device_vertex_buffer);

  build_input.triangleArray.vertexFormat        = OPTIX_VERTEX_FORMAT_FLOAT3;
  build_input.triangleArray.vertexStrideInBytes = 16;
  build_input.triangleArray.numVertices         = mesh->data.vertex_count;
  build_input.triangleArray.vertexBuffers       = &device_vertex_buffer_ptr;

  DEVICE uint32_t* device_index_buffer;
  __FAILURE_HANDLE(device_malloc(&device_index_buffer, sizeof(uint32_t) * 4 * mesh->data.triangle_count));
  __FAILURE_HANDLE(
    device_upload(device_index_buffer, mesh->data.index_buffer, 0, sizeof(uint32_t) * 4 * mesh->data.triangle_count, device->stream_main));

  build_input.triangleArray.indexFormat        = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
  build_input.triangleArray.indexStrideInBytes = 16;
  build_input.triangleArray.numIndexTriplets   = mesh->data.triangle_count;
  build_input.triangleArray.indexBuffer        = DEVICE_CUPTR(device_index_buffer);

  for (uint32_t type = 0; type < OPTIX_BVH_TYPE_COUNT; type++) {
    unsigned int inputFlags = 0;
    inputFlags |= OPTIX_GEOMETRY_FLAG_DISABLE_TRIANGLE_FACE_CULLING;
    inputFlags |= (type == OPTIX_BVH_TYPE_SHADOW) ? OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL : 0;

    build_input.triangleArray.flags         = &inputFlags;
    build_input.triangleArray.numSbtRecords = 1;

    // OMMs is only available on GPUs with RT cores
    if (device->optix_properties.rtcore_version > 0)
      build_input.triangleArray.opacityMicromap = omm->optix_build_input;

    ////////////////////////////////////////////////////////////////////
    // Building BVH
    ////////////////////////////////////////////////////////////////////

    OptixAccelBufferSizes buffer_sizes;
    OPTIX_FAILURE_HANDLE(optixAccelComputeMemoryUsage(device->optix_ctx, &build_options, &build_input, 1, &buffer_sizes));

    DEVICE void* temp_buffer;
    __FAILURE_HANDLE(device_malloc(&temp_buffer, buffer_sizes.tempSizeInBytes));

    DEVICE void* output_buffer;
    __FAILURE_HANDLE(device_malloc(&output_buffer, buffer_sizes.outputSizeInBytes));

    OptixTraversableHandle traversable;

    DEVICE size_t* accel_emit_buffer;
    __FAILURE_HANDLE(device_malloc(&accel_emit_buffer, sizeof(size_t)));

    OptixAccelEmitDesc accel_emit;
    memset(&accel_emit, 0, sizeof(OptixAccelEmitDesc));

    accel_emit.result = DEVICE_CUPTR(accel_emit_buffer);
    accel_emit.type   = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;

    OPTIX_FAILURE_HANDLE(optixAccelBuild(
      device->optix_ctx, device->stream_main, &build_options, &build_input, 1, DEVICE_CUPTR(temp_buffer), buffer_sizes.tempSizeInBytes,
      DEVICE_CUPTR(output_buffer), buffer_sizes.outputSizeInBytes, &traversable, &accel_emit, 1));

    size_t compact_size;
    __FAILURE_HANDLE(device_download(&compact_size, accel_emit_buffer, 0, sizeof(size_t), device->stream_main));

    __FAILURE_HANDLE(device_free(&accel_emit_buffer));
    __FAILURE_HANDLE(device_free(&temp_buffer));

    if (compact_size < buffer_sizes.outputSizeInBytes) {
      log_message("OptiX BVH is being compacted from size %zu to size %zu", buffer_sizes.outputSizeInBytes, compact_size);
      DEVICE void* output_buffer_compact;
      __FAILURE_HANDLE(device_malloc(&output_buffer_compact, compact_size));

      OPTIX_FAILURE_HANDLE(optixAccelCompact(
        device->optix_ctx, device->stream_main, traversable, DEVICE_CUPTR(output_buffer_compact), compact_size, &traversable));

      __FAILURE_HANDLE(device_free(&output_buffer));

      output_buffer                  = output_buffer_compact;
      buffer_sizes.outputSizeInBytes = compact_size;
    }

    bvh->bvh_data[type]       = output_buffer;
    bvh->traversable[type]    = traversable;
    bvh->allocated_mask[type] = true;
  }

  ////////////////////////////////////////////////////////////////////
  // Clean up
  ////////////////////////////////////////////////////////////////////

  __FAILURE_HANDLE(device_free(&device_vertex_buffer));
  __FAILURE_HANDLE(device_free(&device_index_buffer));

  bvh->fast_trace = (build_options.buildFlags & OPTIX_BUILD_FLAG_PREFER_FAST_TRACE);

  return LUMINARY_SUCCESS;
}

LuminaryResult optix_bvh_ias_build(OptixBVH* bvh, Device* device) {
  __CHECK_NULL_ARGUMENT(bvh);
  __CHECK_NULL_ARGUMENT(device);
  __CHECK_NULL_ARGUMENT(device->optix_instance_cache);

  __FAILURE_HANDLE(_optix_bvh_free(bvh));

  if (device->optix_instance_cache->num_instances == 0)
    return LUMINARY_SUCCESS;

  ////////////////////////////////////////////////////////////////////
  // Setting up build input
  ////////////////////////////////////////////////////////////////////

  OptixAccelBuildOptions build_options;
  memset(&build_options, 0, sizeof(OptixAccelBuildOptions));
  build_options.operation             = OPTIX_BUILD_OPERATION_BUILD;
  build_options.motionOptions.flags   = OPTIX_MOTION_FLAG_NONE;
  build_options.motionOptions.numKeys = 1;
  build_options.buildFlags            = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE | OPTIX_BUILD_FLAG_ALLOW_COMPACTION;

  __FAILURE_HANDLE(device_staging_manager_execute(device->staging_manager));

  for (uint32_t type = 0; type < OPTIX_BVH_TYPE_COUNT; type++) {
    OptixBuildInput build_input;
    memset(&build_input, 0, sizeof(OptixBuildInput));
    build_input.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;

    build_input.instanceArray.instances      = DEVICE_CUPTR(device->optix_instance_cache->instances[type]);
    build_input.instanceArray.instanceStride = sizeof(OptixInstance);
    build_input.instanceArray.numInstances   = device->optix_instance_cache->num_instances;

    ////////////////////////////////////////////////////////////////////
    // Building BVH
    ////////////////////////////////////////////////////////////////////

    OptixAccelBufferSizes buffer_sizes;

    OPTIX_FAILURE_HANDLE(optixAccelComputeMemoryUsage(device->optix_ctx, &build_options, &build_input, 1, &buffer_sizes));

    DEVICE void* temp_buffer;
    __FAILURE_HANDLE(device_malloc(&temp_buffer, buffer_sizes.tempSizeInBytes));

    DEVICE void* output_buffer;
    __FAILURE_HANDLE(device_malloc(&output_buffer, buffer_sizes.outputSizeInBytes));

    OptixTraversableHandle traversable;

    DEVICE size_t* accel_emit_buffer;
    __FAILURE_HANDLE(device_malloc(&accel_emit_buffer, sizeof(size_t)));

    OptixAccelEmitDesc accel_emit;
    memset(&accel_emit, 0, sizeof(OptixAccelEmitDesc));

    accel_emit.result = DEVICE_CUPTR(accel_emit_buffer);
    accel_emit.type   = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;

    OPTIX_FAILURE_HANDLE(optixAccelBuild(
      device->optix_ctx, device->stream_main, &build_options, &build_input, 1, DEVICE_CUPTR(temp_buffer), buffer_sizes.tempSizeInBytes,
      DEVICE_CUPTR(output_buffer), buffer_sizes.outputSizeInBytes, &traversable, &accel_emit, 1));

    size_t compact_size;
    __FAILURE_HANDLE(device_download(&compact_size, accel_emit_buffer, 0, sizeof(size_t), device->stream_main));

    __FAILURE_HANDLE(device_free(&accel_emit_buffer));
    __FAILURE_HANDLE(device_free(&temp_buffer));

    if (compact_size < buffer_sizes.outputSizeInBytes) {
      log_message("OptiX BVH is being compacted from size %zu to size %zu", buffer_sizes.outputSizeInBytes, compact_size);
      DEVICE void* output_buffer_compact;
      __FAILURE_HANDLE(device_malloc(&output_buffer_compact, compact_size));

      OPTIX_FAILURE_HANDLE(optixAccelCompact(
        device->optix_ctx, device->stream_main, traversable, DEVICE_CUPTR(output_buffer_compact), compact_size, &traversable));

      __FAILURE_HANDLE(device_free(&output_buffer));

      output_buffer                  = output_buffer_compact;
      buffer_sizes.outputSizeInBytes = compact_size;
    }

    bvh->bvh_data[type]       = output_buffer;
    bvh->traversable[type]    = traversable;
    bvh->allocated_mask[type] = true;
  }

  ////////////////////////////////////////////////////////////////////
  // Finalize
  ////////////////////////////////////////////////////////////////////

  bvh->fast_trace = (build_options.buildFlags & OPTIX_BUILD_FLAG_PREFER_FAST_TRACE);

  return LUMINARY_SUCCESS;
}

LuminaryResult optix_bvh_light_build(OptixBVH* bvh, Device* device, const LightTree* tree) {
  __CHECK_NULL_ARGUMENT(bvh);
  __CHECK_NULL_ARGUMENT(device);
  __CHECK_NULL_ARGUMENT(tree);

  __FAILURE_HANDLE(_optix_bvh_free(bvh));

  ////////////////////////////////////////////////////////////////////
  // Setting up build input
  ////////////////////////////////////////////////////////////////////

  OptixAccelBuildOptions build_options;
  memset(&build_options, 0, sizeof(OptixAccelBuildOptions));
  build_options.operation             = OPTIX_BUILD_OPERATION_BUILD;
  build_options.motionOptions.flags   = OPTIX_MOTION_FLAG_NONE;
  build_options.motionOptions.numKeys = 1;
  build_options.buildFlags            = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE | OPTIX_BUILD_FLAG_ALLOW_COMPACTION;

  OptixBuildInput build_input;
  memset(&build_input, 0, sizeof(OptixBuildInput));
  build_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

  const uint32_t vertex_count = 3 * tree->light_count;

  DEVICE float* device_vertex_buffer;
  __FAILURE_HANDLE(device_malloc(&device_vertex_buffer, sizeof(float) * 4 * vertex_count));
  __FAILURE_HANDLE(
    device_upload(device_vertex_buffer, tree->bvh_vertex_buffer_data, 0, sizeof(float) * 4 * vertex_count, device->stream_main));

  CUdeviceptr device_vertex_buffer_ptr = DEVICE_CUPTR(device_vertex_buffer);

  build_input.triangleArray.vertexFormat        = OPTIX_VERTEX_FORMAT_FLOAT3;
  build_input.triangleArray.vertexStrideInBytes = 16;
  build_input.triangleArray.numVertices         = vertex_count;
  build_input.triangleArray.vertexBuffers       = (vertex_count > 0) ? &device_vertex_buffer_ptr : (CUdeviceptr*) 0;

  build_input.triangleArray.indexFormat        = OPTIX_INDICES_FORMAT_NONE;
  build_input.triangleArray.indexStrideInBytes = 0;
  build_input.triangleArray.numIndexTriplets   = 0;
  build_input.triangleArray.indexBuffer        = (CUdeviceptr) 0;

  unsigned int inputFlags = 0;
  inputFlags |= OPTIX_GEOMETRY_FLAG_DISABLE_TRIANGLE_FACE_CULLING;

  build_input.triangleArray.flags         = &inputFlags;
  build_input.triangleArray.numSbtRecords = 1;

  ////////////////////////////////////////////////////////////////////
  // Building BVH
  ////////////////////////////////////////////////////////////////////

  OptixAccelBufferSizes buffer_sizes;
  OPTIX_FAILURE_HANDLE(optixAccelComputeMemoryUsage(device->optix_ctx, &build_options, &build_input, 1, &buffer_sizes));

  DEVICE void* temp_buffer;
  __FAILURE_HANDLE(device_malloc(&temp_buffer, buffer_sizes.tempSizeInBytes));

  DEVICE void* output_buffer;
  __FAILURE_HANDLE(device_malloc(&output_buffer, buffer_sizes.outputSizeInBytes));

  OptixTraversableHandle traversable;

  DEVICE size_t* accel_emit_buffer;
  __FAILURE_HANDLE(device_malloc(&accel_emit_buffer, sizeof(size_t)));

  OptixAccelEmitDesc accel_emit;
  memset(&accel_emit, 0, sizeof(OptixAccelEmitDesc));

  accel_emit.result = DEVICE_CUPTR(accel_emit_buffer);
  accel_emit.type   = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;

  OPTIX_FAILURE_HANDLE(optixAccelBuild(
    device->optix_ctx, device->stream_main, &build_options, &build_input, 1, DEVICE_CUPTR(temp_buffer), buffer_sizes.tempSizeInBytes,
    DEVICE_CUPTR(output_buffer), buffer_sizes.outputSizeInBytes, &traversable, &accel_emit, 1));

  size_t compact_size;
  __FAILURE_HANDLE(device_download(&compact_size, accel_emit_buffer, 0, sizeof(size_t), device->stream_main));

  __FAILURE_HANDLE(device_free(&accel_emit_buffer));
  __FAILURE_HANDLE(device_free(&temp_buffer));

  if (compact_size < buffer_sizes.outputSizeInBytes) {
    log_message("OptiX BVH is being compacted from size %zu to size %zu", buffer_sizes.outputSizeInBytes, compact_size);
    DEVICE void* output_buffer_compact;
    __FAILURE_HANDLE(device_malloc(&output_buffer_compact, compact_size));

    OPTIX_FAILURE_HANDLE(optixAccelCompact(
      device->optix_ctx, device->stream_main, traversable, DEVICE_CUPTR(output_buffer_compact), compact_size, &traversable));

    __FAILURE_HANDLE(device_free(&output_buffer));

    output_buffer                  = output_buffer_compact;
    buffer_sizes.outputSizeInBytes = compact_size;
  }

  bvh->bvh_data[OPTIX_BVH_TYPE_DEFAULT]       = output_buffer;
  bvh->traversable[OPTIX_BVH_TYPE_DEFAULT]    = traversable;
  bvh->allocated_mask[OPTIX_BVH_TYPE_DEFAULT] = true;

  ////////////////////////////////////////////////////////////////////
  // Clean up
  ////////////////////////////////////////////////////////////////////

  __FAILURE_HANDLE(device_free(&device_vertex_buffer));

  bvh->fast_trace = (build_options.buildFlags & OPTIX_BUILD_FLAG_PREFER_FAST_TRACE);

  return LUMINARY_SUCCESS;
}

LuminaryResult optix_bvh_particles_gas_build(OptixBVH* bvh, Device* device, const DeviceParticlesHandle* particles_handle) {
  __CHECK_NULL_ARGUMENT(bvh);
  __CHECK_NULL_ARGUMENT(device);
  __CHECK_NULL_ARGUMENT(particles_handle);

  __FAILURE_HANDLE(_optix_bvh_free(bvh));

  ////////////////////////////////////////////////////////////////////
  // Setting up build input
  ////////////////////////////////////////////////////////////////////

  OptixAccelBuildOptions build_options;
  memset(&build_options, 0, sizeof(OptixAccelBuildOptions));
  build_options.operation             = OPTIX_BUILD_OPERATION_BUILD;
  build_options.motionOptions.flags   = OPTIX_MOTION_FLAG_NONE;
  build_options.motionOptions.numKeys = 1;
  build_options.buildFlags            = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE | OPTIX_BUILD_FLAG_ALLOW_COMPACTION;

  OptixBuildInput build_input;
  memset(&build_input, 0, sizeof(OptixBuildInput));
  build_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

  const uint32_t vertex_count = particles_handle->count * 6;

  CUdeviceptr device_vertex_buffer_ptr = DEVICE_CUPTR(particles_handle->vertex_buffer);

  build_input.triangleArray.vertexFormat        = OPTIX_VERTEX_FORMAT_FLOAT3;
  build_input.triangleArray.vertexStrideInBytes = 16;
  build_input.triangleArray.numVertices         = vertex_count;
  build_input.triangleArray.vertexBuffers       = (vertex_count > 0) ? &device_vertex_buffer_ptr : (CUdeviceptr*) 0;

  build_input.triangleArray.indexFormat        = OPTIX_INDICES_FORMAT_NONE;
  build_input.triangleArray.indexStrideInBytes = 0;
  build_input.triangleArray.numIndexTriplets   = 0;
  build_input.triangleArray.indexBuffer        = (CUdeviceptr) 0;

  unsigned int inputFlags = 0;
  inputFlags |= OPTIX_GEOMETRY_FLAG_DISABLE_TRIANGLE_FACE_CULLING;

  build_input.triangleArray.flags         = &inputFlags;
  build_input.triangleArray.numSbtRecords = 1;

  ////////////////////////////////////////////////////////////////////
  // Building BVH
  ////////////////////////////////////////////////////////////////////

  OptixAccelBufferSizes buffer_sizes;
  OPTIX_FAILURE_HANDLE(optixAccelComputeMemoryUsage(device->optix_ctx, &build_options, &build_input, 1, &buffer_sizes));

  DEVICE void* temp_buffer;
  __FAILURE_HANDLE(device_malloc(&temp_buffer, buffer_sizes.tempSizeInBytes));

  DEVICE void* output_buffer;
  __FAILURE_HANDLE(device_malloc(&output_buffer, buffer_sizes.outputSizeInBytes));

  OptixTraversableHandle traversable;

  DEVICE size_t* accel_emit_buffer;
  __FAILURE_HANDLE(device_malloc(&accel_emit_buffer, sizeof(size_t)));

  OptixAccelEmitDesc accel_emit;
  memset(&accel_emit, 0, sizeof(OptixAccelEmitDesc));

  accel_emit.result = DEVICE_CUPTR(accel_emit_buffer);
  accel_emit.type   = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;

  OPTIX_FAILURE_HANDLE(optixAccelBuild(
    device->optix_ctx, device->stream_main, &build_options, &build_input, 1, DEVICE_CUPTR(temp_buffer), buffer_sizes.tempSizeInBytes,
    DEVICE_CUPTR(output_buffer), buffer_sizes.outputSizeInBytes, &traversable, &accel_emit, 1));

  size_t compact_size;
  __FAILURE_HANDLE(device_download(&compact_size, accel_emit_buffer, 0, sizeof(size_t), device->stream_main));

  __FAILURE_HANDLE(device_free(&accel_emit_buffer));
  __FAILURE_HANDLE(device_free(&temp_buffer));

  if (compact_size < buffer_sizes.outputSizeInBytes) {
    log_message("OptiX BVH is being compacted from size %zu to size %zu", buffer_sizes.outputSizeInBytes, compact_size);
    DEVICE void* output_buffer_compact;
    __FAILURE_HANDLE(device_malloc(&output_buffer_compact, compact_size));

    OPTIX_FAILURE_HANDLE(optixAccelCompact(
      device->optix_ctx, device->stream_main, traversable, DEVICE_CUPTR(output_buffer_compact), compact_size, &traversable));

    __FAILURE_HANDLE(device_free(&output_buffer));

    output_buffer                  = output_buffer_compact;
    buffer_sizes.outputSizeInBytes = compact_size;
  }

  bvh->bvh_data[OPTIX_BVH_TYPE_DEFAULT]       = output_buffer;
  bvh->traversable[OPTIX_BVH_TYPE_DEFAULT]    = traversable;
  bvh->allocated_mask[OPTIX_BVH_TYPE_DEFAULT] = true;

  ////////////////////////////////////////////////////////////////////
  // Clean up
  ////////////////////////////////////////////////////////////////////

  bvh->fast_trace = (build_options.buildFlags & OPTIX_BUILD_FLAG_PREFER_FAST_TRACE);

  return LUMINARY_SUCCESS;
}

LuminaryResult optix_bvh_particles_ias_build(OptixBVH* bvh, Device* device, const DeviceParticlesHandle* particles_handle) {
  __CHECK_NULL_ARGUMENT(bvh);
  __CHECK_NULL_ARGUMENT(device);
  __CHECK_NULL_ARGUMENT(particles_handle);

  __FAILURE_HANDLE(_optix_bvh_free(bvh));

  ////////////////////////////////////////////////////////////////////
  // Setting up build input
  ////////////////////////////////////////////////////////////////////

  OptixAccelBuildOptions build_options;
  memset(&build_options, 0, sizeof(OptixAccelBuildOptions));
  build_options.operation             = OPTIX_BUILD_OPERATION_BUILD;
  build_options.motionOptions.flags   = OPTIX_MOTION_FLAG_NONE;
  build_options.motionOptions.numKeys = 1;
  build_options.buildFlags            = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE | OPTIX_BUILD_FLAG_ALLOW_COMPACTION;

  __FAILURE_HANDLE(device_staging_manager_execute(device->staging_manager));

  for (uint32_t type = 0; type < OPTIX_BVH_TYPE_COUNT; type++) {
    OptixBuildInput build_input;
    memset(&build_input, 0, sizeof(OptixBuildInput));
    build_input.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;

    build_input.instanceArray.instances      = DEVICE_CUPTR(particles_handle->optix_instances);
    build_input.instanceArray.instanceStride = sizeof(OptixInstance);
    build_input.instanceArray.numInstances   = particles_handle->num_instances;

    ////////////////////////////////////////////////////////////////////
    // Building BVH
    ////////////////////////////////////////////////////////////////////

    OptixAccelBufferSizes buffer_sizes;

    OPTIX_FAILURE_HANDLE(optixAccelComputeMemoryUsage(device->optix_ctx, &build_options, &build_input, 1, &buffer_sizes));

    DEVICE void* temp_buffer;
    __FAILURE_HANDLE(device_malloc(&temp_buffer, buffer_sizes.tempSizeInBytes));

    DEVICE void* output_buffer;
    __FAILURE_HANDLE(device_malloc(&output_buffer, buffer_sizes.outputSizeInBytes));

    OptixTraversableHandle traversable;

    DEVICE size_t* accel_emit_buffer;
    __FAILURE_HANDLE(device_malloc(&accel_emit_buffer, sizeof(size_t)));

    OptixAccelEmitDesc accel_emit;
    memset(&accel_emit, 0, sizeof(OptixAccelEmitDesc));

    accel_emit.result = DEVICE_CUPTR(accel_emit_buffer);
    accel_emit.type   = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;

    OPTIX_FAILURE_HANDLE(optixAccelBuild(
      device->optix_ctx, device->stream_main, &build_options, &build_input, 1, DEVICE_CUPTR(temp_buffer), buffer_sizes.tempSizeInBytes,
      DEVICE_CUPTR(output_buffer), buffer_sizes.outputSizeInBytes, &traversable, &accel_emit, 1));

    size_t compact_size;
    __FAILURE_HANDLE(device_download(&compact_size, accel_emit_buffer, 0, sizeof(size_t), device->stream_main));

    __FAILURE_HANDLE(device_free(&accel_emit_buffer));
    __FAILURE_HANDLE(device_free(&temp_buffer));

    if (compact_size < buffer_sizes.outputSizeInBytes) {
      log_message("OptiX BVH is being compacted from size %zu to size %zu", buffer_sizes.outputSizeInBytes, compact_size);
      DEVICE void* output_buffer_compact;
      __FAILURE_HANDLE(device_malloc(&output_buffer_compact, compact_size));

      OPTIX_FAILURE_HANDLE(optixAccelCompact(
        device->optix_ctx, device->stream_main, traversable, DEVICE_CUPTR(output_buffer_compact), compact_size, &traversable));

      __FAILURE_HANDLE(device_free(&output_buffer));

      output_buffer                  = output_buffer_compact;
      buffer_sizes.outputSizeInBytes = compact_size;
    }

    bvh->bvh_data[type]       = output_buffer;
    bvh->traversable[type]    = traversable;
    bvh->allocated_mask[type] = true;
  }

  ////////////////////////////////////////////////////////////////////
  // Finalize
  ////////////////////////////////////////////////////////////////////

  bvh->fast_trace = (build_options.buildFlags & OPTIX_BUILD_FLAG_PREFER_FAST_TRACE);

  return LUMINARY_SUCCESS;
}

LuminaryResult optix_bvh_destroy(OptixBVH** bvh) {
  __CHECK_NULL_ARGUMENT(bvh);
  __CHECK_NULL_ARGUMENT(*bvh);

  __FAILURE_HANDLE(_optix_bvh_free(*bvh));

  __FAILURE_HANDLE(host_free(bvh));

  return LUMINARY_SUCCESS;
}

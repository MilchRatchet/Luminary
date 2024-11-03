#include "optix_bvh.h"

#include <optix.h>
#include <optix_stubs.h>
#include <stdio.h>
#include <stdlib.h>

#include "ceb.h"
#include "device.h"
#include "device_memory.h"
#include "internal_error.h"
#include "utils.h"

LuminaryResult optix_bvh_create(OptixBVH** bvh, Device* device, const Mesh* mesh, OptixRTBVHType type) {
  __CHECK_NULL_ARGUMENT(bvh);
  __CHECK_NULL_ARGUMENT(device);
  __CHECK_NULL_ARGUMENT(mesh);

  __FAILURE_HANDLE(host_malloc(bvh, sizeof(OptixBVH)));

  ////////////////////////////////////////////////////////////////////
  // Setting up build input
  ////////////////////////////////////////////////////////////////////

  OptixAccelBuildOptions build_options;
  memset(&build_options, 0, sizeof(OptixAccelBuildOptions));
  build_options.operation             = OPTIX_BUILD_OPERATION_BUILD;
  build_options.motionOptions.flags   = OPTIX_MOTION_FLAG_NONE;
  build_options.motionOptions.numKeys = 1;
  build_options.buildFlags            = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE | OPTIX_BUILD_FLAG_ALLOW_COMPACTION;

  uint32_t meshlet_count;
  __FAILURE_HANDLE(array_get_num_elements(mesh->meshlets, &meshlet_count));

  DEVICE float* device_vertex_buffer;
  __FAILURE_HANDLE(device_malloc(&device_vertex_buffer, sizeof(float) * 4 * mesh->data->vertex_count));
  __FAILURE_HANDLE(device_upload(
    device_vertex_buffer, mesh->data->vertex_buffer, 0, sizeof(float) * 4 * mesh->data->vertex_count, device->stream_secondary));

  CUdeviceptr device_vertex_buffer_ptr = DEVICE_CUPTR(device_vertex_buffer);

  DEVICE uint32_t** meshlet_device_index_buffers;
  __FAILURE_HANDLE(host_malloc(&meshlet_device_index_buffers, sizeof(DEVICE uint32_t*) * meshlet_count));

  OptixBuildInput* build_inputs;
  __FAILURE_HANDLE(host_malloc(&build_inputs, sizeof(OptixBuildInput) * meshlet_count));

  unsigned int inputFlags = 0;
  inputFlags |= OPTIX_GEOMETRY_FLAG_DISABLE_TRIANGLE_FACE_CULLING;
  inputFlags |= (type == OPTIX_BVH_TYPE_SHADOW) ? OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL : 0;

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
    __FAILURE_HANDLE(device_upload(
      meshlet_device_index_buffers[meshlet_id], meshlet.index_buffer, 0, sizeof(uint32_t) * 4 * meshlet.triangle_count,
      device->stream_secondary));

    build_input.triangleArray.indexFormat        = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    build_input.triangleArray.indexStrideInBytes = 16;
    build_input.triangleArray.numIndexTriplets   = meshlet.triangle_count;
    build_input.triangleArray.indexBuffer        = DEVICE_CUPTR(meshlet_device_index_buffers[meshlet_id]);

    build_input.triangleArray.flags         = &inputFlags;
    build_input.triangleArray.numSbtRecords = 1;

    // Lower 16 bits is the triangle id inside the meshlet, upper bits are the meshlet id.
    build_input.triangleArray.primitiveIndexOffset = (meshlet_id << 16);

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

  ////////////////////////////////////////////////////////////////////
  // Building BVH
  ////////////////////////////////////////////////////////////////////

  OptixAccelBufferSizes buffer_sizes;

  OPTIX_FAILURE_HANDLE(optixAccelComputeMemoryUsage(device->optix_ctx, &build_options, build_inputs, meshlet_count, &buffer_sizes));

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
    device->optix_ctx, device->stream_secondary, &build_options, build_inputs, meshlet_count, DEVICE_CUPTR(temp_buffer),
    buffer_sizes.tempSizeInBytes, DEVICE_CUPTR(output_buffer), buffer_sizes.outputSizeInBytes, &traversable, &accel_emit, 1));

  size_t compact_size;
  __FAILURE_HANDLE(device_download(&compact_size, accel_emit_buffer, 0, sizeof(size_t), device->stream_secondary));
  CUDA_FAILURE_HANDLE(cuStreamSynchronize(device->stream_secondary));

  __FAILURE_HANDLE(device_free(&accel_emit_buffer));
  __FAILURE_HANDLE(device_free(&temp_buffer));

  if (compact_size < buffer_sizes.outputSizeInBytes) {
    log_message("OptiX BVH is being compacted from size %zu to size %zu", buffer_sizes.outputSizeInBytes, compact_size);
    DEVICE void* output_buffer_compact;
    __FAILURE_HANDLE(device_malloc(&output_buffer_compact, compact_size));

    OPTIX_FAILURE_HANDLE(optixAccelCompact(
      device->optix_ctx, device->stream_secondary, traversable, DEVICE_CUPTR(output_buffer_compact), compact_size, &traversable));

    __FAILURE_HANDLE(device_free(&output_buffer));

    output_buffer                  = output_buffer_compact;
    buffer_sizes.outputSizeInBytes = compact_size;
  }

  ////////////////////////////////////////////////////////////////////
  // Clean up
  ////////////////////////////////////////////////////////////////////

  __FAILURE_HANDLE(host_free(&build_inputs));
  __FAILURE_HANDLE(device_free(&device_vertex_buffer));

  for (uint32_t meshlet_id = 0; meshlet_id < meshlet_count; meshlet_id++) {
    __FAILURE_HANDLE(device_free(&meshlet_device_index_buffers[meshlet_id]));
  }

  __FAILURE_HANDLE(host_free(&meshlet_device_index_buffers));

  (*bvh)->bvh_data    = output_buffer;
  (*bvh)->traversable = traversable;

  return LUMINARY_SUCCESS;
}

LuminaryResult optix_bvh_destroy(OptixBVH** bvh) {
  __CHECK_NULL_ARGUMENT(bvh);

  __FAILURE_HANDLE(device_free(&(*bvh)->bvh_data));

  __FAILURE_HANDLE(host_free(bvh));

  return LUMINARY_SUCCESS;
}

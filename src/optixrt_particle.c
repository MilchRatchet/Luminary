#include "optixrt_particle.h"

#include "bench.h"
#include "buffer.h"
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

void optixrt_particle_init(RaytraceInstance* instance) {
  bench_tic("Particles BVH Setup (OptiX)");

  OptixAccelBuildOptions build_options;
  memset(&build_options, 0, sizeof(OptixAccelBuildOptions));
  build_options.operation             = OPTIX_BUILD_OPERATION_BUILD;
  build_options.motionOptions.flags   = OPTIX_MOTION_FLAG_NONE;
  build_options.motionOptions.numKeys = 1;
  build_options.buildFlags            = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE | OPTIX_BUILD_FLAG_ALLOW_COMPACTION;

  OptixBuildInput build_inputs;
  memset(&build_inputs, 0, sizeof(OptixBuildInput));
  build_inputs.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

  CUdeviceptr vertex_buffer = (CUdeviceptr) device_buffer_get_pointer(instance->particles_instance.vertex_buffer);
  CUdeviceptr index_buffer  = (CUdeviceptr) device_buffer_get_pointer(instance->particles_instance.index_buffer);

  build_inputs.triangleArray.vertexFormat        = OPTIX_VERTEX_FORMAT_FLOAT3;
  build_inputs.triangleArray.vertexStrideInBytes = 16;
  build_inputs.triangleArray.numVertices         = instance->particles_instance.vertex_count;
  build_inputs.triangleArray.vertexBuffers       = &vertex_buffer;

  build_inputs.triangleArray.indexFormat        = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
  build_inputs.triangleArray.indexStrideInBytes = 12;
  build_inputs.triangleArray.numIndexTriplets   = instance->particles_instance.triangle_count;
  build_inputs.triangleArray.indexBuffer        = index_buffer;

  unsigned int inputFlags[1] = {OPTIX_GEOMETRY_FLAG_DISABLE_TRIANGLE_FACE_CULLING};

  build_inputs.triangleArray.flags         = inputFlags;
  build_inputs.triangleArray.numSbtRecords = 1;

  OptixAccelBufferSizes buffer_sizes;

  OPTIX_CHECK(optixAccelComputeMemoryUsage(instance->optix_ctx, &build_options, &build_inputs, 1, &buffer_sizes));
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
    instance->optix_ctx, 0, &build_options, &build_inputs, 1, (CUdeviceptr) temp_buffer, buffer_sizes.tempSizeInBytes,
    (CUdeviceptr) output_buffer, buffer_sizes.outputSizeInBytes, &traversable, &accel_emit, 1));
  gpuErrchk(cudaDeviceSynchronize());

  size_t compact_size;

  device_download(&compact_size, (void*) accel_emit.result, sizeof(size_t));

  device_free((void*) accel_emit.result, sizeof(size_t));
  device_free(temp_buffer, buffer_sizes.tempSizeInBytes);

  if (compact_size < buffer_sizes.outputSizeInBytes) {
    log_message("OptiX BVH is being compacted from size %zu to size %zu", buffer_sizes.outputSizeInBytes, compact_size);
    void* output_buffer_compact;
    device_malloc(&output_buffer_compact, compact_size);

    OPTIX_CHECK(optixAccelCompact(instance->optix_ctx, 0, traversable, (CUdeviceptr) output_buffer_compact, compact_size, &traversable));
    gpuErrchk(cudaDeviceSynchronize());

    device_free(output_buffer, buffer_sizes.outputSizeInBytes);

    output_buffer                  = output_buffer_compact;
    buffer_sizes.outputSizeInBytes = compact_size;
  }

  instance->particles_instance.optix.bvh_data    = output_buffer;
  instance->particles_instance.optix.traversable = traversable;

  device_update_symbol(optix_bvh_particles, instance->particles_instance.optix.traversable);

  instance->particles_instance.optix.initialized = 1;

  bench_toc();
}

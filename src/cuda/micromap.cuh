#ifndef CU_MICROMAP_H
#define CU_MICROMAP_H

#include <optix.h>
#include <optix_stubs.h>

#include "buffer.h"
#include "device.h"
#include "utils.cuh"

#define OPACITY_MICROMAP_STATE_SIZE(__level__, __format__) \
  (((1 << (__level__ * 2)) * ((__format__ == OPTIX_OPACITY_MICROMAP_FORMAT_2_STATE) ? 1 : 2) + 7) / 8)

__device__ int micromap_get_opacity(const uint32_t t_id, const uint32_t level, const uint32_t mt_id) {
  return OPTIX_OPACITY_MICROMAP_STATE_OPAQUE;
}

//
// This kernel computes a level 0 format 4 base micromap array.
// The triangles are grouped in 4 so that the state of the four triangles make 1 byte.
//
__global__ void micromap_opacity_level_0_format_4(uint8_t* dst, uint8_t* level_record) {
  int id                        = threadIdx.x + blockIdx.x * blockDim.x;
  const uint32_t triangle_count = (device.scene.triangle_data.triangle_count + 3) / 4;

  while (id < triangle_count) {
    uint8_t* ptr = dst + id;

    const int opacity0 = micromap_get_opacity(4 * id + 0, 0, 0);
    const int opacity1 = micromap_get_opacity(4 * id + 1, 0, 0);
    const int opacity2 = micromap_get_opacity(4 * id + 2, 0, 0);
    const int opacity3 = micromap_get_opacity(4 * id + 3, 0, 0);

    if (opacity0 ^ 0b10)
      level_record[4 * id + 0] = 0;
    if (opacity1 ^ 0b10)
      level_record[4 * id + 1] = 0;
    if (opacity2 ^ 0b10)
      level_record[4 * id + 2] = 0;
    if (opacity3 ^ 0b10)
      level_record[4 * id + 3] = 0;

    uint8_t v = 0;

    v |= (opacity0 << 6);
    v |= (opacity1 << 4);
    v |= (opacity2 << 2);
    v |= (opacity3 << 0);

    ptr[id] = v;

    id += blockDim.x * gridDim.x;
  }
}

static size_t _micromap_opacity_get_micromap_array_size(
  const uint32_t count, const uint32_t level, const OptixOpacityMicromapFormat format) {
  if (format != OPTIX_OPACITY_MICROMAP_FORMAT_2_STATE && format != OPTIX_OPACITY_MICROMAP_FORMAT_4_STATE) {
    return 0;
  }

  if (level == 0) {
    if (format == OPTIX_OPACITY_MICROMAP_FORMAT_2_STATE) {
      return (count + 7) / 8;
    }

    if (format == OPTIX_OPACITY_MICROMAP_FORMAT_4_STATE) {
      return (count + 3) / 4;
    }
  }

  if (level == 1) {
    if (format == OPTIX_OPACITY_MICROMAP_FORMAT_2_STATE) {
      return (count + 1) / 2;
    }
  }

  const uint32_t state_size = OPACITY_MICROMAP_STATE_SIZE(level, format);

  return state_size * count;
}

OptixBuildInputOpacityMicromap micromap_opacity_build(RaytraceInstance* instance) {
  const uint32_t level                    = 0;
  const OptixOpacityMicromapFormat format = OPTIX_OPACITY_MICROMAP_FORMAT_4_STATE;

  const uint32_t max_level = 9;
  uint32_t num_levels      = 0;

  uint32_t* triangles_per_level = (uint32_t*) calloc(1, sizeof(uint32_t) * max_level);

  void* data;
  device_malloc(&data, _micromap_opacity_get_micromap_array_size(instance->scene.triangle_data.triangle_count, level, format));

  // For each triangle, we store the final level, 0xFF specifies that the triangle has not reached its final level yet
  uint8_t* triangle_level = (uint8_t*) malloc(instance->scene.triangle_data.triangle_count);
  void* triangle_level_buffer;
  device_malloc(&triangle_level_buffer, instance->scene.triangle_data.triangle_count);

  for (uint32_t i = 0; i < instance->scene.triangle_data.triangle_count; i++) {
    triangle_level[i] = 0xFF;
  }
  device_upload(triangle_level_buffer, triangle_level, instance->scene.triangle_data.triangle_count);

  micromap_opacity_level_0_format_4<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>((uint8_t*) data, (uint8_t*) triangle_level_buffer);
  gpuErrchk(cudaDeviceSynchronize());

  device_download(triangle_level, triangle_level_buffer, instance->scene.triangle_data.triangle_count);
  gpuErrchk(cudaDeviceSynchronize());

  uint32_t remaining_triangles = 0;
  for (uint32_t i = 0; i < instance->scene.triangle_data.triangle_count; i++) {
    if (triangle_level[i] == 0xFF)
      remaining_triangles++;
  }

  triangles_per_level[num_levels] = instance->scene.triangle_data.triangle_count - remaining_triangles;

  num_levels++;

  while (remaining_triangles && num_levels < max_level) {
    // Refine triangles
    device_download(triangle_level, triangle_level_buffer, instance->scene.triangle_data.triangle_count);
    gpuErrchk(cudaDeviceSynchronize());

    for (uint32_t i = 0; i < instance->scene.triangle_data.triangle_count; i++) {
      if (triangle_level[i] == 0xFF)
        remaining_triangles++;
    }

    triangles_per_level[num_levels] = triangles_per_level[num_levels - 1] - remaining_triangles;

    num_levels++;
  }

  OptixOpacityMicromapHistogramEntry* histogram =
    (OptixOpacityMicromapHistogramEntry*) malloc(sizeof(OptixOpacityMicromapHistogramEntry) * num_levels);
  OptixOpacityMicromapUsageCount* usage = (OptixOpacityMicromapUsageCount*) malloc(sizeof(OptixOpacityMicromapUsageCount) * num_levels);
  OptixOpacityMicromapDesc* desc        = (OptixOpacityMicromapDesc*) malloc(sizeof(OptixOpacityMicromapDesc) * num_levels);
  for (uint32_t i = 0; i < num_levels; i++) {
    histogram[i].count            = triangles_per_level[i];
    histogram[i].subdivisionLevel = i;
    histogram[i].format           = format;

    usage[i].count            = triangles_per_level[i];
    usage[i].subdivisionLevel = i;
    usage[i].format           = format;

    desc[i].byteOffset       = 0;
    desc[i].subdivisionLevel = i;
    desc[i].format           = format;
  }

  free(triangles_per_level);
  free(triangle_level);
  device_free(triangle_level_buffer, instance->scene.triangle_data.triangle_count);

  void* desc_buffer;
  device_malloc(&desc_buffer, sizeof(OptixOpacityMicromapDesc) * num_levels);
  device_upload(desc_buffer, desc, sizeof(OptixOpacityMicromapDesc) * num_levels);

  OptixOpacityMicromapArrayBuildInput array_build_input;
  memset(&array_build_input, 0, sizeof(OptixOpacityMicromapArrayBuildInput));

  array_build_input.flags                        = OPTIX_OPACITY_MICROMAP_FLAG_PREFER_FAST_TRACE;
  array_build_input.inputBuffer                  = (CUdeviceptr) data;
  array_build_input.numMicromapHistogramEntries  = num_levels;
  array_build_input.micromapHistogramEntries     = histogram;
  array_build_input.perMicromapDescBuffer        = (CUdeviceptr) desc_buffer;
  array_build_input.perMicromapDescStrideInBytes = sizeof(OptixOpacityMicromapDesc);

  OptixMicromapBufferSizes buffer_sizes;
  memset(&buffer_sizes, 0, sizeof(OptixMicromapBufferSizes));

  OPTIX_CHECK(optixOpacityMicromapArrayComputeMemoryUsage(instance->optix_ctx, &array_build_input, &buffer_sizes));

  void* output_buffer;
  device_malloc(&output_buffer, buffer_sizes.outputSizeInBytes);
  void* temp_buffer;
  device_malloc(&temp_buffer, buffer_sizes.tempSizeInBytes);

  OptixMicromapBuffers buffers;
  memset(&buffers, 0, sizeof(OptixMicromapBuffers));

  buffers.output            = (CUdeviceptr) output_buffer;
  buffers.outputSizeInBytes = buffer_sizes.outputSizeInBytes;
  buffers.temp              = (CUdeviceptr) temp_buffer;
  buffers.tempSizeInBytes   = buffer_sizes.tempSizeInBytes;

  OPTIX_CHECK(optixOpacityMicromapArrayBuild(instance->optix_ctx, 0, &array_build_input, &buffers));

  device_free(desc_buffer, sizeof(OptixOpacityMicromapDesc) * num_levels);
  device_free(temp_buffer, buffer_sizes.tempSizeInBytes);

  OptixBuildInputOpacityMicromap bvh_input;
  memset(&bvh_input, 0, sizeof(OptixBuildInputOpacityMicromap));

  bvh_input.indexingMode           = OPTIX_OPACITY_MICROMAP_ARRAY_INDEXING_MODE_LINEAR;
  bvh_input.opacityMicromapArray   = (CUdeviceptr) output_buffer;
  bvh_input.numMicromapUsageCounts = num_levels;
  bvh_input.micromapUsageCounts    = usage;

  return bvh_input;
}

void micromap_opacity_free(OptixBuildInputOpacityMicromap data) {
  free((void*) data.micromapUsageCounts);
}

#endif /* CU_MICROMAP_H */

#ifndef CU_MICROMAP_H
#define CU_MICROMAP_H

#include <optix.h>
#include <optix_micromap.h>
#include <optix_stubs.h>

#include "buffer.h"
#include "device.h"
#include "memory.cuh"
#include "utils.cuh"

#define OPACITY_MICROMAP_STATE_SIZE(__level__, __format__) \
  (((1 << (__level__ * 2)) * ((__format__ == OPTIX_OPACITY_MICROMAP_FORMAT_2_STATE) ? 1 : 2) + 7) / 8)

// Load triangle only once for the refinement steps
__device__ int micromap_get_opacity(const uint32_t t_id, const uint32_t level, const uint32_t mt_id) {
  const float* t_ptr = (float*) (device.scene.triangles + t_id);

  float2 data0 = __ldg((float2*) (t_ptr + 18));
  float4 data1 = __ldg((float4*) (t_ptr + 20));

  uint32_t object_map = __ldg((uint32_t*) (t_ptr + 24));
  uint16_t tex        = device.scene.texture_assignments[object_map].albedo_map;

  if (tex == TEXTURE_NONE) {
    return OPTIX_OPACITY_MICROMAP_STATE_OPAQUE;
  }

  const DeviceTexture dev_tex = device.ptrs.albedo_atlas[tex];

  const UV vertex_texture = get_UV(data0.x, data0.y);
  const UV edge1_texture  = get_UV(data1.x, data1.y);
  const UV edge2_texture  = get_UV(data1.z, data1.w);

  float2 bary0;
  float2 bary1;
  float2 bary2;
  optixMicromapIndexToBaseBarycentrics(mt_id, level, bary0, bary1, bary2);

  const UV uv0 = lerp_uv(vertex_texture, edge1_texture, edge2_texture, bary0);
  const UV uv1 = lerp_uv(vertex_texture, edge1_texture, edge2_texture, bary1);
  const UV uv2 = lerp_uv(vertex_texture, edge1_texture, edge2_texture, bary2);

  const float max_u = fmaxf(uv0.u, fmaxf(uv1.u, uv2.u));
  const float min_u = fminf(uv0.u, fminf(uv1.u, uv2.u));
  const float max_v = fmaxf(uv0.v, fmaxf(uv1.v, uv2.v));
  const float min_v = fminf(uv0.v, fminf(uv1.v, uv2.v));

  bool found_opaque      = false;
  bool found_transparent = false;

  for (float v = min_v; v <= max_v; v += dev_tex.inv_height) {
    for (float u = min_u; u <= max_u; u += dev_tex.inv_width) {
      float4 value = tex2D<float4>(dev_tex.tex, u, 1.0f - v);

      if (value.w > 0.0f)
        found_opaque = true;

      if (value.w < 1.0f)
        found_transparent = true;
    }
  }

  if (found_transparent && !found_opaque)
    return OPTIX_OPACITY_MICROMAP_STATE_TRANSPARENT;

  if (found_opaque && !found_transparent)
    return OPTIX_OPACITY_MICROMAP_STATE_OPAQUE;

  return OPTIX_OPACITY_MICROMAP_STATE_UNKNOWN_OPAQUE;
}

//
// This kernel computes a level 0 format 4 base micromap array.
//
__global__ void micromap_opacity_level_0_format_4(uint8_t* dst, uint8_t* level_record) {
  int id                        = threadIdx.x + blockIdx.x * blockDim.x;
  const uint32_t triangle_count = device.scene.triangle_data.triangle_count;

  while (id < triangle_count) {
    const int opacity = micromap_get_opacity(id, 0, 0);

    const uint8_t v = opacity;

    if (opacity ^ 0b10)
      level_record[id] = 0;

    dst[id] = v;

    id += blockDim.x * gridDim.x;
  }
}

static size_t _micromap_opacity_get_micromap_array_size(
  const uint32_t count, const uint32_t level, const OptixOpacityMicromapFormat format) {
  if (format != OPTIX_OPACITY_MICROMAP_FORMAT_2_STATE && format != OPTIX_OPACITY_MICROMAP_FORMAT_4_STATE) {
    return 0;
  }

  // OMMs are byte aligned, hence even the low subdivision levels are at least 1 byte in size
  const uint32_t state_size = OPACITY_MICROMAP_STATE_SIZE(level, format);

  return state_size * count;
}

OptixBuildInputOpacityMicromap micromap_opacity_build(RaytraceInstance* instance) {
  const uint32_t level                    = 0;
  const OptixOpacityMicromapFormat format = OPTIX_OPACITY_MICROMAP_FORMAT_4_STATE;

  // Highest allowed level is 12 according to OptiX Docs
  const uint32_t max_num_levels = 1;
  uint32_t num_levels           = 0;

  uint32_t* triangles_per_level = (uint32_t*) calloc(1, sizeof(uint32_t) * max_num_levels);

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

  while (remaining_triangles && num_levels < max_num_levels) {
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

  // Some triangles needed more refinement but max level was reached.
  if (remaining_triangles) {
    triangles_per_level[num_levels - 1] += remaining_triangles;
  }

  OptixOpacityMicromapHistogramEntry* histogram =
    (OptixOpacityMicromapHistogramEntry*) malloc(sizeof(OptixOpacityMicromapHistogramEntry) * num_levels);
  OptixOpacityMicromapUsageCount* usage = (OptixOpacityMicromapUsageCount*) malloc(sizeof(OptixOpacityMicromapUsageCount) * num_levels);
  for (uint32_t i = 0; i < num_levels; i++) {
    histogram[i].count            = triangles_per_level[i];
    histogram[i].subdivisionLevel = i;
    histogram[i].format           = format;

    usage[i].count            = triangles_per_level[i];
    usage[i].subdivisionLevel = i;
    usage[i].format           = format;
  }

  OptixOpacityMicromapDesc* desc =
    (OptixOpacityMicromapDesc*) malloc(sizeof(OptixOpacityMicromapDesc) * instance->scene.triangle_data.triangle_count);
  for (uint32_t i = 0; i < instance->scene.triangle_data.triangle_count; i++) {
    desc[i].byteOffset       = i;
    desc[i].subdivisionLevel = 0;
    desc[i].format           = format;
  }

  free(triangles_per_level);
  free(triangle_level);
  device_free(triangle_level_buffer, instance->scene.triangle_data.triangle_count);

  void* desc_buffer;
  device_malloc(&desc_buffer, sizeof(OptixOpacityMicromapDesc) * instance->scene.triangle_data.triangle_count);
  device_upload(desc_buffer, desc, sizeof(OptixOpacityMicromapDesc) * instance->scene.triangle_data.triangle_count);

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

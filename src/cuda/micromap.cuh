#ifndef CU_MICROMAP_H
#define CU_MICROMAP_H

#include <optix.h>
#include <optix_micromap.h>
#include <optix_stubs.h>

#include "buffer.h"
#include "device.h"
#include "memory.cuh"
#include "utils.cuh"

#define OMM_STATE_SIZE(__level__, __format__) \
  (((1 << (__level__ * 2)) * ((__format__ == OPTIX_OPACITY_MICROMAP_FORMAT_2_STATE) ? 1 : 2) + 7) / 8)

struct OMMTextureTriangle {
  UV vertex;
  UV edge1;
  UV edge2;
  DeviceTexture tex;
  uint16_t tex_id;
} typedef OMMTextureTriangle;

__device__ OMMTextureTriangle micromap_get_ommtexturetriangle(const uint32_t id) {
  const float* t_ptr = (float*) (device.scene.triangles + id);

  uint32_t object_map = __ldg((uint32_t*) (t_ptr + 24));
  uint16_t tex        = device.scene.texture_assignments[object_map].albedo_map;

  OMMTextureTriangle tri;
  tri.tex_id = tex;

  if (tex == TEXTURE_NONE) {
    return tri;
  }

  float2 data0 = __ldg((float2*) (t_ptr + 18));
  float4 data1 = __ldg((float4*) (t_ptr + 20));

  tri.tex = device.ptrs.albedo_atlas[tex];

  tri.vertex = get_UV(data0.x, data0.y);
  tri.edge1  = get_UV(data1.x, data1.y);
  tri.edge2  = get_UV(data1.z, data1.w);

  return tri;
}

// Load triangle only once for the refinement steps
__device__ int micromap_get_opacity(const OMMTextureTriangle tri, const uint32_t level, const uint32_t mt_id) {
  if (tri.tex_id == TEXTURE_NONE) {
    return OPTIX_OPACITY_MICROMAP_STATE_OPAQUE;
  }

  float2 bary0;
  float2 bary1;
  float2 bary2;
  optixMicromapIndexToBaseBarycentrics(mt_id, level, bary0, bary1, bary2);

  const UV uv0 = lerp_uv(tri.vertex, tri.edge1, tri.edge2, bary0);
  const UV uv1 = lerp_uv(tri.vertex, tri.edge1, tri.edge2, bary1);
  const UV uv2 = lerp_uv(tri.vertex, tri.edge1, tri.edge2, bary2);

  const float max_v = fmaxf(uv0.v, fmaxf(uv1.v, uv2.v));
  const float min_v = fminf(uv0.v, fminf(uv1.v, uv2.v));

  float m0 = (uv0.u - uv1.u) / (uv0.v - uv1.v);
  float m1 = (uv1.u - uv2.u) / (uv1.v - uv2.v);
  float m2 = (uv2.u - uv0.u) / (uv2.v - uv0.v);

  if (isinf(m0) || isnan(m0)) {
    m0 = 1.0f;
  }

  if (isinf(m1) || isnan(m1)) {
    m1 = 1.0f;
  }

  if (isinf(m2) || isnan(m2)) {
    m2 = 1.0f;
  }

  const float a0 = uv0.u - m0 * uv0.v;
  const float a1 = uv1.u - m1 * uv1.v;
  const float a2 = uv2.u - m2 * uv2.v;

  const float min_e_0 = a0 + fminf(uv0.v * m0, uv1.v * m0);
  const float max_e_0 = a0 + fmaxf(uv0.v * m0, uv1.v * m0);

  const float min_e_1 = a1 + fminf(uv1.v * m1, uv2.v * m1);
  const float max_e_1 = a1 + fmaxf(uv1.v * m1, uv2.v * m1);

  const float min_e_2 = a2 + fminf(uv2.v * m2, uv0.v * m2);
  const float max_e_2 = a2 + fmaxf(uv2.v * m2, uv0.v * m2);

  bool found_opaque      = false;
  bool found_transparent = false;

  for (float v = min_v; v <= max_v; v += tri.tex.inv_height) {
    const float e0    = fmaxf(fminf(a0 + v * m0, max_e_0), min_e_0);
    const float e1    = fmaxf(fminf(a1 + v * m1, max_e_1), min_e_1);
    const float e2    = fmaxf(fminf(a2 + v * m2, max_e_2), min_e_2);
    const float min_u = fminf(e0, fminf(e1, e2));
    const float max_u = fmaxf(e0, fmaxf(e1, e2));

    for (float u = min_u; u <= max_u; u += tri.tex.inv_width) {
      const float4 value = tex2D<float4>(tri.tex.tex, u, 1.0f - v);

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
__global__ void omm_level_0_format_4(uint8_t* dst, uint8_t* level_record) {
  int id                        = threadIdx.x + blockIdx.x * blockDim.x;
  const uint32_t triangle_count = device.scene.triangle_data.triangle_count;

  while (id < triangle_count) {
    OMMTextureTriangle tri = micromap_get_ommtexturetriangle(id);

    const int opacity = micromap_get_opacity(tri, 0, 0);

    const uint8_t v = opacity;

    if (opacity != OPTIX_OPACITY_MICROMAP_STATE_UNKNOWN_OPAQUE)
      level_record[id] = 0;

    dst[id] = v;

    id += blockDim.x * gridDim.x;
  }
}

__global__ void omm_refine_format_4(uint8_t* dst, const uint8_t* src, uint8_t* level_record, const uint32_t src_level) {
  int id                        = threadIdx.x + blockIdx.x * blockDim.x;
  const uint32_t triangle_count = device.scene.triangle_data.triangle_count;

  while (id < triangle_count) {
    if (level_record[id] != 0xFF) {
      id += blockDim.x * gridDim.x;
      continue;
    }

    OMMTextureTriangle tri = micromap_get_ommtexturetriangle(id);

    const uint32_t src_tri_count  = 1 << (2 * src_level);
    const uint32_t src_state_size = (src_tri_count + 3) / 4;
    const uint32_t dst_state_size = src_tri_count;

    const uint8_t* src_tri_ptr = src + id * src_state_size;
    uint8_t* dst_tri_ptr       = dst + id * dst_state_size;

    bool unknowns_left = false;

    for (uint32_t i = 0; i < src_tri_count; i++) {
      uint8_t src_v = src_tri_ptr[i / 4];
      src_v         = (src_v >> (2 * (i & 0b11))) & 0b11;

      uint8_t dst_v = 0;
      if (src_v == OPTIX_OPACITY_MICROMAP_STATE_UNKNOWN_OPAQUE) {
        for (uint32_t j = 0; j < 4; j++) {
          const uint8_t opacity = micromap_get_opacity(tri, src_level + 1, 4 * i + j);

          if (opacity == OPTIX_OPACITY_MICROMAP_STATE_UNKNOWN_OPAQUE)
            unknowns_left = true;

          dst_v = dst_v | (opacity << (2 * j));
        }
      }
      else {
        dst_v = src_v | (src_v << 2) | (src_v << 4) | (src_v << 6);
      }

      dst_tri_ptr[i] = dst_v;
    }

    if (!unknowns_left)
      level_record[id] = src_level + 1;

    id += blockDim.x * gridDim.x;
  }
}

__global__ void omm_gather_array_format_4(
  uint8_t* dst, const uint8_t* src, const uint32_t level, const uint8_t* level_record, const OptixOpacityMicromapDesc* desc) {
  int id                        = threadIdx.x + blockIdx.x * blockDim.x;
  const uint32_t triangle_count = device.scene.triangle_data.triangle_count;
  const uint32_t state_size     = OMM_STATE_SIZE(level, OPTIX_OPACITY_MICROMAP_FORMAT_4_STATE);

  while (id < triangle_count) {
    if (level_record[id] != level) {
      id += blockDim.x * gridDim.x;
      continue;
    }

    for (uint32_t j = 0; j < state_size; j++) {
      dst[desc[id].byteOffset + j] = src[id * state_size + j];
    }

    id += blockDim.x * gridDim.x;
  }
}

static size_t _omm_array_size(const uint32_t count, const uint32_t level, const OptixOpacityMicromapFormat format) {
  if (format != OPTIX_OPACITY_MICROMAP_FORMAT_2_STATE && format != OPTIX_OPACITY_MICROMAP_FORMAT_4_STATE) {
    return 0;
  }

  // OMMs are byte aligned, hence even the low subdivision levels are at least 1 byte in size
  const uint32_t state_size = OMM_STATE_SIZE(level, format);

  return state_size * count;
}

OptixBuildInputOpacityMicromap micromap_opacity_build(RaytraceInstance* instance) {
  const OptixOpacityMicromapFormat format = OPTIX_OPACITY_MICROMAP_FORMAT_4_STATE;
  const uint32_t total_tri_count          = instance->scene.triangle_data.triangle_count;

  // Highest allowed level is 12 according to OptiX Docs
  const uint32_t max_num_levels = 6;
  uint32_t num_levels           = 0;

  uint32_t* triangles_per_level = (uint32_t*) calloc(1, sizeof(uint32_t) * max_num_levels);
  void** data                   = (void**) malloc(sizeof(void*) * max_num_levels);

  // For each triangle, we store the final level, 0xFF specifies that the triangle has not reached its final level yet
  uint8_t* triangle_level = (uint8_t*) malloc(total_tri_count);
  void* triangle_level_buffer;
  device_malloc(&triangle_level_buffer, total_tri_count);

  for (uint32_t i = 0; i < total_tri_count; i++) {
    triangle_level[i] = 0xFF;
  }
  device_upload(triangle_level_buffer, triangle_level, total_tri_count);

  uint32_t remaining_triangles = 0;
  for (; num_levels < max_num_levels;) {
    const size_t data_size = _omm_array_size(total_tri_count, num_levels, format);
    device_malloc(data + num_levels, data_size);
    gpuErrchk(cudaDeviceSynchronize());

    if (num_levels) {
      omm_refine_format_4<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(
        (uint8_t*) data[num_levels], (uint8_t*) data[num_levels - 1], (uint8_t*) triangle_level_buffer, num_levels - 1);
    }
    else {
      omm_level_0_format_4<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>((uint8_t*) data[0], (uint8_t*) triangle_level_buffer);
    }
    gpuErrchk(cudaDeviceSynchronize());

    device_download(triangle_level, triangle_level_buffer, total_tri_count);
    gpuErrchk(cudaDeviceSynchronize());

    remaining_triangles         = 0;
    uint32_t triangles_at_level = 0;

    for (uint32_t i = 0; i < total_tri_count; i++) {
      if (triangle_level[i] == 0xFF)
        remaining_triangles++;

      if (triangle_level[i] == num_levels)
        triangles_at_level++;
    }

    log_message("[OptiX OMM] Remaining triangles after %u iterations: %u", num_levels, remaining_triangles);

    triangles_per_level[num_levels] = triangles_at_level;

    num_levels++;

    if (!remaining_triangles) {
      break;
    }
  }

  // Some triangles needed more refinement but max level was reached.
  if (remaining_triangles) {
    triangles_per_level[num_levels - 1] += remaining_triangles;

    for (uint32_t i = 0; i < total_tri_count; i++) {
      if (triangle_level[i] == 0xFF) {
        triangle_level[i] = num_levels - 1;
      }
    }

    device_upload(triangle_level_buffer, triangle_level, total_tri_count);
  }

  size_t final_array_size        = 0;
  size_t* array_offset_per_level = (size_t*) malloc(sizeof(size_t) * num_levels);
  size_t* array_size_per_level   = (size_t*) malloc(sizeof(size_t) * num_levels);
  for (uint32_t i = 0; i < num_levels; i++) {
    const size_t state_size = OMM_STATE_SIZE(i, format);

    log_message("[OptiX OMM] Total triangles at subdivision level %u: %u", i, triangles_per_level[i]);

    array_size_per_level[i]   = state_size * triangles_per_level[i];
    array_offset_per_level[i] = (i) ? array_offset_per_level[i - 1] + array_size_per_level[i - 1] : 0;

    final_array_size += array_size_per_level[i];
  }

  void* omm_array;
  device_malloc(&omm_array, final_array_size);

  OptixOpacityMicromapDesc* desc = (OptixOpacityMicromapDesc*) malloc(sizeof(OptixOpacityMicromapDesc) * total_tri_count);

  for (uint32_t i = 0; i < total_tri_count; i++) {
    const uint32_t level = (triangle_level[i] == 0xFF) ? max_num_levels - 1 : triangle_level[i];

    desc[i].byteOffset       = array_offset_per_level[level];
    desc[i].subdivisionLevel = level;
    desc[i].format           = format;

    const size_t state_size = OMM_STATE_SIZE(level, format);

    array_offset_per_level[level] += state_size;
  }

  free(array_offset_per_level);
  free(array_size_per_level);

  void* desc_buffer;
  device_malloc(&desc_buffer, sizeof(OptixOpacityMicromapDesc) * total_tri_count);
  device_upload(desc_buffer, desc, sizeof(OptixOpacityMicromapDesc) * total_tri_count);

  for (uint32_t i = 0; i < num_levels; i++) {
    omm_gather_array_format_4<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(
      (uint8_t*) omm_array, (uint8_t*) data[i], i, (uint8_t*) triangle_level_buffer, (OptixOpacityMicromapDesc*) desc_buffer);
  }
  gpuErrchk(cudaDeviceSynchronize());

  for (uint32_t i = 0; i < num_levels; i++) {
    const size_t data_size = _omm_array_size(total_tri_count, i, format);
    device_free(data[i], data_size);
  }

  free(data);

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

  free(triangles_per_level);
  free(triangle_level);
  device_free(triangle_level_buffer, total_tri_count);

  OptixOpacityMicromapArrayBuildInput array_build_input;
  memset(&array_build_input, 0, sizeof(OptixOpacityMicromapArrayBuildInput));

  array_build_input.flags                        = OPTIX_OPACITY_MICROMAP_FLAG_PREFER_FAST_TRACE;
  array_build_input.inputBuffer                  = (CUdeviceptr) omm_array;
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

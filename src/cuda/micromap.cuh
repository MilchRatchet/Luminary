#ifndef CU_MICROMAP_H
#define CU_MICROMAP_H

#include <optix.h>
#include <optix_micromap.h>
#include <optix_stubs.h>

#include "buffer.h"
#include "device.h"
#include "memory.cuh"
#include "utils.cuh"

// OMMs should not occupy too much memory
#define MAX_MEMORY_USAGE 100000000ul

#define OMM_STATE_SIZE(__level__, __format__) \
  (((1u << (__level__ * 2u)) * ((__format__ == OPTIX_OPACITY_MICROMAP_FORMAT_2_STATE) ? 1u : 2u) + 7u) / 8u)

struct OMMTextureTriangle {
  UV vertex;
  UV edge1;
  UV edge2;
  DeviceTexture tex;
  uint16_t tex_id;
} typedef OMMTextureTriangle;

__device__ OMMTextureTriangle micromap_get_ommtexturetriangle(const uint32_t id) {
  const uint32_t material_id = load_triangle_material_id(id);
  const uint16_t tex         = device.scene.materials[material_id].albedo_map;

  OMMTextureTriangle tri;
  tri.tex_id = tex;

  if (tex == TEXTURE_NONE) {
    return tri;
  }

  const float2 data0 = __ldg((float2*) triangle_get_entry_address(4, 2, id));
  const float4 data1 = __ldg((float4*) triangle_get_entry_address(5, 0, id));

  tri.tex = device.ptrs.albedo_atlas[tex];

  tri.vertex = get_uv(data0.x, data0.y);
  tri.edge1  = get_uv(data1.x, data1.y);
  tri.edge2  = get_uv(data1.z, data1.w);

  return tri;
}

// Load triangle only once for the refinement steps
__device__ int micromap_get_opacity(const OMMTextureTriangle tri, const uint32_t level, const uint32_t mt_id) {
  // TODO: Handle texture less transparency of materials.
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
    float max_u       = fmaxf(e0, fmaxf(e1, e2));

    if (max_u > min_u + 1.0f)
      max_u = min_u + 1.0f;

    for (float u = min_u; u <= max_u; u += tri.tex.inv_width) {
      const float alpha = tex2D<float4>(tri.tex.tex, u, 1.0f - v).w;

      if (alpha > 0.0f)
        found_opaque = true;

      if (alpha < 1.0f)
        found_transparent = true;
    }

    if (found_opaque && found_transparent)
      break;
  }

  // Handling IOR means that we cannot just ignore fully transparent geometry.
  // TODO: Allow fast path for when we are in IOR 1.0 media.
#if 0
  if (found_transparent && !found_opaque)
    return OPTIX_OPACITY_MICROMAP_STATE_TRANSPARENT;
#endif

  if (found_opaque && !found_transparent)
    return OPTIX_OPACITY_MICROMAP_STATE_OPAQUE;

  return OPTIX_OPACITY_MICROMAP_STATE_UNKNOWN_OPAQUE;
}

//
// This kernel computes a level 0 format 4 base micromap array.
//
LUMINARY_KERNEL void omm_level_0_format_4(uint8_t* dst, uint8_t* level_record) {
  int id                        = THREAD_ID;
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

LUMINARY_KERNEL void omm_refine_format_4(uint8_t* dst, const uint8_t* src, uint8_t* level_record, const uint32_t src_level) {
  int id                        = THREAD_ID;
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

LUMINARY_KERNEL void omm_gather_array_format_4(
  uint8_t* dst, const uint8_t* src, const uint32_t level, const uint8_t* level_record, const OptixOpacityMicromapDesc* desc) {
  int id                        = THREAD_ID;
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
  const uint8_t max_num_levels = 6;
  uint8_t num_levels           = 0;

  ////////////////////////////////////////////////////////////////////
  // OMM construction
  ////////////////////////////////////////////////////////////////////

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

  size_t memory_usage = 0;

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

    log_message("[OptiX OMM] Remaining triangles after %u iterations: %u.", num_levels, remaining_triangles);

    triangles_per_level[num_levels] = triangles_at_level;

    memory_usage += triangles_at_level * OMM_STATE_SIZE(num_levels, format);

    num_levels++;

    if (!remaining_triangles) {
      break;
    }

    if (memory_usage + remaining_triangles * OMM_STATE_SIZE(num_levels, format) > MAX_MEMORY_USAGE) {
      log_message("[OptiX OMM] Exceeded memory budget at subdivision level %u.", num_levels);
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

    log_message("[OptiX OMM] Total triangles at subdivision level %u: %u.", i, triangles_per_level[i]);

    array_size_per_level[i]   = state_size * triangles_per_level[i];
    array_offset_per_level[i] = (i) ? array_offset_per_level[i - 1] + array_size_per_level[i - 1] : 0;

    final_array_size += array_size_per_level[i];
  }

  void* omm_array;
  device_malloc(&omm_array, final_array_size);

  ////////////////////////////////////////////////////////////////////
  // Description setup
  ////////////////////////////////////////////////////////////////////

  OptixOpacityMicromapDesc* desc = (OptixOpacityMicromapDesc*) malloc(sizeof(OptixOpacityMicromapDesc) * total_tri_count);

  for (uint32_t i = 0; i < total_tri_count; i++) {
    const uint32_t level = (triangle_level[i] == 0xFF) ? max_num_levels - 1 : triangle_level[i];

    desc[i].byteOffset       = (unsigned int) array_offset_per_level[level];
    desc[i].subdivisionLevel = (unsigned short) level;
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

  free(triangle_level);
  device_free(triangle_level_buffer, total_tri_count);

  for (uint32_t i = 0; i < num_levels; i++) {
    const size_t data_size = _omm_array_size(total_tri_count, i, format);
    device_free(data[i], data_size);
  }

  free(data);

  ////////////////////////////////////////////////////////////////////
  // Histogram setup
  ////////////////////////////////////////////////////////////////////

  OptixOpacityMicromapHistogramEntry* histogram =
    (OptixOpacityMicromapHistogramEntry*) malloc(sizeof(OptixOpacityMicromapHistogramEntry) * num_levels);

  for (uint32_t i = 0; i < num_levels; i++) {
    histogram[i].count            = triangles_per_level[i];
    histogram[i].subdivisionLevel = i;
    histogram[i].format           = format;
  }

  ////////////////////////////////////////////////////////////////////
  // Usage count setup
  ////////////////////////////////////////////////////////////////////

  OptixOpacityMicromapUsageCount* usage = (OptixOpacityMicromapUsageCount*) malloc(sizeof(OptixOpacityMicromapUsageCount) * num_levels);
  for (uint32_t i = 0; i < num_levels; i++) {
    usage[i].count            = triangles_per_level[i];
    usage[i].subdivisionLevel = i;
    usage[i].format           = format;
  }

  free(triangles_per_level);

  ////////////////////////////////////////////////////////////////////
  // DMM array construction
  ////////////////////////////////////////////////////////////////////

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
  device_free(omm_array, final_array_size);

  ////////////////////////////////////////////////////////////////////
  // BVH input construction
  ////////////////////////////////////////////////////////////////////

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

#define DMM_NONE 0xFFFFFFFF

struct DMMTextureTriangle {
  UV vertex;
  UV edge1;
  UV edge2;
  DeviceTexture tex;
  uint16_t tex_id;
} typedef DMMTextureTriangle;

__device__ DMMTextureTriangle micromap_get_dmmtexturetriangle(const uint32_t id) {
  const uint32_t material_id = load_triangle_material_id(id);
  const uint16_t tex         = device.scene.materials[material_id].normal_map;

  DMMTextureTriangle tri;
  tri.tex_id = tex;

  if (tex == TEXTURE_NONE) {
    return tri;
  }

  const float2 data0 = __ldg((float2*) triangle_get_entry_address(4, 2, id));
  const float4 data1 = __ldg((float4*) triangle_get_entry_address(5, 0, id));

  tri.tex = device.ptrs.normal_atlas[tex];

  tri.vertex = get_uv(data0.x, data0.y);
  tri.edge1  = get_uv(data1.x, data1.y);
  tri.edge2  = get_uv(data1.z, data1.w);

  return tri;
}

LUMINARY_KERNEL void dmm_precompute_indices(uint32_t* dst) {
  int id                        = THREAD_ID;
  const uint32_t triangle_count = device.scene.triangle_data.triangle_count;

  while (id < triangle_count) {
    const uint32_t material_id = load_triangle_material_id(id);
    const uint16_t tex         = __ldg(&(device.scene.materials[material_id].normal_map));

    dst[id] = (tex == TEXTURE_NONE) ? 0 : 1;

    id += blockDim.x * gridDim.x;
  }
}

LUMINARY_KERNEL void dmm_setup_vertex_directions(half* dst) {
  int id = THREAD_ID;

  while (id < device.scene.triangle_data.triangle_count) {
    const float4 data0 = __ldg((float4*) triangle_get_entry_address(2, 0, id));
    const float4 data1 = __ldg((float4*) triangle_get_entry_address(3, 0, id));
    const float2 data2 = __ldg((float2*) triangle_get_entry_address(4, 0, id));

    const vec3 vertex_normal = get_vector(data0.y, data0.z, data0.w);
    const vec3 edge1_normal  = get_vector(data1.x, data1.y, data1.z);
    const vec3 edge2_normal  = get_vector(data1.w, data2.x, data2.y);

    vec3 v0 = vertex_normal;
    vec3 v1 = add_vector(vertex_normal, edge1_normal);
    vec3 v2 = add_vector(vertex_normal, edge2_normal);

    v0 = scale_vector(v0, 0.1f);
    v1 = scale_vector(v1, 0.1f);
    v2 = scale_vector(v2, 0.1f);

    dst[9 * id + 0] = (half) v0.x;
    dst[9 * id + 1] = (half) v0.y;
    dst[9 * id + 2] = (half) v0.z;
    dst[9 * id + 3] = (half) v1.x;
    dst[9 * id + 4] = (half) v1.y;
    dst[9 * id + 5] = (half) v1.z;
    dst[9 * id + 6] = (half) v2.x;
    dst[9 * id + 7] = (half) v2.y;
    dst[9 * id + 8] = (half) v2.z;

    id += blockDim.x * gridDim.x;
  }
}

__device__ const uint8_t dmm_umajor_id_level_3_format_64[45] = {0,  15, 6, 21, 3, 39, 12, 42, 2,  17, 16, 23, 22, 41, 40,
                                                                44, 43, 8, 18, 7, 24, 14, 36, 13, 20, 19, 26, 25, 38, 37,
                                                                5,  27, 9, 33, 4, 29, 28, 35, 34, 11, 30, 10, 32, 31, 1};

//
// This function is taken from the optixDisplacementMicromesh example code provided in the OptiX 7.7 installation.
//
__device__ void dmm_set_displacement(uint8_t* dst, const uint32_t u_vert_id, const float disp) {
  const uint32_t vert_id = dmm_umajor_id_level_3_format_64[u_vert_id];

  const uint16_t v = (uint16_t) (disp * ((uint16_t) 0x7FF));

  uint32_t bit_offset = 11 * vert_id;
  uint32_t v_offset   = 0;

  while (v_offset < 11) {
    uint32_t num = (~bit_offset & 7) + 1;

    if (11 - v_offset < num) {
      num = 11 - v_offset;
    }

    const uint16_t mask  = (1u << num) - 1u;
    const uint32_t id    = bit_offset >> 3;
    const uint32_t shift = bit_offset & 7;

    const uint8_t bits = (uint8_t) ((v >> v_offset) & mask);
    dst[id] &= ~(mask << shift);
    dst[id] |= bits << shift;

    bit_offset += num;
    v_offset += num;
  }
}

LUMINARY_KERNEL void dmm_build_level_3_format_64(uint8_t* dst, const uint32_t* mapping, const uint32_t count) {
  int id = THREAD_ID;

  while (id < count) {
    const uint32_t tri_id = mapping[id];

    if (tri_id == DMM_NONE) {
      uint4* ptr    = (uint4*) (dst + 64 * id);
      const uint4 v = make_uint4(0, 0, 0, 0);

      __stcs(ptr + 0, v);
      __stcs(ptr + 1, v);
      __stcs(ptr + 2, v);
      __stcs(ptr + 3, v);
    }
    else {
      const DMMTextureTriangle tri = micromap_get_dmmtexturetriangle(tri_id);

      float2 bary0;
      float2 bary1;
      float2 bary2;
      optixMicromapIndexToBaseBarycentrics(0, 0, bary0, bary1, bary2);

      const UV uv0 = lerp_uv(tri.vertex, tri.edge1, tri.edge2, bary0);
      const UV uv1 = lerp_uv(tri.vertex, tri.edge1, tri.edge2, bary1);
      const UV uv2 = lerp_uv(tri.vertex, tri.edge1, tri.edge2, bary2);

      uint32_t u_vert_id          = 0;
      const uint32_t num_segments = 1 << 3;

      for (uint32_t i = 0; i < num_segments + 1; i++) {
        for (uint32_t j = 0; j < num_segments + 1 - i; j++) {
          const float2 micro_vertex_bary = make_float2(((float) i) / num_segments, ((float) j) / num_segments);

          UV uv;
          uv.u = (1.0f - micro_vertex_bary.x - micro_vertex_bary.y) * uv0.u + micro_vertex_bary.x * uv1.u + micro_vertex_bary.y * uv2.u;
          uv.v = (1.0f - micro_vertex_bary.x - micro_vertex_bary.y) * uv0.v + micro_vertex_bary.x * uv1.v + micro_vertex_bary.y * uv2.v;

          const float disp = tex2D<float4>(tri.tex.tex, uv.u, 1.0f - uv.v).w;

          dmm_set_displacement(dst + 64 * id, u_vert_id, disp);
          u_vert_id++;
        }
      }
    }

    id += blockDim.x * gridDim.x;
  }
}

OptixBuildInputDisplacementMicromap micromap_displacement_build(RaytraceInstance* instance) {
  // Initial implementation only supports the basic one uncompressed block as a DMM
  // It is planned to at least support uncompressed blocks that sum to 1024 microtriangles in the future
  // level  = 3
  // format = OPTIX_DISPLACEMENT_MICROMAP_FORMAT_64_MICRO_TRIS_64_BYTES

  const uint32_t total_tri_count = instance->scene.triangle_data.triangle_count;

  ////////////////////////////////////////////////////////////////////
  // Index computation
  ////////////////////////////////////////////////////////////////////

  void* indices_buffer;
  device_malloc(&indices_buffer, total_tri_count * sizeof(uint32_t));

  dmm_precompute_indices<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>((uint32_t*) indices_buffer);
  gpuErrchk(cudaDeviceSynchronize());

  uint32_t* indices = (uint32_t*) malloc(sizeof(uint32_t) * total_tri_count);
  uint32_t* mapping = (uint32_t*) malloc(sizeof(uint32_t) * total_tri_count);

  device_download(indices, indices_buffer, total_tri_count * sizeof(uint32_t));
  gpuErrchk(cudaDeviceSynchronize());

  mapping[0] = DMM_NONE;

  uint32_t dmm_count = 1;
  for (uint32_t i = 0; i < total_tri_count; i++) {
    if (indices[i]) {
      indices[i]         = dmm_count;
      mapping[dmm_count] = i;
      dmm_count++;
    }
  }

  if (dmm_count == 1) {
    log_message("[Optix DMM] No displacement maps exist. No DMM was built.");

    OptixBuildInputDisplacementMicromap empty_result;
    memset(&empty_result, 0, sizeof(OptixBuildInputDisplacementMicromap));

    free(indices);
    free(mapping);
    device_free(indices_buffer, total_tri_count * sizeof(uint32_t));

    return empty_result;
  }

  device_upload(indices_buffer, indices, total_tri_count * sizeof(uint32_t));
  gpuErrchk(cudaDeviceSynchronize());
  free(indices);

  void* mapping_buffer;
  device_malloc(&mapping_buffer, dmm_count * sizeof(uint32_t));
  device_upload(mapping_buffer, mapping, dmm_count * sizeof(uint32_t));
  gpuErrchk(cudaDeviceSynchronize());
  free(mapping);

  ////////////////////////////////////////////////////////////////////
  // Vertex direction computation
  ////////////////////////////////////////////////////////////////////

  void* vertex_direction_buffer;
  device_malloc(&vertex_direction_buffer, total_tri_count * sizeof(half) * 9);

  dmm_setup_vertex_directions<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>((half*) vertex_direction_buffer);
  gpuErrchk(cudaDeviceSynchronize());

  ////////////////////////////////////////////////////////////////////
  // Usage count setup
  ////////////////////////////////////////////////////////////////////

  OptixDisplacementMicromapUsageCount* usage =
    (OptixDisplacementMicromapUsageCount*) malloc(sizeof(OptixDisplacementMicromapUsageCount) * 2);

  usage[0].count            = total_tri_count - (dmm_count - 1);
  usage[0].format           = OPTIX_DISPLACEMENT_MICROMAP_FORMAT_64_MICRO_TRIS_64_BYTES;
  usage[0].subdivisionLevel = 0;

  usage[1].count            = dmm_count - 1;
  usage[1].format           = OPTIX_DISPLACEMENT_MICROMAP_FORMAT_64_MICRO_TRIS_64_BYTES;
  usage[1].subdivisionLevel = 3;

  ////////////////////////////////////////////////////////////////////
  // Description setup
  ////////////////////////////////////////////////////////////////////

  OptixDisplacementMicromapDesc* desc = (OptixDisplacementMicromapDesc*) malloc(sizeof(OptixDisplacementMicromapDesc) * dmm_count);
  for (uint32_t i = 0; i < dmm_count; i++) {
    desc[i].byteOffset       = i * 64;
    desc[i].format           = OPTIX_DISPLACEMENT_MICROMAP_FORMAT_64_MICRO_TRIS_64_BYTES;
    desc[i].subdivisionLevel = (i) ? 3 : 0;
  }

  void* desc_buffer;
  device_malloc(&desc_buffer, sizeof(OptixDisplacementMicromapDesc) * dmm_count);
  device_upload(desc_buffer, desc, sizeof(OptixDisplacementMicromapDesc) * dmm_count);
  gpuErrchk(cudaDeviceSynchronize());

  free(desc);

  ////////////////////////////////////////////////////////////////////
  // Histogram setup
  ////////////////////////////////////////////////////////////////////

  OptixDisplacementMicromapHistogramEntry* histogram =
    (OptixDisplacementMicromapHistogramEntry*) malloc(sizeof(OptixDisplacementMicromapHistogramEntry) * 2);

  histogram[0].count            = 1;
  histogram[0].format           = OPTIX_DISPLACEMENT_MICROMAP_FORMAT_64_MICRO_TRIS_64_BYTES;
  histogram[0].subdivisionLevel = 0;

  histogram[1].count            = dmm_count - 1;
  histogram[1].format           = OPTIX_DISPLACEMENT_MICROMAP_FORMAT_64_MICRO_TRIS_64_BYTES;
  histogram[1].subdivisionLevel = 3;

  ////////////////////////////////////////////////////////////////////
  // DMM construction
  ////////////////////////////////////////////////////////////////////

  const size_t dmm_data_size = 64 * dmm_count;

  void* data;
  device_malloc(&data, dmm_data_size);

  dmm_build_level_3_format_64<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>((uint8_t*) data, (uint32_t*) mapping_buffer, dmm_count);
  gpuErrchk(cudaDeviceSynchronize());

  device_free(mapping_buffer, dmm_count * sizeof(uint32_t));

  ////////////////////////////////////////////////////////////////////
  // DMM array construction
  ////////////////////////////////////////////////////////////////////

  OptixDisplacementMicromapArrayBuildInput array_build_input;
  memset(&array_build_input, 0, sizeof(OptixDisplacementMicromapArrayBuildInput));

  array_build_input.flags = OPTIX_DISPLACEMENT_MICROMAP_FLAG_PREFER_FAST_TRACE;

  array_build_input.displacementMicromapHistogramEntries    = histogram;
  array_build_input.numDisplacementMicromapHistogramEntries = 2;

  array_build_input.perDisplacementMicromapDescBuffer        = (CUdeviceptr) desc_buffer;
  array_build_input.perDisplacementMicromapDescStrideInBytes = 0;

  array_build_input.displacementValuesBuffer = (CUdeviceptr) data;

  OptixMicromapBufferSizes buffer_sizes;
  memset(&buffer_sizes, 0, sizeof(OptixMicromapBufferSizes));

  OPTIX_CHECK(optixDisplacementMicromapArrayComputeMemoryUsage(instance->optix_ctx, &array_build_input, &buffer_sizes));
  gpuErrchk(cudaDeviceSynchronize());

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

  OPTIX_CHECK(optixDisplacementMicromapArrayBuild(instance->optix_ctx, 0, &array_build_input, &buffers));
  gpuErrchk(cudaDeviceSynchronize());

  device_free(temp_buffer, buffer_sizes.tempSizeInBytes);
  device_free(data, dmm_data_size);
  free(histogram);

  ////////////////////////////////////////////////////////////////////
  // BVH input construction
  ////////////////////////////////////////////////////////////////////

  OptixBuildInputDisplacementMicromap bvh_input;
  memset(&bvh_input, 0, sizeof(OptixBuildInputDisplacementMicromap));

  bvh_input.indexingMode = OPTIX_DISPLACEMENT_MICROMAP_ARRAY_INDEXING_MODE_INDEXED;

  bvh_input.displacementMicromapUsageCounts    = usage;
  bvh_input.numDisplacementMicromapUsageCounts = 2;

  bvh_input.displacementMicromapArray = (CUdeviceptr) output_buffer;

  bvh_input.displacementMicromapIndexBuffer        = (CUdeviceptr) indices_buffer;
  bvh_input.displacementMicromapIndexSizeInBytes   = 4;
  bvh_input.displacementMicromapIndexStrideInBytes = 0;
  bvh_input.displacementMicromapIndexOffset        = 0;

  bvh_input.vertexDirectionsBuffer       = (CUdeviceptr) vertex_direction_buffer;
  bvh_input.vertexDirectionFormat        = OPTIX_DISPLACEMENT_MICROMAP_DIRECTION_FORMAT_HALF3;
  bvh_input.vertexDirectionStrideInBytes = 0;

  return bvh_input;
}

#endif /* CU_MICROMAP_H */

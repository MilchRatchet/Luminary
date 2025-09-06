#ifndef CU_LUMINARY_MICROMAP_H
#define CU_LUMINARY_MICROMAP_H

#include <optix_micromap.h>

#include "math.cuh"
#include "memory.cuh"
#include "utils.cuh"

// OMMs should not occupy too much memory
#define MAX_MEMORY_USAGE 100000000ul

#define OMM_STATE_SIZE(__level__, __format__) \
  (((1u << (__level__ * 2u)) * ((__format__ == OPTIX_OPACITY_MICROMAP_FORMAT_2_STATE) ? 1u : 2u) + 7u) / 8u)

struct OMMTextureTriangle {
  UV vertex;
  UV vertex1;
  UV vertex2;
  DeviceTextureObject tex;
  uint16_t tex_id;
  bool is_invisible;
} typedef OMMTextureTriangle;

__device__ OMMTextureTriangle micromap_get_ommtexturetriangle(const uint32_t mesh_id, const uint32_t tri_id) {
  const uint32_t material_id = material_id_load(mesh_id, tri_id);
  const uint16_t tex         = __ldg(&device.ptrs.materials[material_id].albedo_tex);

  OMMTextureTriangle tri;
  tri.tex_id       = tex;
  tri.is_invisible = false;

  if (tex == TEXTURE_NONE) {
    tri.is_invisible = __ldg(&device.ptrs.materials[material_id].albedo_a) == 0;
    return tri;
  }

  const DeviceTriangle* tri_ptr = (const DeviceTriangle*) __ldg((uint64_t*) (device.ptrs.triangles + mesh_id));
  const uint32_t triangle_count = __ldg(device.ptrs.triangle_counts + mesh_id);

  const float4 t2 = __ldg((float4*) triangle_get_entry_address(tri_ptr, 2, 0, tri_id, triangle_count));

  tri.tex = device.ptrs.textures[tex];

  tri.vertex  = uv_unpack(__float_as_uint(t2.y));
  tri.vertex1 = uv_unpack(__float_as_uint(t2.z));
  tri.vertex2 = uv_unpack(__float_as_uint(t2.w));

  return tri;
}

// Load triangle only once for the refinement steps
__device__ uint8_t micromap_get_opacity(const OMMTextureTriangle tri, const uint32_t level, const uint32_t mt_id) {
  if (tri.tex_id == TEXTURE_NONE) {
    // Materials without an albedo texture can have their opacity changed, so we cannot use OMMs there.
    return OPTIX_OPACITY_MICROMAP_STATE_UNKNOWN_OPAQUE;
  }

  if (texture_is_valid(tri.tex) == false) {
    // Materials with an invalid texture are finicky, they rely on hardcoded behaviour, don't fast path such edge cases.
    return OPTIX_OPACITY_MICROMAP_STATE_UNKNOWN_OPAQUE;
  }

  float2 bary0;
  float2 bary1;
  float2 bary2;
  optixMicromapIndexToBaseBarycentrics(mt_id, level, bary0, bary1, bary2);

  const UV uv0 = lerp_uv(tri.vertex, tri.vertex1, tri.vertex2, bary0);
  const UV uv1 = lerp_uv(tri.vertex, tri.vertex1, tri.vertex2, bary1);
  const UV uv2 = lerp_uv(tri.vertex, tri.vertex1, tri.vertex2, bary2);

  const float max_v = fmaxf(uv0.v, fmaxf(uv1.v, uv2.v));
  const float min_v = fminf(uv0.v, fminf(uv1.v, uv2.v));

  const float span_v   = max_v - min_v;
  const float texels_v = span_v * tri.tex.height;

  if (texels_v <= 0.0f)
    return OPTIX_OPACITY_MICROMAP_STATE_UNKNOWN_OPAQUE;

  const float mip_level_v = fmaxf(log2f(texels_v), 0.0f);

  float m0 = (uv0.u - uv1.u) / (uv0.v - uv1.v);
  float m1 = (uv1.u - uv2.u) / (uv1.v - uv2.v);
  float m2 = (uv2.u - uv0.u) / (uv2.v - uv0.v);

  if (is_non_finite(m0)) {
    m0 = 1.0f;
  }

  if (is_non_finite(m1)) {
    m1 = 1.0f;
  }

  if (is_non_finite(m2)) {
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

  const float inv_width  = 1.0f / tri.tex.width;
  const float inv_height = 1.0f / tri.tex.height;

  float step_v = inv_height;

  for (float v = min_v; v <= max_v;) {
    v = fminf(v, max_v);

    const float e0    = fmaxf(fminf(a0 + v * m0, max_e_0), min_e_0);
    const float e1    = fmaxf(fminf(a1 + v * m1, max_e_1), min_e_1);
    const float e2    = fmaxf(fminf(a2 + v * m2, max_e_2), min_e_2);
    const float min_u = fminf(e0, fminf(e1, e2));
    float max_u       = fmaxf(e0, fmaxf(e1, e2));

    // There is no reason to iterate over the whole texture more than once.
    if (max_u > min_u + 1.0f)
      max_u = min_u + 1.0f;

    const float span_u   = max_u - min_u;
    const float texels_u = span_u * tri.tex.width;

    if (texels_u <= 0.0f) {
      v += step_v;
      continue;
    }

    const float mip_level_u = fmaxf(log2f(texels_u), 0.0f);

    // Taking the minimum would be accurate but has issues with thin triangles using tiling and high resolution textures.
    // Using the max introduces some error but is robust in terms of generation time.
    const float mip_level = fmaxf(fmaxf(mip_level_u, mip_level_v) - 1.0f, 0.0f);

    TextureLoadArgs tex_load_args = texture_get_default_args();
    tex_load_args.mip_level       = mip_level;

    const float step_u = exp2f(mip_level) * inv_width;

    for (float u = min_u; u <= max_u; u += step_u) {
      u = fminf(u, max_u);

      const float alpha = texture_load(tri.tex, get_uv(u, v), tex_load_args).w;

      if (alpha > 0.0f)
        found_opaque = true;

      if (alpha < 1.0f)
        found_transparent = true;
    }

    if (found_opaque && found_transparent)
      break;

    step_v = exp2f(mip_level) * inv_height;
    v += step_v;
  }

  if (found_transparent && !found_opaque)
    return OPTIX_OPACITY_MICROMAP_STATE_TRANSPARENT;

  if (found_opaque && !found_transparent)
    return OPTIX_OPACITY_MICROMAP_STATE_OPAQUE;

  // This is a case that should never happen
  if (!found_opaque && !found_transparent)
    return OPTIX_OPACITY_MICROMAP_STATE_UNKNOWN_OPAQUE;

  return OPTIX_OPACITY_MICROMAP_STATE_UNKNOWN_TRANSPARENT;
}

//
// This kernel computes a level 0 format 4 base micromap array.
//
LUMINARY_KERNEL void omm_level_0_format_4(const KernelArgsOMMLevel0Format4 args) {
  uint32_t tri_id = THREAD_ID;

  while (tri_id < args.triangle_count) {
    const OMMTextureTriangle tri = micromap_get_ommtexturetriangle(args.mesh_id, tri_id);

    const uint8_t opacity = micromap_get_opacity(tri, 0, 0);

    if (opacity != OPTIX_OPACITY_MICROMAP_STATE_UNKNOWN_TRANSPARENT)
      args.level_record[tri_id] = 0;

    args.dst[tri_id] = opacity;

    tri_id += blockDim.x * gridDim.x;
  }
}

LUMINARY_KERNEL void omm_refine_format_4(const KernelArgsOMMRefineFormat4 args) {
  uint32_t tri_id = THREAD_ID;

  while (tri_id < args.triangle_count) {
    if (args.level_record[tri_id] != 0xFF) {
      tri_id += blockDim.x * gridDim.x;
      continue;
    }

    const OMMTextureTriangle tri = micromap_get_ommtexturetriangle(args.mesh_id, tri_id);

    const uint32_t src_tri_count  = 1 << (2 * args.src_level);
    const uint32_t src_state_size = (src_tri_count + 3) / 4;
    const uint32_t dst_state_size = src_tri_count;

    const uint8_t* src_tri_ptr = args.src + tri_id * src_state_size;
    uint8_t* dst_tri_ptr       = args.dst + tri_id * dst_state_size;

    bool unknowns_left = false;

    for (uint32_t i = 0; i < src_tri_count; i++) {
      uint8_t src_v = src_tri_ptr[i / 4];
      src_v         = (src_v >> (2 * (i & 0b11))) & 0b11;

      uint8_t dst_v = 0;
      if (src_v == OPTIX_OPACITY_MICROMAP_STATE_UNKNOWN_TRANSPARENT) {
        for (uint32_t j = 0; j < 4; j++) {
          const uint8_t opacity = micromap_get_opacity(tri, args.src_level + 1, 4 * i + j);

          if (opacity == OPTIX_OPACITY_MICROMAP_STATE_UNKNOWN_TRANSPARENT)
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
      args.level_record[tri_id] = args.src_level + 1;

    tri_id += blockDim.x * gridDim.x;
  }
}

LUMINARY_KERNEL void omm_gather_array_format_4(const KernelArgsOMMGatherArrayFormat4 args) {
  uint32_t tri_id           = THREAD_ID;
  const uint32_t state_size = OMM_STATE_SIZE(args.level, OPTIX_OPACITY_MICROMAP_FORMAT_4_STATE);

  while (tri_id < args.triangle_count) {
    if (args.level_record[tri_id] != args.level) {
      tri_id += blockDim.x * gridDim.x;
      continue;
    }

    for (uint32_t j = 0; j < state_size; j++) {
      args.dst[args.desc[tri_id].byteOffset + j] = args.src[tri_id * state_size + j];
    }

    tri_id += blockDim.x * gridDim.x;
  }
}

#endif /* CU_LUMINARY_MICROMAP_H */

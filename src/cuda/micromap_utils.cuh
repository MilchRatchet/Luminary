#ifndef CU_MICROMAP_UTILS_H
#define CU_MICROMAP_UTILS_H

#include <optix.h>
#include <optix_micromap.h>
#include <optix_stubs.h>

#include "math.cuh"
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

LUM_DEVICE_FUNC OMMTextureTriangle micromap_get_ommtexturetriangle(const uint32_t id) {
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

  tri.vertex = get_UV(data0.x, data0.y);
  tri.edge1  = get_UV(data1.x, data1.y);
  tri.edge2  = get_UV(data1.z, data1.w);

  return tri;
}

// Load triangle only once for the refinement steps
LUM_DEVICE_FUNC int micromap_get_opacity(const OMMTextureTriangle tri, const uint32_t level, const uint32_t mt_id) {
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

  if (found_transparent && !found_opaque)
    return OPTIX_OPACITY_MICROMAP_STATE_TRANSPARENT;

  if (found_opaque && !found_transparent)
    return OPTIX_OPACITY_MICROMAP_STATE_OPAQUE;

  return OPTIX_OPACITY_MICROMAP_STATE_UNKNOWN_OPAQUE;
}

#define DMM_NONE 0xFFFFFFFF

struct DMMTextureTriangle {
  UV vertex;
  UV edge1;
  UV edge2;
  DeviceTexture tex;
  uint16_t tex_id;
} typedef DMMTextureTriangle;

LUM_DEVICE_FUNC DMMTextureTriangle micromap_get_dmmtexturetriangle(const uint32_t id) {
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

  tri.vertex = get_UV(data0.x, data0.y);
  tri.edge1  = get_UV(data1.x, data1.y);
  tri.edge2  = get_UV(data1.z, data1.w);

  return tri;
}

#endif /* CU_MICROMAP_UTILS_H */

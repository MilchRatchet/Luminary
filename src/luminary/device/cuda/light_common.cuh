#ifndef CU_LUMINARY_LIGHT_COMMON_H
#define CU_LUMINARY_LIGHT_COMMON_H

#include "material.cuh"
#include "utils.cuh"

template <MaterialType TYPE>
struct LightSampleResult {
  TriangleHandle handle;
  vec3 ray;
  RGBF light_color;
  float dist;
  bool is_refraction;
};

template <>
struct LightSampleResult<MATERIAL_VOLUME> {
  TriangleHandle handle;
  RGBF light_color;
  uint32_t seed;
  Quaternion rotation;
  float scale;
};

/*
 * This struct is computed on the device when needed, it is not stored in memory.
 * The expectation is that the vertex positions are already transformed and the UV
 * is already transformed using the triangle transforms.
 */
struct TriangleLight {
  vec3 vertex;
  vec3 edge1;
  vec3 edge2;
  UV tex_coords;
  uint16_t material_id;
} typedef TriangleLight;

#endif /* CU_LUMINARY_LIGHT_COMMON_H */

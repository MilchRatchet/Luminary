#ifndef CU_LUMINARY_LIGHT_COMMON_H
#define CU_LUMINARY_LIGHT_COMMON_H

#include "material.cuh"
#include "utils.cuh"

////////////////////////////////////////////////////////////////////
// Utils
////////////////////////////////////////////////////////////////////

LUMINARY_FUNCTION bool light_allow_geo_direct_lighting(const DeviceTask task) {
  return ((task.state & STATE_FLAG_VOLUME_SCATTERED) == 0);
}

////////////////////////////////////////////////////////////////////
// Bridges
////////////////////////////////////////////////////////////////////

// This must correspond to the G term used when computing the LUT.
#define BRIDGES_HG_G_TERM (0.85f)
#define BRIDGES_INITIAL_VERTEX_FORWARD_PROB (0.95f)
#define BRIDGES_MAX_VERTEX_COUNT 15

LUMINARY_FUNCTION vec3 bridges_phase_sample(const vec3 ray, const float2 r_dir) {
  const float cos_angle = henyey_greenstein_phase_sample(BRIDGES_HG_G_TERM, r_dir.x);

  return phase_sample_basis(cos_angle, r_dir.y, ray);
}

LUMINARY_FUNCTION float bridges_phase_function(const float cos_angle) {
  return henyey_greenstein_phase_function(cos_angle, BRIDGES_HG_G_TERM);
}

template <MaterialType TYPE>
struct LightSampleResult {
  uint32_t light_id;
  vec3 ray;
  RGBF light_color;
  float dist;
};

template <>
struct LightSampleResult<MATERIAL_GEOMETRY> {
  uint32_t light_id;
  vec3 ray;
  RGBF light_color;
  float dist;
  float light_tree_root_sum;
};

template <>
struct LightSampleResult<MATERIAL_VOLUME> {
  uint32_t light_id;
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
  bool bidirectional;
} typedef TriangleLight;

#endif /* CU_LUMINARY_LIGHT_COMMON_H */

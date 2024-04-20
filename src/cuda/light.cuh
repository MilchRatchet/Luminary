#ifndef CU_LIGHT_H
#define CU_LIGHT_H

#include "memory.cuh"
#include "texture_utils.cuh"
#include "utils.cuh"

/*
 * Surface sample a triangle and return its area and emission luminance.
 * @param triangle Triangle.
 * @param origin Point to sample from.
 * @param area Solid angle of the triangle.
 * @param seed Random seed used to sample the triangle.
 * @param lum Output emission luminance of the triangle at the sampled point.
 * @result Normalized direction to the point on the triangle.
 *
 * Robust triangle sampling.
 */
__device__ vec3 light_sample_triangle(
  const TriangleLight triangle, const GBufferData data, const ushort2 pixel, const uint32_t light_ray_index, float& pdf, float& dist,
  RGBF& color) {
  const float2 random = quasirandom_sequence_2D(QUASI_RANDOM_TARGET_TBD_1 + light_ray_index, pixel);

  float r1 = sqrtf(random.x);
  float r2 = random.y;

  // Map random numbers uniformly into [0.025,0.975].
  r1 = 0.025f + 0.95f * r1;
  r2 = 0.025f + 0.95f * r2;

  const float u = 1.0f - r1;
  const float v = r1 * r2;

  const vec3 p   = add_vector(triangle.vertex, add_vector(scale_vector(triangle.edge1, u), scale_vector(triangle.edge2, v)));
  const vec3 dir = vector_direction_stable(p, data.position);

  const vec3 face_normal = cross_product(triangle.edge1, triangle.edge2);

  if (device.scene.material.light_side_mode != LIGHT_SIDE_MODE_BOTH) {
    const float side = (device.scene.material.light_side_mode == LIGHT_SIDE_MODE_ONE_CW) ? 1.0f : -1.0f;

    if (dot_product(face_normal, dir) * side > 0.0f) {
      // Reject side with no emission
      pdf = 0.0f;
      return get_vector(0.0f, 0.0f, 0.0f);
    }
  }

  dist = get_length(sub_vector(p, data.position));

  // Use that surface * cos_term = 0.5 * |a x b| * |normal(a x b)^Td| = 0.5 * |a x b| * |(a x b)^Td|/|a x b| = 0.5 * |(a x b)^Td|.
  const float surface_cos_term = 0.5f * fabsf(dot_product(face_normal, dir));

  float solid_angle = surface_cos_term / (dist * dist);

  if (isnan(solid_angle) || isinf(solid_angle) || solid_angle < 1e-7f) {
    pdf = 0.0f;
    return get_vector(0.0f, 0.0f, 0.0f);
  }

  pdf = 1.0f / solid_angle;

  const uint16_t illum_tex = device.scene.materials[triangle.material_id].luminance_map;

  // TODO: Add support for constant colors
  color = get_color(0.0f, 0.0f, 0.0f);

  if (illum_tex != TEXTURE_NONE) {
    const UV tex_coords   = load_triangle_tex_coords(triangle.triangle_id, make_float2(u, v));
    const float4 emission = texture_load(device.ptrs.luminance_atlas[illum_tex], tex_coords);

    color = scale_color(get_color(emission.x, emission.y, emission.z), device.scene.material.default_material.b * emission.w);
  }

  return dir;
}

#endif /* CU_LIGHT_H */

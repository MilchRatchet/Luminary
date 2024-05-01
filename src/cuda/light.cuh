#ifndef CU_LIGHT_H
#define CU_LIGHT_H

#if defined(SHADING_KERNEL)

#include "memory.cuh"
#include "sky_utils.cuh"
#include "texture_utils.cuh"
#include "utils.cuh"

enum LightRayTarget { LIGHT_RAY_TARGET_SUN = 0, LIGHT_RAY_TARGET_TOY = 1, LIGHT_RAY_TARGET_GEOMETRY = 2 } typedef LightRayTarget;

__device__ float light_triangle_intersection_uv(const TriangleLight triangle, const vec3 origin, const vec3 ray, float2& coords) {
  const vec3 h  = cross_product(ray, triangle.edge2);
  const float a = dot_product(triangle.edge1, h);

  const float f = 1.0f / a;
  const vec3 s  = sub_vector(origin, triangle.vertex);
  const float u = f * dot_product(s, h);

  const vec3 q  = cross_product(s, triangle.edge1);
  const float v = f * dot_product(ray, q);

  coords = make_float2(u, v);

  //  The third check is inverted to catch NaNs since NaNs always return false, the not will turn it into a true
  if (v < 0.0f || u < 0.0f || !(u + v <= 1.0f))
    return FLT_MAX;

  const float t = f * dot_product(triangle.edge2, q);

  return __fslctf(t, FLT_MAX, t);
}

/*
 * Solid angle sample a triangle.
 * @param triangle Triangle.
 * @param data Data about shading point.
 * @param random Random numbers.
 * @param pdf PDF of sampled direction.
 * @param dist Distance to sampled point on triangle.
 * @param color Emission from triangle at sampled point.
 * @result Normalized direction to the point on the triangle.
 *
 * Robust solid angle sampling method from
 * C. Peters, "BRDF Importance Sampling for Linear Lights", Computer Graphics Forum (Proc. HPG) 40, 8, 2021.
 *
 */
__device__ vec3 light_sample_triangle(
  const TriangleLight triangle, const GBufferData data, const float2 random, float& solid_angle, float& dist, RGBF& color) {
  const vec3 v0 = normalize_vector(sub_vector(triangle.vertex, data.position));
  const vec3 v1 = normalize_vector(sub_vector(add_vector(triangle.vertex, triangle.edge1), data.position));
  const vec3 v2 = normalize_vector(sub_vector(add_vector(triangle.vertex, triangle.edge2), data.position));

  const float G0 = fabsf(dot_product(cross_product(v0, v1), v2));
  const float G1 = dot_product(v0, v2) + dot_product(v1, v2);
  const float G2 = 1.0f + dot_product(v0, v1);

  solid_angle = 2.0f * atan2f(G0, G1 + G2);

  if (isnan(solid_angle) || isinf(solid_angle) || solid_angle < 1e-7f) {
    solid_angle = 0.0f;
    return get_vector(0.0f, 0.0f, 0.0f);
  }

  const float sampled_solid_angle = random.x * solid_angle;

  const vec3 r = add_vector(
    scale_vector(v0, G0 * cosf(0.5f * sampled_solid_angle) - G1 * sinf(0.5f * sampled_solid_angle)),
    scale_vector(v2, G2 * sinf(0.5f * sampled_solid_angle)));

  const vec3 v2_t = sub_vector(scale_vector(r, 2.0f * dot_product(v0, r) / dot_product(r, r)), v0);

  const float s2 = dot_product(v1, v2_t);
  const float s  = (1.0f - random.y) + random.y * s2;
  const float t  = sqrtf(fmaxf((1.0f - s * s) / (1.0f - s2 * s2), 0.0f));

  const vec3 dir = add_vector(scale_vector(v1, s - t * s2), scale_vector(v2_t, t));

  if (device.scene.material.light_side_mode != LIGHT_SIDE_MODE_BOTH) {
    const vec3 face_normal = cross_product(triangle.edge1, triangle.edge2);
    const float side       = (device.scene.material.light_side_mode == LIGHT_SIDE_MODE_ONE_CW) ? 1.0f : -1.0f;

    if (dot_product(face_normal, dir) * side > 0.0f) {
      // Reject side with no emission
      solid_angle = 0.0f;
      return get_vector(0.0f, 0.0f, 0.0f);
    }
  }

  float2 coords;
  dist = light_triangle_intersection_uv(triangle, data.position, dir, coords);

  const uint16_t illum_tex = device.scene.materials[triangle.material_id].luminance_map;

  // TODO: Add support for constant colors
  color = get_color(0.0f, 0.0f, 0.0f);

  if (illum_tex != TEXTURE_NONE) {
    const UV tex_coords   = load_triangle_tex_coords(triangle.triangle_id, coords);
    const float4 emission = texture_load(device.ptrs.luminance_atlas[illum_tex], tex_coords);

    color = scale_color(get_color(emission.x, emission.y, emission.z), device.scene.material.default_material.b * emission.w);
  }

  return dir;
}

__device__ float light_sample_solid_angle(const LightRayTarget light_target, const uint32_t light_id, const GBufferData data) {
  switch (light_target) {
    case LIGHT_RAY_TARGET_SUN: {
      vec3 sky_pos = world_to_sky_transform(data.position);
      return sample_sphere_solid_angle(device.sun_pos, SKY_SUN_RADIUS, sky_pos);
    }
    case LIGHT_RAY_TARGET_TOY:
      return toy_get_solid_angle(data.position);
    case LIGHT_RAY_TARGET_GEOMETRY:
    default: {
      const TriangleLight light = load_triangle_light(device.scene.triangle_lights, light_id);
      return sample_triangle_solid_angle(light, data.position);
    }
  }
}

__device__ vec3 light_sample(
  const LightRayTarget light_target, const uint32_t light_id, const GBufferData data, const ushort2 pixel, const uint32_t light_ray_index,
  float& solid_angle, float& dist, RGBF& color) {
  const float2 random = quasirandom_sequence_2D(QUASI_RANDOM_TARGET_TBD_1 + light_ray_index, pixel);

  switch (light_target) {
    case LIGHT_RAY_TARGET_SUN: {
      vec3 sky_pos = world_to_sky_transform(data.position);
      vec3 ray     = sample_sphere(device.sun_pos, SKY_SUN_RADIUS, sky_pos, random, solid_angle);
      color        = sky_get_sun_color(sky_pos, ray);
      dist         = FLT_MAX;
      return ray;
    }
    case LIGHT_RAY_TARGET_TOY: {
      solid_angle = toy_get_solid_angle(data.position);
      color       = scale_color(device.scene.toy.emission, device.scene.toy.material.b);
      // Approximation, it is not super important what the actual distance is
      dist = get_length(sub_vector(data.position, device.scene.toy.position));

      return toy_sample_ray(data.position, random);
    }
    case LIGHT_RAY_TARGET_GEOMETRY:
    default: {
      const TriangleLight light = load_triangle_light(device.scene.triangle_lights, light_id);
      return light_sample_triangle(light, data, random, solid_angle, dist, color);
    }
  }
}

#endif /* SHADING_KERNEL */

#endif /* CU_LIGHT_H */

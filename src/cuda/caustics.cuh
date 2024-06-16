#ifndef CU_CAUSTICS_H
#define CU_CAUSTICS_H

#include "math.cuh"
#include "ocean_utils.cuh"
#include "random.cuh"
#include "utils.cuh"

#define CAUSTICS_DEBUG 0

// TODO: This must be improved. The set of all point for which we have convergence is extremely small, we must get
// a good initial guess if we want to achieve anything here.
__device__ void caustics_get_domain(const GBufferData data, vec3& base_point, vec3& edge1, vec3& edge2) {
  // Algorithm
  // If above surface
  //  Consider line defined by shading point and shading normal
  //  Intersection of this line with ocean plane forms the center of one of the domain edges
  //  Project line onto ocean plane, from this we get center of the opposite domain edge
  //  From this we get all the necessary vectors.
  //

  // Preliminary version
  const float height = device.scene.ocean.height;

  const vec3 projected_point = get_vector(data.position.x, height, data.position.z);

  base_point = add_vector(projected_point, get_vector(-10.0f, 0.0f, -10.0f));
  edge1      = get_vector(20.0f, 0.0f, 0.0f);
  edge2      = get_vector(0.0f, 0.0f, 20.0f);
}

__device__ vec3 caustics_transform(const vec3 V, const vec3 normal, const bool is_refraction) {
  if (is_refraction) {
    bool total_reflection;
    return refract_vector(V, normal, device.scene.ocean.refractive_index, total_reflection);
  }
  else {
    return reflect_vector(V, normal);
  }
}

__device__ void caustics_spherical_coords(const vec3 V, float& azimuth, float& altitude) {
  altitude = acosf(V.y);
  azimuth  = atan2f(V.z, V.x);

  if (azimuth < 0.0f)
    azimuth += 2.0f * PI;
}

__device__ void caustics_compute_residual(
  const GBufferData data, const ushort2 index, const vec3 light_direction, const bool is_refraction, const vec3 point, float2& residual) {
  vec3 V = sub_vector(data.position, point);

  const float current_dist = get_length(V);

  if (current_dist < 1e-4f) {
    residual = make_float2(FLT_MAX, FLT_MAX);
    return;
  }

  const float current_recip_dist = 1.0f / current_dist;

  V = scale_vector(V, current_recip_dist);

  const vec3 normal = scale_vector(ocean_get_normal(point, OCEAN_ITERATIONS_NORMAL_CAUSTICS), (is_refraction) ? -1.0f : 1.0f);

  const vec3 L = caustics_transform(V, normal, is_refraction);

  float azimuth_light, altitude_light;
  caustics_spherical_coords(light_direction, azimuth_light, altitude_light);

  float azimuth_L, altitude_L;
  caustics_spherical_coords(L, azimuth_L, altitude_L);

  const float diff_altitude = altitude_light - altitude_L;
  float diff_azimuth        = azimuth_light - azimuth_L;

  if (diff_azimuth < -PI) {
    diff_azimuth += 2.0f * PI;
  }
  else if (diff_azimuth > PI) {
    diff_azimuth -= 2.0f * PI;
  }

  residual = make_float2(diff_altitude, diff_azimuth);
}

__device__ bool caustics_find_connection_point(
  const GBufferData data, const ushort2 index, const vec3 light_direction, const bool is_refraction, const uint32_t iteration,
  vec3& connection_point) {
  const float2 initial_sample = quasirandom_sequence_2D(QUASI_RANDOM_TARGET_CAUSTIC_INITIAL + iteration, index);

  vec3 domain_base, domain_edge1, domain_edge2;
  caustics_get_domain(data, domain_base, domain_edge1, domain_edge2);

  const vec3 initial_sampling_point =
    add_vector(domain_base, add_vector(scale_vector(domain_edge1, initial_sample.x), scale_vector(domain_edge2, initial_sample.y)));

#if CAUSTICS_DEBUG
  if (is_selected_pixel(index)) {
    printf("==== START SEARCH ====\n");
    printf("Shading Point:          %f %f\n", data.position.x, data.position.z);
    printf("Initial Sample:         %f %f\n", initial_sampling_point.x, initial_sampling_point.z);
    printf("Light Dir:              %f %f %f\n", light_direction.x, light_direction.y, light_direction.z);
  }
#endif /* CAUSTICS_DEBUG */

  vec3 current_point = initial_sampling_point;

  float beta             = 1.0f;
  const float step_scale = 1.0f;
  const float res_tol    = 1e-4f;

  float res_norm = FLT_MAX;

  for (int i = 0; i < 16; i++) {
#if CAUSTICS_DEBUG
    if (is_selected_pixel(index)) {
      printf("==== Iter %d ====\n", i);
      printf("P:                    %f %f\n", current_point.x, current_point.z);
    }

#endif /* CAUSTICS_DEBUG */

    const float du = 2.0f * eps * fmaxf(1.0f, fabsf(current_point.x));
    const float dv = 2.0f * eps * fmaxf(1.0f, fabsf(current_point.z));

    const vec3 current_point_du = add_vector(current_point, get_vector(du, 0.0f, 0.0f));
    const vec3 current_point_dv = add_vector(current_point, get_vector(0.0f, 0.0f, dv));

    float2 residual, residual_du, residual_dv;
    caustics_compute_residual(data, index, light_direction, is_refraction, current_point, residual);
    caustics_compute_residual(data, index, light_direction, is_refraction, current_point_du, residual_du);
    caustics_compute_residual(data, index, light_direction, is_refraction, current_point_dv, residual_dv);

    const float H_00 = (residual_du.x - residual.x) / du;
    const float H_10 = (residual_du.y - residual.y) / du;
    const float H_01 = (residual_dv.x - residual.x) / dv;
    const float H_11 = (residual_dv.y - residual.y) / dv;

    const float determinant = H_00 * H_11 - H_10 * H_01;

#if CAUSTICS_DEBUG
    if (is_selected_pixel(index)) {
      printf("H:                    %f %f\n", H_00, H_01);
      printf("                      %f %f\n", H_10, H_11);
      printf("Determinant:          %f\n", determinant);

      const float m = 0.5f * (H_00 + H_11);
      const float s = sqrtf(fabsf(m * m - determinant));

      const bool is_real = m * m - determinant >= 0.0f;

      if (is_real) {
        printf("Eigenvalues:          %f %f\n", m + s, m - s);
      }
      else {
        printf("Eigenvalues:          %f+%fi %f-%fi\n", m, s, m, s);
      }
    }
#endif /* CAUSTICS_DEBUG */

    if (fabsf(determinant) < 1e-6f)
      return false;

    const float recip_determinant = 1.0f / determinant;

    const float step_u = recip_determinant * (residual.x * H_11 - residual.y * H_01);
    const float step_v = recip_determinant * (residual.y * H_00 - residual.x * H_10);

    const float current_res_norm = sqrtf(residual.x * residual.x + residual.y * residual.y);

    res_norm = current_res_norm;

#if CAUSTICS_DEBUG
    if (is_selected_pixel(index)) {
      printf("Residual:             %f %f\n", residual.x, residual.y);
      printf("Residual Norm:        %f\n", res_norm);
    }
#endif /* CAUSTICS_DEBUG */

    if (isnan(res_norm))
      return false;

    // We are too far away, it is not worth it to keep trying.
    if (res_norm > 1.0f)
      return false;

    if (res_norm < res_tol)
      break;

#if CAUSTICS_DEBUG
    if (is_selected_pixel(index))
      printf("Step:                 %f %f\n", step_scale * beta * step_u, step_scale * beta * step_v);
#endif /* CAUSTICS_DEBUG */

    current_point.x -= step_scale * beta * step_u;
    current_point.z -= step_scale * beta * step_v;
  }

  if (!(res_norm < res_tol))
    return false;

  current_point.y = device.scene.ocean.height + ocean_get_height(current_point, OCEAN_ITERATIONS_INTERSECTION);

  connection_point = current_point;

#if CAUSTICS_DEBUG
  if (is_selected_pixel(index)) {
    printf("SUCCESS\nSUCCESS\nSUCCESS\n");
  }
#endif /* CAUSTICS_DEBUG */

  return true;
}

#endif /* CU_CAUSTICS_H */

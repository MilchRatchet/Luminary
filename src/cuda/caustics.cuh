#ifndef CU_CAUSTICS_H
#define CU_CAUSTICS_H

#include "math.cuh"
#include "ocean_utils.cuh"
#include "random.cuh"
#include "utils.cuh"

struct CausticsSamplingDomain {
  vec3 base;
  vec3 edge1;
  vec3 edge2;
  float area;
  float ior_in;
  float ior_out;
} typedef CausticsSamplingDomain;

// Assuming a flat plane with a uniform shading normal, find the unique solution.
__device__ vec3 caustics_solve_for_normal(const GBufferData data, const vec3 L, const bool is_underwater, const float dx, const float dz) {
  // TODO: Come up with a better solution
  // The issue is that getting the domain bounds by solving for the most extreme normals does not
  // work because those normals might not be capable of ever being present in a valid connection point
  // due to things like V.y < 0.0f for reflections. There are two things I came up with so far:
  // 1) Adjust the most extreme normal so that it is a normal that can produce valid points.
  //    (This would in theory work but in practice the bounds would still be gigantic)
  // 2) Assume that most connection points come for flat parts of the surface, i.e. where the normal
  //    is mostly just looking up. This is what I am doing here through this really small factor here.
  const float max_height_change = fminf(OCEAN_LIPSCHITZ * 0.06f + eps, 0.12f);

  // Construct normal
  const float hx = dx * max_height_change;
  const float hz = dz * max_height_change;

  const vec3 forward = normalize_vector(get_vector(L.x, 0.0f, L.z));
  const vec3 right   = cross_product(forward, get_vector(0.0f, 1.0f, 0.0f));

  vec3 normal = get_vector(0.0f, 1.0f, 0.0f);
  normal      = add_vector(normal, scale_vector(right, hx));
  normal      = add_vector(normal, scale_vector(forward, hz));

  normal = normalize_vector(normal);

  // Get view vector
  vec3 V;
  if (is_underwater) {
    bool total_reflection;
    V = refract_vector(L, normal, 1.0f / device.scene.ocean.refractive_index, total_reflection);
  }
  else {
    V = reflect_vector(L, normal);
  }

  // Get intersection distance from position to ocean plane along V
  const float dist = ((data.position.y - device.scene.ocean.height) / V.y);

  return sub_vector(data.position, scale_vector(V, dist));
}

// TODO: This must be improved. The set of all point for which we have convergence is extremely small, we must get
// a good initial guess if we want to achieve anything here.
__device__ CausticsSamplingDomain caustics_get_domain(const GBufferData data, const vec3 L, const bool is_underwater) {
  // Some things:
  // We only care about caustics in the direct of the light, the other are super rare and few, we don't care about them.
  // This is my idea:
  // First solve for the perfect reflection/refraction vector, this has a closed form solution
  // Then based on the amplitude, create a domain around the above obtained point
  // This domain should be stretched along the direction towards the light based on the amplitude
  // Small amplitude -> small domain
  // This depends on a few things like distance to the initial point etc
  // Once we have a domain like that, clip it using the normal of the shading point
  // This process should be done once and the domain should be reused for all attemps.

  CausticsSamplingDomain domain;

  domain.base  = caustics_solve_for_normal(data, L, is_underwater, 1.0f, 1.0f);
  domain.edge1 = caustics_solve_for_normal(data, L, is_underwater, 1.0f, -1.0f);
  domain.edge2 = caustics_solve_for_normal(data, L, is_underwater, -1.0f, 1.0f);

  domain.edge1 = sub_vector(domain.edge1, domain.base);
  domain.edge2 = sub_vector(domain.edge2, domain.base);

  domain.area = get_length(cross_product(domain.edge1, domain.edge2));

  domain.ior_in  = (is_underwater) ? device.scene.ocean.refractive_index : 1.0f;
  domain.ior_out = (is_underwater) ? 1.0f : device.scene.ocean.refractive_index;

  return domain;
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
  const GBufferData data, const ushort2 index, const CausticsSamplingDomain domain, const bool is_refraction, const uint32_t iteration,
  vec3& point, float& target_weight, float& recip_pdf) {
  const float2 sample = quasirandom_sequence_2D(QUASI_RANDOM_TARGET_CAUSTIC_INITIAL + iteration, index);

  point   = add_vector(domain.base, add_vector(scale_vector(domain.edge1, sample.x), scale_vector(domain.edge2, sample.y)));
  point.y = device.scene.ocean.height + ocean_get_height(point, OCEAN_ITERATIONS_INTERSECTION);

  vec3 V = sub_vector(data.position, point);

  const float dist_sq = dot_product(V, V);
  V                   = normalize_vector(V);

  const vec3 normal = scale_vector(ocean_get_normal(point, OCEAN_ITERATIONS_NORMAL_CAUSTICS), (is_refraction) ? -1.0f : 1.0f);

  const float NdotV = dot_product(V, normal);

  if (NdotV < 0.0f)
    return false;

  const vec3 L = caustics_transform(V, normal, is_refraction);

  const vec3 sky_point = world_to_sky_transform(point);

  const bool sun_hit = sphere_ray_hit(L, sky_point, device.sun_pos, SKY_SUN_RADIUS * device.scene.ocean.caustics_regularization);

  if (!sun_hit)
    return false;

  recip_pdf = NdotV * domain.area / dist_sq;

  bool total_reflection;
  const vec3 refraction_dir = refract_vector(V, normal, domain.ior_in / domain.ior_out, total_reflection);

  const float reflection_coefficient =
    ocean_reflection_coefficient(normal, scale_vector(V, -1.0f), refraction_dir, domain.ior_in, domain.ior_out);

  target_weight = (is_refraction) ? 1.0f - reflection_coefficient : reflection_coefficient;

  return true;
}

#endif /* CU_CAUSTICS_H */

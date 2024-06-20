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

// Assuming a flat plane with a normal of (0,1,0), find the unique solution.
__device__ vec3 caustics_solve_for_normal(const GBufferData data, const vec3 L, const bool is_underwater, const float dx, const float dz) {
  // Get view vector
  vec3 V;
  if (is_underwater) {
    bool total_reflection;
    V = refract_vector(L, get_vector(0.0f, 1.0f, 0.0f), 1.0f / device.scene.ocean.refractive_index, total_reflection);
  }
  else {
    V = get_vector(-L.x, L.y, -L.z);
  }

  // Get intersection distance from position to ocean plane along V
  const float dist = fabsf((data.position.y - device.scene.ocean.height) / V.y);

  return sub_vector(data.position, scale_vector(V, dist));
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

__device__ CausticsSamplingDomain caustics_get_domain(const GBufferData data, const vec3 L, const bool is_underwater) {
  const vec3 center = caustics_solve_for_normal(data, L, is_underwater, 0.0f, 0.0f);

  // Fast path for flat ocean
  // TODO: Allow for soft shadows
  if (device.scene.ocean.amplitude == 0.0f) {
    CausticsSamplingDomain domain;

    domain.base  = center;
    domain.edge1 = get_vector(0.0f, 0.0f, 0.0f);
    domain.edge2 = get_vector(0.0f, 0.0f, 0.0f);

    domain.area = sample_sphere_solid_angle(device.sun_pos, SKY_SUN_RADIUS, world_to_sky_transform(data.position));

    domain.ior_in  = (is_underwater) ? device.scene.ocean.refractive_index : 1.0f;
    domain.ior_out = (is_underwater) ? 1.0f : device.scene.ocean.refractive_index;

    return domain;
  }

  const vec3 center_dir = normalize_vector(sub_vector(center, data.position));

  float azimuth, altitude;
  direction_to_angles(center_dir, azimuth, altitude);

  const float angle = __saturatef(0.3f * device.scene.ocean.caustics_domain_scale);

  const vec3 v0_dir = angles_to_direction(altitude - angle, azimuth - angle);
  const vec3 v1_dir = angles_to_direction(altitude - angle, azimuth + angle);
  const vec3 v2_dir = angles_to_direction(altitude + angle, azimuth - angle);

  const float v0_dist = fabsf(data.position.y - device.scene.ocean.height) / fmaxf(0.01f, fabsf(v0_dir.y));
  const float v1_dist = fabsf(data.position.y - device.scene.ocean.height) / fmaxf(0.01f, fabsf(v1_dir.y));
  const float v2_dist = fabsf(data.position.y - device.scene.ocean.height) / fmaxf(0.01f, fabsf(v2_dir.y));

  const vec3 v0 = add_vector(data.position, scale_vector(v0_dir, v0_dist));
  const vec3 v1 = add_vector(data.position, scale_vector(v1_dir, v1_dist));
  const vec3 v2 = add_vector(data.position, scale_vector(v2_dir, v2_dist));

  CausticsSamplingDomain domain;

  domain.base  = v0;
  domain.edge1 = sub_vector(v1, v0);
  domain.edge2 = sub_vector(v2, v0);

  domain.area = get_length(cross_product(domain.edge1, domain.edge2));

  domain.ior_in  = (is_underwater) ? device.scene.ocean.refractive_index : 1.0f;
  domain.ior_out = (is_underwater) ? 1.0f : device.scene.ocean.refractive_index;

  return domain;
}

__device__ bool caustics_find_connection_point(
  const GBufferData data, const ushort2 index, const CausticsSamplingDomain domain, const bool is_refraction, const uint32_t iteration,
  vec3& point, float& sample_weight) {
  const float2 sample = quasirandom_sequence_2D(QUASI_RANDOM_TARGET_CAUSTIC_INITIAL + iteration, index);

  point   = add_vector(domain.base, add_vector(scale_vector(domain.edge1, sample.x), scale_vector(domain.edge2, sample.y)));
  point.y = device.scene.ocean.height + ocean_get_height(point, OCEAN_ITERATIONS_INTERSECTION);

  vec3 V = sub_vector(data.position, point);

  const float dist_sq = dot_product(V, V);
  V                   = scale_vector(V, 1.0f / sqrtf(dist_sq));

  const vec3 normal = scale_vector(ocean_get_normal_fast(point, OCEAN_ITERATIONS_NORMAL_CAUSTICS), (is_refraction) ? -1.0f : 1.0f);

  const float NdotV = dot_product(V, normal);

  if (NdotV < 0.0f)
    return false;

  const vec3 L = caustics_transform(V, normal, is_refraction);

  const vec3 sky_point = world_to_sky_transform(point);
  const bool sun_hit   = sphere_ray_hit(L, sky_point, device.sun_pos, SKY_SUN_RADIUS * device.scene.ocean.caustics_regularization);

  if (!sun_hit)
    return false;

  // Assume flat plane for the dot product because that is how we sampled it.
  // TODO: Figure out the correct weight, for some reason I have a severe uniform energy loss that seems to be exactly 2 * 2 * PI,
  // that makes no sense but I will just take it for now.
  sample_weight = (device.scene.ocean.amplitude > 0.0f) ? 2.0f * 2.0f * PI * fabsf(V.y) * domain.area / dist_sq : domain.area;

  return true;
}

#endif /* CU_CAUSTICS_H */

#ifndef CU_CAUSTICS_H
#define CU_CAUSTICS_H

#include "material.cuh"
#include "math.cuh"
#include "ocean_utils.cuh"
#include "random.cuh"
#include "ris.cuh"
#include "utils.cuh"

struct CausticsSamplingDomain {
  bool valid;
  vec3 base;
  vec3 edge1;
  vec3 edge2;
  float area;
} typedef CausticsSamplingDomain;

// Assuming a flat plane with a normal of (0,1,0), find the unique solution.
template <MaterialType TYPE>
LUMINARY_FUNCTION vec3 caustics_solve_for_normal(const MaterialContext<TYPE> ctx, const vec3 L, const bool is_underwater, bool& valid) {
  // Get view vector
  vec3 ray;
  if (is_underwater) {
    bool total_reflection;
    ray = refract_vector(L, get_vector(0.0f, 1.0f, 0.0f), 1.0f / device.ocean.refractive_index, total_reflection);
    ray = scale_vector(ray, -1.0f);
  }
  else {
    ray = get_vector(L.x, -L.y, L.z);
  }

  const float dist = ocean_intersection_distance(ctx.position, ray);

  valid = dist != FLT_MAX;

  return add_vector(ctx.position, scale_vector(ray, dist));
}

LUMINARY_FUNCTION vec3 caustics_transform(const vec3 V, const vec3 normal, const bool is_refraction) {
  if (is_refraction) {
    bool total_reflection;
    return refract_vector(V, normal, device.ocean.refractive_index, total_reflection);
  }
  else {
    return reflect_vector(V, normal);
  }
}

LUMINARY_FUNCTION bool caustics_is_fast_path(const uint16_t state) {
  bool fast_path = false;
  fast_path |= (device.ocean.amplitude == 0.0f);  // Fast path is assuming amplitude == 0, so if that is actually true we can just do it.
  fast_path |= (device.ocean.caustics_active == false);  // Caustics not active still means we want the shift in direction.
  fast_path |= ((state & STATE_FLAG_DELTA_PATH) == 0);   // If we are indirect lighting, proper caustics are too noisy.

  return fast_path;
}

template <MaterialType TYPE>
LUMINARY_FUNCTION CausticsSamplingDomain caustics_get_domain_fast(const MaterialContext<TYPE> ctx, const vec3 L, const bool is_underwater) {
  bool is_valid;
  const vec3 center = caustics_solve_for_normal(ctx, L, is_underwater, is_valid);

  CausticsSamplingDomain domain;

  domain.valid = is_valid;
  domain.base  = center;
  domain.edge1 = get_vector(0.0f, 0.0f, 0.0f);
  domain.edge2 = get_vector(0.0f, 0.0f, 0.0f);

  domain.area = (is_valid) ? sample_sphere_solid_angle(device.sky.sun_pos, SKY_SUN_RADIUS, world_to_sky_transform(ctx.position)) : 0.0f;

  return domain;
}

template <MaterialType TYPE>
LUMINARY_FUNCTION CausticsSamplingDomain caustics_get_domain_full(const MaterialContext<TYPE> ctx, const vec3 L, const bool is_underwater) {
  bool is_valid;
  const vec3 center = caustics_solve_for_normal(ctx, L, is_underwater, is_valid);

  const vec3 center_dir = normalize_vector(sub_vector(center, ctx.position));

  float azimuth, altitude;
  direction_to_angles(center_dir, azimuth, altitude);

  const float angle        = 0.3f * device.ocean.caustics_domain_scale;
  const float plane_height = center.y;

  const vec3 v0_dir = angles_to_direction(altitude - angle, azimuth - angle);
  const vec3 v1_dir = angles_to_direction(altitude - angle, azimuth + angle);
  const vec3 v2_dir = angles_to_direction(altitude + angle, azimuth - angle);

  const float v0_dist = fabsf(ctx.position.y - plane_height) / fmaxf(0.01f, fabsf(v0_dir.y));
  const float v1_dist = fabsf(ctx.position.y - plane_height) / fmaxf(0.01f, fabsf(v1_dir.y));
  const float v2_dist = fabsf(ctx.position.y - plane_height) / fmaxf(0.01f, fabsf(v2_dir.y));

  const vec3 v0 = add_vector(ctx.position, scale_vector(v0_dir, v0_dist));
  const vec3 v1 = add_vector(ctx.position, scale_vector(v1_dir, v1_dist));
  const vec3 v2 = add_vector(ctx.position, scale_vector(v2_dir, v2_dist));

  CausticsSamplingDomain domain;

  domain.valid = is_valid;
  domain.base  = v0;
  domain.edge1 = sub_vector(v1, v0);
  domain.edge2 = sub_vector(v2, v0);

  domain.area = get_length(cross_product(domain.edge1, domain.edge2));

  return domain;
}

template <MaterialType TYPE>
LUMINARY_FUNCTION bool caustics_find_connection_point(
  const MaterialContext<TYPE> ctx, const PathID& path_id, const CausticsSamplingDomain domain, const bool is_refraction,
  const uint32_t iteration, const uint32_t num_iterations, vec3& point, float& sample_weight) {
  const float2 initial_random = random_2D(MaterialContext<TYPE>::RANDOM_DL_SUN::CAUSTIC_INITIAL + iteration, path_id);
  const float2 sample         = ris_transform_stratum_2D(iteration, num_iterations, initial_random);

  point = add_vector(domain.base, add_vector(scale_vector(domain.edge1, sample.x), scale_vector(domain.edge2, sample.y)));

  vec3 V = sub_vector(ctx.position, point);

  const vec3 normal = scale_vector(ocean_get_normal_fast(point, OCEAN_ITERATIONS_NORMAL_CAUSTICS), (is_refraction) ? -1.0f : 1.0f);

  // V does not need to be normalized here since we only care about the sign.
  if (dot_product(V, normal) < 0.0f)
    return false;

  const float inv_dist_sq = 1.0f / dot_product(V, V);

  V = scale_vector(V, sqrtf(inv_dist_sq));

  const vec3 L = caustics_transform(V, normal, is_refraction);

  const vec3 sky_point = world_to_sky_transform(point);
  const bool sun_hit   = sphere_ray_hit(L, sky_point, device.sky.sun_pos, SKY_SUN_RADIUS);

  if (sun_hit == false)
    return false;

  // Assume flat plane for the dot product because that is how we sampled it.
  // Note: This is just one over the PDF, the target distribution is just a dirac delta.
  sample_weight = fabsf(V.y) * domain.area * inv_dist_sq;

  return true;
}

template <MaterialType TYPE>
LUMINARY_FUNCTION float caustics_sample_fast(MaterialContext<TYPE> ctx, const vec3 sun_dir, const PathID& path_id, vec3& connection_point) {
  // Caustics are only for underwater starting with v1.2.0
  constexpr bool IS_UNDERWATER = true;

  const CausticsSamplingDomain sampling_domain = caustics_get_domain_fast(ctx, sun_dir, IS_UNDERWATER);

  connection_point = sampling_domain.base;

  return sampling_domain.area;
}

template <MaterialType TYPE>
LUMINARY_FUNCTION float caustics_sample_full(MaterialContext<TYPE> ctx, const vec3 sun_dir, const PathID& path_id, vec3& connection_point) {
  // Caustics are only for underwater starting with v1.2.0
  constexpr bool IS_UNDERWATER = true;

  const CausticsSamplingDomain sampling_domain = caustics_get_domain_full(ctx, sun_dir, IS_UNDERWATER);

  connection_point = get_vector(0.0f, 0.0f, 0.0f);

  if (sampling_domain.valid == false)
    return 0.0f;

  const float resampling_random = random_1D(MaterialContext<TYPE>::RANDOM_DL_SUN::CAUSTIC_RESAMPLING, path_id);
  const uint32_t num_samples    = device.ocean.caustics_ris_sample_count + 1;

  RISStratifiedReservoir reservoir = ris_stratified_reservoir_init(resampling_random, num_samples);

  const float mis_weight = 1.0f / num_samples;

  uint32_t index;
  while (ris_stratified_reservoir_next(reservoir, index)) {
    vec3 sample_point;
    float sample_weight = 0.0f;
    const bool valid_hit =
      caustics_find_connection_point(ctx, path_id, sampling_domain, IS_UNDERWATER, index, num_samples, sample_point, sample_weight);

    const float target = (valid_hit) ? 1.0f : 0.0f;
    sample_weight      = (valid_hit) ? mis_weight * sample_weight : 0.0f;

    if (ris_stratified_reservoir_add_sample(reservoir, target, sample_weight)) {
      connection_point = sample_point;
    }
  }

  float connection_weight = ris_stratified_reservoir_get_sampling_weight(reservoir);

  // Make no mistake. I do in fact have no idea what I am doing. So I just empirically gathered that these
  // weights are correct (they are very unlikely to be correct). I will look into fixing this the moment
  // I start caring.

  // Inspired by the famous factor required for refraction when sampling importance. Note that one of the IOR is 1.0f.
  connection_weight *= device.ocean.refractive_index * device.ocean.refractive_index;

  if (IS_UNDERWATER) {
    // ... and why not apply it again, we are just sampling some extra importance clearly. And a 2.0f for good measure.
    connection_weight *= device.ocean.refractive_index * device.ocean.refractive_index * 2.0f;
  }

  return connection_weight;
}

template <MaterialType TYPE>
LUMINARY_FUNCTION float caustics_sample(MaterialContext<TYPE> ctx, const vec3 sun_dir, const PathID& path_id, vec3& connection_point) {
  if constexpr (MaterialContext<TYPE>::FORCE_FAST_OCEAN_CAUSTICS) {
    return caustics_sample_fast(ctx, sun_dir, path_id, connection_point);
  }
  else {
    if (caustics_is_fast_path(ctx.state)) {
      return caustics_sample_fast(ctx, sun_dir, path_id, connection_point);
    }
    else {
      return caustics_sample_full(ctx, sun_dir, path_id, connection_point);
    }
  }
}

#endif /* CU_CAUSTICS_H */

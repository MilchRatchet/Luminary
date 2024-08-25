#ifndef CU_BRIDGES_H
#define CU_BRIDGES_H

#if defined(OPTIX_KERNEL) && defined(VOLUME_KERNEL)

#include "light.cuh"
#include "math.cuh"
#include "memory.cuh"
#include "optix_shadow_trace.cuh"
#include "random.cuh"
#include "utils.cuh"
#include "utils.h"
#include "volume_utils.cuh"

__device__ Quaternion bridges_compute_rotation(const GBufferData data, const vec3 light_point, const vec3 end_vertex) {
  const vec3 target_dir = normalize_vector(sub_vector(light_point, data.position));
  const vec3 actual_dir = normalize_vector(sub_vector(end_vertex, data.position));

  const float dot = dot_product(actual_dir, target_dir);

  Quaternion rotation;

  if (dot > 0.999f) {
    rotation.x = 0.0f;
    rotation.y = 0.0f;
    rotation.z = 0.0f;
    rotation.w = 1.0f;

    return rotation;
  }
  else if (dot < -0.999f) {
    rotation.x = 1.0f;
    rotation.y = 0.0f;
    rotation.z = 0.0f;
    rotation.w = 0.0f;

    return rotation;
  }

  const vec3 cross = cross_product(actual_dir, target_dir);

  rotation.x = cross.x;
  rotation.y = cross.y;
  rotation.z = cross.z;
  rotation.w = 1.0f + dot;

  rotation = normalize_quaternion(rotation);

  return rotation;
}

__device__ float bridges_log_factorial(const uint32_t vertex_count) {
  if (vertex_count == 1)
    return 0.0f;

  const float n = (float) (vertex_count - 1);

  // Ramanjuan approximation
  const float t0 = n * logf(n);
  const float t1 = (1.0f / 6.0f) * logf(n * (1.0f + 4.0f * n * (1.0f + 2.0f * n)));
  const float t2 = 0.5f * logf(PI);

  return t0 + t1 + t2 - n;
}

__device__ float bridges_get_vertex_count_importance(const uint32_t vertex_count, const float effective_dist) {
  const uint32_t lut_offset = (vertex_count - 1) * 21;

  const float min_dist    = __ldg(device.bridge_lut + lut_offset + 0);
  const float center_dist = __ldg(device.bridge_lut + lut_offset + 1);
  const float max_dist    = __ldg(device.bridge_lut + lut_offset + 2);

  if (effective_dist > max_dist)
    return 0.0f;

  if (effective_dist < min_dist) {
    const float linear_falloff = __ldg(device.bridge_lut + lut_offset + 3);

    return linear_falloff * effective_dist / min_dist;
  }

  float step, left_dist;
  uint32_t left_index;

  if (effective_dist < center_dist) {
    step = (center_dist - min_dist) * 0.25f;

    const uint32_t step_id = (uint32_t) ((effective_dist - min_dist) / step);

    left_dist  = min_dist + step_id * step;
    left_index = 3 + 2 * step_id;
  }
  else {
    step = (max_dist - center_dist) * 0.25f;

    const uint32_t step_id = (uint32_t) ((effective_dist - center_dist) / step);

    left_dist  = center_dist + step_id * step;
    left_index = 3 + 2 * (step_id + 4);
  }

  const float y0  = __ldg(device.bridge_lut + lut_offset + left_index + 0);
  const float dy0 = __ldg(device.bridge_lut + lut_offset + left_index + 1);
  const float y1  = __ldg(device.bridge_lut + lut_offset + left_index + 2);
  const float dy1 = __ldg(device.bridge_lut + lut_offset + left_index + 3);

  const float t  = __saturatef((effective_dist - left_dist) / step);
  const float t2 = t * t;
  const float t3 = t2 * t;

  const float h00 = 2.0f * t3 - 3.0f * t2 + 1.0f;
  const float h10 = t3 - 2.0f * t2 + t;
  const float h01 = -2.0f * t3 + 3.0f * t2;
  const float h11 = t3 - t2;

  return h00 * y0 + h10 * step * dy0 + h01 * y1 + h11 * step * dy1;
}

__device__ uint32_t
  bridges_sample_vertex_count(const VolumeDescriptor volume, const float light_dist, const uint32_t seed, const ushort2 pixel, float& pdf) {
  const float effective_dist = light_dist * volume.max_scattering;

  float random = quasirandom_sequence_1D(QUASI_RANDOM_TARGET_BRIDGE_VERTEX_COUNT + seed, pixel);

  float sum_importance = 0.0f;
  float count_importance[32];

  for (uint32_t i = 0; i < device.bridge_settings.max_num_vertices; i++) {
    const float importance = bridges_get_vertex_count_importance(i + 1, effective_dist);

    count_importance[i] = importance;
    sum_importance += importance;
  }

  random *= sum_importance;

  uint32_t selected_vertex_count = 0;

  for (uint32_t i = 0; i < device.bridge_settings.max_num_vertices; i++) {
    const float importance = count_importance[i];

    random -= importance;

    if (random < 0.0f) {
      selected_vertex_count = i;

      pdf = importance / sum_importance;
      break;
    }
  }

  return 1 + selected_vertex_count;
}

__device__ RGBF bridges_sample_bridge(
  const GBufferData data, const VolumeDescriptor volume, const vec3 light_point, const uint32_t seed, const ushort2 pixel, float& path_pdf,
  vec3& end_vertex, float& scale) {
  ////////////////////////////////////////////////////////////////////
  // Sample vertex count
  ////////////////////////////////////////////////////////////////////
  float vertex_count_pdf;
  const uint32_t vertex_count =
    bridges_sample_vertex_count(volume, get_length(sub_vector(light_point, data.position)), seed, pixel, vertex_count_pdf);

  vec3 current_point     = data.position;
  vec3 current_direction = normalize_vector(sub_vector(light_point, data.position));

  float sum_dist = 0.0f;

  {
    const float random_dist = quasirandom_sequence_1D(QUASI_RANDOM_TARGET_BRIDGE_DISTANCE + seed * 32 + 0, pixel);
    const float dist        = -logf(random_dist);
    current_point           = add_vector(current_point, scale_vector(current_direction, dist));

    sum_dist += dist;
  }

  for (uint32_t i = 1; i < vertex_count; i++) {
    const float2 random_phase = quasirandom_sequence_2D(QUASI_RANDOM_TARGET_BRIDGE_PHASE + seed * 32 + i, pixel);
    const float random_method = quasirandom_sequence_1D(QUASI_RANDOM_TARGET_BRIDGE_PHASE_METHOD + seed * 32 + i, pixel);

    current_direction = jendersie_eon_phase_sample(current_direction, device.scene.fog.droplet_diameter, random_phase, random_method);

    const float random_dist = quasirandom_sequence_1D(QUASI_RANDOM_TARGET_BRIDGE_DISTANCE + seed * 32 + i, pixel);
    const float dist        = -logf(random_dist);
    current_point           = add_vector(current_point, scale_vector(current_direction, dist));

    sum_dist += dist;
  }

  const float target_scale = get_length(sub_vector(light_point, data.position));
  const float actual_scale = get_length(sub_vector(current_point, data.position));
  scale                    = target_scale / actual_scale;

  sum_dist *= scale;

  end_vertex = current_point;

  RGBF path_weight = get_color(
    expf((vertex_count - 1) * logf(volume.scattering.r) - sum_dist * (volume.scattering.r + volume.absorption.r)),
    expf((vertex_count - 1) * logf(volume.scattering.g) - sum_dist * (volume.scattering.g + volume.absorption.g)),
    expf((vertex_count - 1) * logf(volume.scattering.b) - sum_dist * (volume.scattering.b + volume.absorption.b)));

  const float log_path_pdf = bridges_log_factorial(vertex_count) - vertex_count * logf(sum_dist);

  path_pdf = vertex_count_pdf * expf(log_path_pdf) * target_scale * target_scale * target_scale;

  return path_weight;
}

__device__ RGBF bridges_evaluate_bridge(
  const GBufferData data, const VolumeDescriptor volume, const uint32_t light_id, const uint32_t seed, const ushort2 pixel,
  const Quaternion rotation, const float scale) {
  const TriangleLight light = load_triangle_light(device.scene.triangle_lights, light_id);

  const float2 random_light_point = quasirandom_sequence_2D(QUASI_RANDOM_TARGET_BRIDGE_LIGHT_POINT + seed, pixel);

  RGBF light_color;
  float solid_angle;
  float light_dist;
  const vec3 light_dir = light_sample_triangle(light, data, random_light_point, solid_angle, light_dist, light_color);

  const vec3 light_point = add_vector(data.position, scale_vector(light_dir, light_dist));

  const vec3 initial_direction = sub_vector(light_point, data.position);

  ////////////////////////////////////////////////////////////////////
  // Sample vertex count
  ////////////////////////////////////////////////////////////////////
  float vertex_count_pdf;
  const uint32_t vertex_count = bridges_sample_vertex_count(volume, get_length(initial_direction), seed, pixel, vertex_count_pdf);

  vec3 current_point     = data.position;
  vec3 current_direction = rotate_vector_by_quaternion(normalize_vector(initial_direction), rotation);

  float dist;
  RGBF visibility = get_color(1.0f, 1.0f, 1.0f);

  unsigned int compressed_ior = ior_compress(data.ior_in);

  // Apply phase function of first direction.
  const float cos_angle           = -dot_product(current_direction, data.V);
  const JendersieEonParams params = jendersie_eon_phase_parameters(device.scene.fog.droplet_diameter);
  visibility                      = scale_color(visibility, jendersie_eon_phase_function(cos_angle, params));

  float sum_dist = 0.0f;

  {
    const float random_dist = quasirandom_sequence_1D(QUASI_RANDOM_TARGET_BRIDGE_DISTANCE + seed * 32 + 0, pixel);

    dist = -logf(random_dist) * scale;

    visibility = mul_color(visibility, optix_geometry_shadowing(current_point, current_direction, dist, light_id, pixel, compressed_ior));

    sum_dist += dist;
  }

  for (int i = 1; i < vertex_count; i++) {
    current_point = add_vector(current_point, scale_vector(current_direction, dist));

    const float2 random_phase = quasirandom_sequence_2D(QUASI_RANDOM_TARGET_BRIDGE_PHASE + seed * 32 + i, pixel);
    const float random_method = quasirandom_sequence_1D(QUASI_RANDOM_TARGET_BRIDGE_PHASE_METHOD + seed * 32 + i, pixel);

    current_direction = jendersie_eon_phase_sample(current_direction, device.scene.fog.droplet_diameter, random_phase, random_method);

    const float random_dist = quasirandom_sequence_1D(QUASI_RANDOM_TARGET_BRIDGE_DISTANCE + seed * 32 + i, pixel);

    dist = -logf(random_dist) * scale;

    visibility = mul_color(visibility, optix_geometry_shadowing(current_point, current_direction, dist, light_id, pixel, compressed_ior));

    sum_dist += dist;
  }

  visibility.r *= expf((vertex_count - 1) * logf(volume.scattering.r) - sum_dist * (volume.scattering.r + volume.absorption.r));
  visibility.g *= expf((vertex_count - 1) * logf(volume.scattering.g) - sum_dist * (volume.scattering.g + volume.absorption.g));
  visibility.b *= expf((vertex_count - 1) * logf(volume.scattering.b) - sum_dist * (volume.scattering.b + volume.absorption.b));

  return mul_color(light_color, visibility);
}

__device__ RGBF bridges_sample(const GBufferData data, const ushort2 pixel) {
  uint32_t selected_seed     = 0xFFFFFFFF;
  uint32_t selected_light_id = LIGHT_ID_NONE;
  Quaternion selected_rotation;
  float selected_scale;
  float selected_target_pdf = FLT_MAX;

  const JendersieEonParams params = jendersie_eon_phase_parameters(device.scene.fog.droplet_diameter);
  const VolumeDescriptor volume   = volume_get_descriptor_preset(VOLUME_HIT_TYPE(data.hit_id));

  float sum_weight = 0.0f;

  float random_resampling = quasirandom_sequence_1D(QUASI_RANDOM_TARGET_BRIDGE_RESAMPLING, pixel);

  for (uint32_t i = 0; i < device.bridge_settings.num_ris_samples; i++) {
    float sample_pdf = 1.0f;

    ////////////////////////////////////////////////////////////////////
    // Sample light
    ////////////////////////////////////////////////////////////////////

    const float random_light_tree = quasirandom_sequence_1D(QUASI_RANDOM_TARGET_BRIDGE_LIGHT_TREE + i, pixel);

    uint32_t light_list_length;
    float light_list_pdf;
    const uint32_t light_list_ptr =
      light_tree_traverse(data.position, scale_vector(data.V, -1.0f), random_light_tree, light_list_length, light_list_pdf);

    sample_pdf *= light_list_pdf;

    const float random_light_list = quasirandom_sequence_1D(QUASI_RANDOM_TARGET_BRIDGE_LIGHT_LIST + i, pixel);

    const uint32_t light_id = uint32_t(__fmul_rd(random_light_list, light_list_length)) + light_list_ptr;

    sample_pdf *= 1.0f / light_list_length;

    const TriangleLight light = load_triangle_light(device.scene.triangle_lights, light_id);

    const float2 random_light_point = quasirandom_sequence_2D(QUASI_RANDOM_TARGET_BRIDGE_LIGHT_POINT + i, pixel);

    RGBF light_color;
    float solid_angle;
    float light_dist;
    const vec3 light_dir = light_sample_triangle(light, data, random_light_point, solid_angle, light_dist, light_color);

    // We sampled a point that emits no light, skip.
    if (color_importance(light_color) == 0.0f || solid_angle < eps)
      continue;

    const vec3 light_point = add_vector(data.position, scale_vector(light_dir, light_dist));

    const bool light_point_underwater = ocean_get_relative_height(data.position, OCEAN_ITERATIONS_NORMAL) < 0.0f;

    if ((light_point_underwater && data.hit_id == HIT_TYPE_VOLUME_FOG) || (!light_point_underwater && data.hit_id == HIT_TYPE_VOLUME_OCEAN))
      continue;

    sample_pdf *= 1.0f / solid_angle;

    ////////////////////////////////////////////////////////////////////
    // Sample path
    ////////////////////////////////////////////////////////////////////

    float sample_path_pdf;
    float sample_path_scale;
    vec3 sample_path_end_vertex;
    RGBF sample_path_weight =
      bridges_sample_bridge(data, volume, light_point, i, pixel, sample_path_pdf, sample_path_end_vertex, sample_path_scale);

    sample_pdf *= sample_path_pdf;

    ////////////////////////////////////////////////////////////////////
    // Modify path
    ////////////////////////////////////////////////////////////////////

    const Quaternion sample_rotation = bridges_compute_rotation(data, light_point, sample_path_end_vertex);

    const vec3 rotation_initial_direction = rotate_vector_by_quaternion(light_dir, sample_rotation);
    const float cos_angle                 = -dot_product(rotation_initial_direction, data.V);

    sample_path_weight = scale_color(sample_path_weight, jendersie_eon_phase_function(cos_angle, params));

    ////////////////////////////////////////////////////////////////////
    // Resample
    ////////////////////////////////////////////////////////////////////

    const float target_pdf = color_importance(sample_path_weight);

    if (target_pdf == 0.0f)
      continue;

    const float weight = target_pdf / sample_pdf;

    sum_weight += weight;

    const float resampling_probability = weight / sum_weight;

    if (random_resampling < resampling_probability) {
      selected_seed       = i;
      selected_light_id   = light_id;
      selected_rotation   = sample_rotation;
      selected_scale      = sample_path_scale;
      selected_target_pdf = target_pdf;

      random_resampling = random_resampling / resampling_probability;
    }
    else {
      random_resampling = (random_resampling - resampling_probability) / (1.0f - resampling_probability);
    }
  }

  if (selected_light_id == LIGHT_ID_NONE)
    return get_color(0.0f, 0.0f, 0.0f);

  sum_weight /= device.bridge_settings.num_ris_samples;

  ////////////////////////////////////////////////////////////////////
  // Evaluate sampled path
  ////////////////////////////////////////////////////////////////////

  RGBF bridge_color = bridges_evaluate_bridge(data, volume, selected_light_id, selected_seed, pixel, selected_rotation, selected_scale);

  bridge_color = scale_color(bridge_color, sum_weight / selected_target_pdf);

  return bridge_color;
}

#endif /* OPTIX_KERNEL */

#endif /* CU_BRIDGES_H */

#ifndef CU_BRIDGES_H
#define CU_BRIDGES_H

#if defined(OPTIX_KERNEL) && defined(VOLUME_KERNEL)

#include "light.cuh"
#include "math.cuh"
#include "memory.cuh"
#include "ocean_utils.cuh"
#include "optix_include.cuh"
#include "random.cuh"
#include "utils.cuh"
#include "utils.h"
#include "volume_utils.cuh"

#define BRIDGES_MAX_DEPTH (8)

// This must correspond to the G term used when computing the LUT.
#define BRIDGES_HG_G_TERM (0.85f)

__device__ vec3 bridges_phase_sample(const vec3 ray, const float2 r_dir) {
  const float cos_angle = henyey_greenstein_phase_sample(BRIDGES_HG_G_TERM, r_dir.x);

  return phase_sample_basis(cos_angle, r_dir.y, ray);
}

__device__ float bridges_phase_function(const float cos_angle) {
  return henyey_greenstein_phase_function(cos_angle, BRIDGES_HG_G_TERM);
}

__device__ Quaternion bridges_compute_rotation(const vec3 initial_vertex, const vec3 light_point, const vec3 end_vertex) {
  const vec3 target_dir = normalize_vector(sub_vector(light_point, initial_vertex));
  const vec3 actual_dir = normalize_vector(sub_vector(end_vertex, initial_vertex));

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

  // Ramanujan approximation
  const float t0 = n * logf(n);
  const float t1 = (1.0f / 6.0f) * logf(n * (1.0f + 4.0f * n * (1.0f + 2.0f * n)));
  const float t2 = 0.5f * logf(PI);

  return t0 + t1 + t2 - n;
}

// TODO: Package the LUT differently so I achieve good alignment, this can all be done using no more than 2 load instructions.
__device__ float bridges_get_vertex_count_importance(const uint32_t vertex_count, const float effective_dist) {
  const uint32_t lut_offset = (vertex_count - 1) * 21;

  const float min_dist    = __ldg(device.ptrs.bridge_lut + lut_offset + 0);
  const float center_dist = __ldg(device.ptrs.bridge_lut + lut_offset + 1);
  const float max_dist    = __ldg(device.ptrs.bridge_lut + lut_offset + 2);

  if (effective_dist > max_dist)
    return 0.0f;

  if (effective_dist < min_dist) {
    const float linear_falloff = __ldg(device.ptrs.bridge_lut + lut_offset + 3);

    return linear_falloff * effective_dist / min_dist;
  }

  const float low_dist  = (effective_dist < center_dist) ? min_dist : center_dist;
  const float high_dist = (effective_dist < center_dist) ? center_dist : max_dist;

  const float step       = (high_dist - low_dist) * 0.25f;
  const uint32_t step_id = (uint32_t) ((effective_dist - low_dist) / step);
  const float floor_dist = low_dist + step_id * step;
  const uint32_t index   = (effective_dist < center_dist) ? (3 + 2 * step_id) : (3 + 2 * (step_id + 4));

  const float y0  = __ldg(device.ptrs.bridge_lut + lut_offset + index + 0);
  const float dy0 = __ldg(device.ptrs.bridge_lut + lut_offset + index + 1);
  const float y1  = __ldg(device.ptrs.bridge_lut + lut_offset + index + 2);
  const float dy1 = __ldg(device.ptrs.bridge_lut + lut_offset + index + 3);

  const float t  = __saturatef((effective_dist - floor_dist) / step);
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

  ////////////////////////////////////////////////////////////////////
  // Compute importance for each vertex count
  ////////////////////////////////////////////////////////////////////

  float sum_importance = 0.0f;
  float count_importance[BRIDGES_MAX_DEPTH];

  for (uint32_t i = 0; i < device.settings.bridge_max_num_vertices; i++) {
    // TODO: The paper uses some additional terms here for the importance
    const float importance = bridges_get_vertex_count_importance(i + 1, effective_dist);

    count_importance[i] = importance;
    sum_importance += importance;
  }

  ////////////////////////////////////////////////////////////////////
  // Choose a vertex count proportional to the evaluated importance
  ////////////////////////////////////////////////////////////////////

  random *= sum_importance;

  // Fallback values. These come into play if sum_importance is 0.0f.
  // This happens when effective dist is too large.
  uint32_t selected_vertex_count = device.settings.bridge_max_num_vertices - 1;
  pdf                            = 1.0f;

  for (uint32_t i = 0; i < device.settings.bridge_max_num_vertices; i++) {
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
  const vec3 initial_vertex, const VolumeDescriptor volume, const vec3 light_point, const uint32_t seed, const ushort2 pixel,
  float& path_pdf, vec3& end_vertex, float& scale) {
  const vec3 light_vector  = sub_vector(light_point, initial_vertex);
  const float target_scale = get_length(light_vector);

  ////////////////////////////////////////////////////////////////////
  // Sample vertex count
  ////////////////////////////////////////////////////////////////////

  float vertex_count_pdf;
  const uint32_t vertex_count = bridges_sample_vertex_count(volume, target_scale, seed, pixel, vertex_count_pdf);

  ////////////////////////////////////////////////////////////////////
  // Sample path
  ////////////////////////////////////////////////////////////////////

  vec3 current_vertex    = initial_vertex;
  vec3 current_direction = normalize_vector(light_vector);

  float sum_dist = 0.0f;

  {
    const float random_dist = quasirandom_sequence_1D(QUASI_RANDOM_TARGET_BRIDGE_DISTANCE + seed * 32 + 0, pixel);
    const float dist        = -logf(random_dist);
    current_vertex          = add_vector(current_vertex, scale_vector(current_direction, dist));

    sum_dist += dist;
  }

  for (uint32_t i = 1; i < vertex_count; i++) {
    const float2 random_phase = quasirandom_sequence_2D(QUASI_RANDOM_TARGET_BRIDGE_PHASE + seed * 32 + i, pixel);

    current_direction = bridges_phase_sample(current_direction, random_phase);

    const float random_dist = quasirandom_sequence_1D(QUASI_RANDOM_TARGET_BRIDGE_DISTANCE + seed * 32 + i, pixel);
    const float dist        = -logf(random_dist);
    current_vertex          = add_vector(current_vertex, scale_vector(current_direction, dist));

    sum_dist += dist;
  }

  ////////////////////////////////////////////////////////////////////
  // Compute PDF and approximate visibility of path
  ////////////////////////////////////////////////////////////////////

  const float actual_scale = get_length(sub_vector(current_vertex, initial_vertex));

  if (actual_scale == 0.0f) {
    path_pdf = 0.0f;

    return get_color(0.0f, 0.0f, 0.0f);
  }

  scale = target_scale / actual_scale;

  sum_dist *= scale;

  end_vertex = current_vertex;

  RGBF path_weight = get_color(
    expf(vertex_count * logf(volume.scattering.r) - sum_dist * (volume.scattering.r + volume.absorption.r)),
    expf(vertex_count * logf(volume.scattering.g) - sum_dist * (volume.scattering.g + volume.absorption.g)),
    expf(vertex_count * logf(volume.scattering.b) - sum_dist * (volume.scattering.b + volume.absorption.b)));

  const float log_path_pdf = bridges_log_factorial(vertex_count) - vertex_count * logf(sum_dist);

  path_pdf = vertex_count_pdf * expf(log_path_pdf) * target_scale * target_scale * target_scale;

  return path_weight;
}

// TODO: Check if I can maybe get rid of all these duplicate computations and just reuse what I computed in the RIS step.
__device__ RGBF bridges_evaluate_bridge(
  const DeviceTask task, const VolumeDescriptor volume, const TriangleHandle light_handle, const uint32_t seed, const Quaternion rotation,
  const float scale) {
  bool bridge_is_valid = true;
  bridge_is_valid &= light_handle.instance_id != LIGHT_ID_NONE;
  bridge_is_valid &= seed != 0xFFFFFFFF;

  ////////////////////////////////////////////////////////////////////
  // Get light sample
  ////////////////////////////////////////////////////////////////////

  OptixTraceStatus trace_status = (bridge_is_valid) ? OPTIX_TRACE_STATUS_EXECUTE : OPTIX_TRACE_STATUS_ABORT;
  RGBF light_color;
  vec3 light_vector;
  if (bridge_is_valid) {
    const DeviceTransform light_transform = load_transform(light_handle.instance_id);

    uint3 light_packed_uv;
    TriangleLight light = light_load_sample_init(light_handle, light_transform, light_packed_uv);

    const float2 random_light_point = quasirandom_sequence_2D(QUASI_RANDOM_TARGET_BRIDGE_LIGHT_POINT + seed, task.index);

    vec3 light_dir;
    float light_dist, area;
    light_load_sample_finalize_bridges(light, light_packed_uv, task.origin, random_light_point, light_dir, light_dist, area);

    light_color  = light_get_color(light);
    light_vector = scale_vector(light_dir, light_dist);
  }
  else {
    light_color  = get_color(0.0f, 0.0f, 0.0f);
    light_vector = get_vector(0.0f, 0.0f, 1.0f);
  }

  ////////////////////////////////////////////////////////////////////
  // Get sampled vertex count
  ////////////////////////////////////////////////////////////////////

  uint32_t vertex_count;
  if (bridge_is_valid) {
    float vertex_count_pdf;
    vertex_count = bridges_sample_vertex_count(volume, get_length(light_vector), seed, task.index, vertex_count_pdf);
  }
  else {
    vertex_count = 1;
  }

  ////////////////////////////////////////////////////////////////////
  // Compute visibility of path
  ////////////////////////////////////////////////////////////////////

  vec3 current_vertex            = task.origin;
  vec3 current_direction_sampled = normalize_vector(light_vector);
  vec3 current_direction         = quaternion_apply(rotation, current_direction_sampled);

  // Apply phase function of first direction.
  const float cos_angle = dot_product(current_direction, task.ray);

  const float phase_function_weight = bridges_phase_function(cos_angle);

  light_color = scale_color(light_color, phase_function_weight);

  float sum_dist = 0.0f;

  float dist = -logf(quasirandom_sequence_1D(QUASI_RANDOM_TARGET_BRIDGE_DISTANCE + seed * 32 + 0, task.index)) * scale;

  light_color = mul_color(light_color, optix_geometry_shadowing(current_vertex, current_direction, dist, light_handle, trace_status));

  sum_dist += dist;

  for (int i = 1; i < vertex_count; i++) {
    current_vertex = add_vector(current_vertex, scale_vector(current_direction, dist));

    const float2 random_phase = quasirandom_sequence_2D(QUASI_RANDOM_TARGET_BRIDGE_PHASE + seed * 32 + i, task.index);

    current_direction_sampled = bridges_phase_sample(current_direction_sampled, random_phase);

    current_direction = quaternion_apply(rotation, current_direction_sampled);

    dist = -logf(quasirandom_sequence_1D(QUASI_RANDOM_TARGET_BRIDGE_DISTANCE + seed * 32 + i, task.index)) * scale;

    light_color = mul_color(light_color, optix_geometry_shadowing(current_vertex, current_direction, dist, light_handle, trace_status));

    sum_dist += dist;
  }

  light_color.r *= expf(vertex_count * logf(volume.scattering.r) - sum_dist * (volume.scattering.r + volume.absorption.r));
  light_color.g *= expf(vertex_count * logf(volume.scattering.g) - sum_dist * (volume.scattering.g + volume.absorption.g));
  light_color.b *= expf(vertex_count * logf(volume.scattering.b) - sum_dist * (volume.scattering.b + volume.absorption.b));

  return light_color;
}

__device__ RGBF bridges_sample(const DeviceTask task, const VolumeDescriptor volume) {
  if (LIGHTS_ARE_PRESENT == false)
    return splat_color(0.0f);

  uint32_t selected_seed         = 0xFFFFFFFF;
  TriangleHandle selected_handle = triangle_handle_get(LIGHT_ID_NONE, 0);
  Quaternion selected_rotation   = {0.0f, 0.0f, 0.0f, 1.0f};
  float selected_scale           = 0.0f;
  float selected_target_pdf      = FLT_MAX;

  float sum_weight = 0.0f;

  float random_resampling = quasirandom_sequence_1D(QUASI_RANDOM_TARGET_BRIDGE_RESAMPLING, task.index);

  for (uint32_t i = 0; i < device.settings.bridge_num_ris_samples; i++) {
    float sample_pdf = 1.0f;

    ////////////////////////////////////////////////////////////////////
    // Sample light
    ////////////////////////////////////////////////////////////////////

    const float random_light_tree = quasirandom_sequence_1D(QUASI_RANDOM_TARGET_BRIDGE_LIGHT_TREE + i, task.index);

    float light_list_pdf;
    DeviceTransform light_transform;
    const TriangleHandle light_handle = light_tree_query(volume, task.origin, task.ray, random_light_tree, light_list_pdf, light_transform);

    sample_pdf *= light_list_pdf;

    uint3 light_packed_uv;
    TriangleLight light = light_load_sample_init(light_handle, light_transform, light_packed_uv);

    ////////////////////////////////////////////////////////////////////
    // Sample light point
    ////////////////////////////////////////////////////////////////////

    const float2 random_light_point = quasirandom_sequence_2D(QUASI_RANDOM_TARGET_BRIDGE_LIGHT_POINT + i, task.index);

    vec3 light_dir;
    float area, light_dist;
    light_load_sample_finalize_bridges(light, light_packed_uv, task.origin, random_light_point, light_dir, light_dist, area);

    if (light_dist == FLT_MAX || area < eps)
      continue;

    RGBF light_color = light_get_color(light);

    // We sampled a point that emits no light, skip.
    if (color_importance(light_color) == 0.0f)
      continue;

    const vec3 light_point = add_vector(task.origin, scale_vector(light_dir, light_dist));

    if (light_point.y < volume.min_height || light_point.y > volume.max_height)
      continue;

    sample_pdf *= 1.0f / area;

    ////////////////////////////////////////////////////////////////////
    // Sample path
    ////////////////////////////////////////////////////////////////////

    float sample_path_pdf;
    float sample_path_scale;
    vec3 sample_path_end_vertex;
    RGBF sample_path_weight =
      bridges_sample_bridge(task.origin, volume, light_point, i, task.index, sample_path_pdf, sample_path_end_vertex, sample_path_scale);

    if (sample_path_pdf == 0.0f)
      continue;

    sample_path_weight = mul_color(sample_path_weight, light_color);

    sample_pdf *= sample_path_pdf;

    ////////////////////////////////////////////////////////////////////
    // Modify path
    ////////////////////////////////////////////////////////////////////

    const Quaternion sample_rotation = bridges_compute_rotation(task.origin, light_point, sample_path_end_vertex);

    const vec3 rotation_initial_direction = quaternion_apply(sample_rotation, light_dir);
    const float cos_angle                 = dot_product(rotation_initial_direction, task.ray);

    const float phase_function_weight = bridges_phase_function(cos_angle);

    sample_path_weight = scale_color(sample_path_weight, phase_function_weight);

    ////////////////////////////////////////////////////////////////////
    // Resample
    ////////////////////////////////////////////////////////////////////

    const float target_pdf = color_importance(sample_path_weight);

    if (target_pdf == 0.0f)
      continue;

    const float weight = (sample_pdf > 0.0f) ? target_pdf / sample_pdf : 0.0f;

    sum_weight += weight;

    const float resampling_probability = weight / sum_weight;

    if (random_resampling < resampling_probability) {
      selected_seed       = i;
      selected_handle     = light_handle;
      selected_rotation   = sample_rotation;
      selected_scale      = sample_path_scale;
      selected_target_pdf = target_pdf;

      random_resampling = random_resampling / resampling_probability;
    }
    else {
      random_resampling = (random_resampling - resampling_probability) / (1.0f - resampling_probability);
    }
  }

  sum_weight /= device.settings.bridge_num_ris_samples;

  ////////////////////////////////////////////////////////////////////
  // Evaluate sampled path
  ////////////////////////////////////////////////////////////////////

  RGBF bridge_color = bridges_evaluate_bridge(task, volume, selected_handle, selected_seed, selected_rotation, selected_scale);

  bridge_color = (selected_target_pdf > 0.0f) ? scale_color(bridge_color, sum_weight / selected_target_pdf) : splat_color(0.0f);

  UTILS_CHECK_NANS(task.index, bridge_color);

  return bridge_color;
}

#endif /* OPTIX_KERNEL */

#endif /* CU_BRIDGES_H */

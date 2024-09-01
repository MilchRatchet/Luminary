#ifndef CU_BRIDGES_H
#define CU_BRIDGES_H

#if defined(OPTIX_KERNEL) && defined(VOLUME_KERNEL)

#include "light.cuh"
#include "math.cuh"
#include "memory.cuh"
#include "ocean_utils.cuh"
#include "optix_shadow_trace.cuh"
#include "random.cuh"
#include "utils.cuh"
#include "utils.h"
#include "volume_utils.cuh"

#define BRIDGES_MAX_DEPTH 32

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

  const float low_dist  = (effective_dist < center_dist) ? min_dist : center_dist;
  const float high_dist = (effective_dist < center_dist) ? center_dist : max_dist;

  const float step       = (high_dist - low_dist) * 0.25f;
  const uint32_t step_id = (uint32_t) ((effective_dist - low_dist) / step);
  const float floor_dist = low_dist + step_id * step;
  const uint32_t index   = (effective_dist < center_dist) ? (3 + 2 * step_id) : (3 + 2 * (step_id + 4));

  const float y0  = __ldg(device.bridge_lut + lut_offset + index + 0);
  const float dy0 = __ldg(device.bridge_lut + lut_offset + index + 1);
  const float y1  = __ldg(device.bridge_lut + lut_offset + index + 2);
  const float dy1 = __ldg(device.bridge_lut + lut_offset + index + 3);

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

  for (uint32_t i = 0; i < device.bridge_settings.max_num_vertices; i++) {
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
  uint32_t selected_vertex_count = device.bridge_settings.max_num_vertices - 1;
  pdf                            = 1.0f;

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
    const float random_method = quasirandom_sequence_1D(QUASI_RANDOM_TARGET_BRIDGE_PHASE_METHOD + seed * 32 + i, pixel);

    if (volume.type == VOLUME_TYPE_FOG) {
      current_direction = jendersie_eon_phase_sample(current_direction, device.scene.fog.droplet_diameter, random_phase, random_method);
    }
    else {
      current_direction = ocean_phase_sampling(current_direction, random_phase, random_method);
    }

    const float random_dist = quasirandom_sequence_1D(QUASI_RANDOM_TARGET_BRIDGE_DISTANCE + seed * 32 + i, pixel);
    const float dist        = -logf(random_dist);
    current_vertex          = add_vector(current_vertex, scale_vector(current_direction, dist));

    sum_dist += dist;
  }

  ////////////////////////////////////////////////////////////////////
  // Compute PDF and approximate visibility of path
  ////////////////////////////////////////////////////////////////////

  const float actual_scale = get_length(sub_vector(current_vertex, initial_vertex));
  scale                    = target_scale / actual_scale;

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

__device__ RGBF bridges_evaluate_bridge(
  const TraceTask task, const VolumeDescriptor volume, const vec3 initial_vertex, const uint32_t light_id, const uint32_t seed,
  const Quaternion rotation, const float scale, const RGBF initial_vertex_transmittance, const float ior, const ushort2 pixel) {
  ////////////////////////////////////////////////////////////////////
  // Get light sample
  ////////////////////////////////////////////////////////////////////

  RGBF light_color;
  vec3 light_vector;
  if (light_id != LIGHT_ID_NONE) {
    const TriangleLight light = load_triangle_light(device.scene.triangle_lights, light_id);

    const float2 random_light_point = quasirandom_sequence_2D(QUASI_RANDOM_TARGET_BRIDGE_LIGHT_POINT + seed, pixel);

    float solid_angle;
    float light_dist;
    const vec3 light_dir = light_sample_triangle(light, initial_vertex, random_light_point, solid_angle, light_dist, light_color);

    const vec3 light_point = add_vector(initial_vertex, scale_vector(light_dir, light_dist));

    light_vector = sub_vector(light_point, initial_vertex);
  }
  else {
    light_color  = get_color(0.0f, 0.0f, 0.0f);
    light_vector = get_vector(0.0f, 0.0f, 1.0f);
  }

  light_color = mul_color(light_color, initial_vertex_transmittance);

  ////////////////////////////////////////////////////////////////////
  // Get sampled vertex count
  ////////////////////////////////////////////////////////////////////

  uint32_t vertex_count;
  if (light_id != LIGHT_ID_NONE) {
    float vertex_count_pdf;
    vertex_count = bridges_sample_vertex_count(volume, get_length(light_vector), seed, pixel, vertex_count_pdf);
  }
  else {
    vertex_count = 1;
  }

  ////////////////////////////////////////////////////////////////////
  // Compute visibility of path
  ////////////////////////////////////////////////////////////////////

  vec3 current_vertex            = initial_vertex;
  vec3 current_direction_sampled = normalize_vector(light_vector);
  vec3 current_direction         = rotate_vector_by_quaternion(current_direction_sampled, rotation);

  unsigned int compressed_ior = ior_compress(ior);

  // Apply phase function of first direction.
  const float cos_angle           = dot_product(current_direction, task.ray);
  const JendersieEonParams params = jendersie_eon_phase_parameters(device.scene.fog.droplet_diameter);

  float phase_function_weight;
  if (volume.type == VOLUME_TYPE_FOG) {
    phase_function_weight = jendersie_eon_phase_function(cos_angle, params);
  }
  else {
    phase_function_weight = ocean_phase(cos_angle);
  }

  light_color = scale_color(light_color, phase_function_weight);

  float sum_dist = 0.0f;

  float dist = -logf(quasirandom_sequence_1D(QUASI_RANDOM_TARGET_BRIDGE_DISTANCE + seed * 32 + 0, pixel)) * scale;

  light_color = mul_color(light_color, optix_geometry_shadowing(current_vertex, current_direction, dist, light_id, pixel, compressed_ior));
  light_color = mul_color(light_color, optix_toy_shadowing(current_vertex, current_direction, dist, compressed_ior));

  sum_dist += dist;

  for (int i = 1; i < vertex_count; i++) {
    current_vertex = add_vector(current_vertex, scale_vector(current_direction, dist));

    const float2 random_phase = quasirandom_sequence_2D(QUASI_RANDOM_TARGET_BRIDGE_PHASE + seed * 32 + i, pixel);
    const float random_method = quasirandom_sequence_1D(QUASI_RANDOM_TARGET_BRIDGE_PHASE_METHOD + seed * 32 + i, pixel);

    if (volume.type == VOLUME_TYPE_FOG) {
      current_direction_sampled =
        jendersie_eon_phase_sample(current_direction_sampled, device.scene.fog.droplet_diameter, random_phase, random_method);
    }
    else {
      current_direction_sampled = ocean_phase_sampling(current_direction_sampled, random_phase, random_method);
    }

    vec3 current_direction = rotate_vector_by_quaternion(current_direction_sampled, rotation);

    dist = -logf(quasirandom_sequence_1D(QUASI_RANDOM_TARGET_BRIDGE_DISTANCE + seed * 32 + i, pixel)) * scale;

    light_color =
      mul_color(light_color, optix_geometry_shadowing(current_vertex, current_direction, dist, light_id, pixel, compressed_ior));
    light_color = mul_color(light_color, optix_toy_shadowing(current_vertex, current_direction, dist, compressed_ior));

    sum_dist += dist;
  }

  light_color.r *= expf(vertex_count * logf(volume.scattering.r) - sum_dist * (volume.scattering.r + volume.absorption.r));
  light_color.g *= expf(vertex_count * logf(volume.scattering.g) - sum_dist * (volume.scattering.g + volume.absorption.g));
  light_color.b *= expf(vertex_count * logf(volume.scattering.b) - sum_dist * (volume.scattering.b + volume.absorption.b));

  return light_color;
}

__device__ float bridges_sample_initial_vertex_target_pdf(
  const TraceTask task, const VolumeDescriptor volume, const TriangleLight light, const vec3 light_center, const JendersieEonParams params,
  const float t) {
  RGBF weight;

  weight.r = expf(-t * (volume.scattering.r + volume.absorption.r));
  weight.g = expf(-t * (volume.scattering.g + volume.absorption.g));
  weight.b = expf(-t * (volume.scattering.b + volume.absorption.b));

  weight = mul_color(weight, volume.scattering);

  const vec3 sample_point = add_vector(task.origin, scale_vector(task.ray, t));

  const float solid_angle = sample_triangle_solid_angle(light, sample_point);

  weight = scale_color(weight, solid_angle);

  const float cos_angle = dot_product(task.ray, normalize_vector(sub_vector(light_center, sample_point)));

  float phase_function_weight;
  if (volume.type == VOLUME_TYPE_FOG) {
    phase_function_weight = jendersie_eon_phase_function(cos_angle, params);
  }
  else {
    phase_function_weight = ocean_phase(cos_angle);
  }

  weight = scale_color(weight, phase_function_weight);

  return color_importance(weight);
}

__device__ float bridges_initial_vertex_uniform_sample(const float uniform_min, const float uniform_max, const float random) {
  return uniform_min + random * (uniform_max - uniform_min);
}

__device__ float bridges_initial_vertex_uniform_sample_pdf(const float uniform_min, const float uniform_max, const float t) {
  if (t > uniform_max || t < uniform_min)
    return 0.0f;

  return 1.0f / (uniform_max - uniform_min);
}

__device__ float2 bridges_initial_vertex_uniform_sample_bounds(
  const TraceTask task, const float limit, const TriangleLight light, const vec3 light_center, const ushort2 pixel) {
  const float light_center_dist = dot_product(sub_vector(light_center, task.origin), task.ray);
  const vec3 p                  = add_vector(task.origin, scale_vector(task.ray, light_center_dist));
  const float radius            = get_length(sub_vector(light_center, p)) * 3.0f;

  const float t0 = sphere_ray_intersection(task.ray, task.origin, light_center, radius);
  const float t1 = t0 + 2.0f * (light_center_dist - t0);

  if (t0 == FLT_MAX) {
    return make_float2(-FLT_MAX, -FLT_MAX * 0.5f);
  }

  const float uniform_min = fmaxf(fminf(fminf(t0, t1), limit), 0.0f);
  const float uniform_max = fmaxf(fminf(fmaxf(t0, t1), limit), 0.0f);

  return make_float2(uniform_min, uniform_max);
}

__device__ vec3 bridges_sample_initial_vertex(
  const TraceTask task, const VolumeDescriptor volume, const float limit, const TriangleLight light, const uint32_t seed,
  const ushort2 pixel, float& pdf, RGBF& transmittance) {
  pdf           = 1.0f;
  transmittance = get_color(0.0f, 0.0f, 0.0f);

  float selected_target_pdf = 1.0f;
  float selected_t          = -FLT_MAX;

  const JendersieEonParams params = jendersie_eon_phase_parameters(device.scene.fog.droplet_diameter);
  const vec3 light_center         = add_vector(light.vertex, scale_vector(add_vector(light.edge1, light.edge2), 0.33f));

  const float2 uniform_bounds = bridges_initial_vertex_uniform_sample_bounds(task, limit, light, light_center, pixel);
  const float uniform_min     = uniform_bounds.x;
  const float uniform_max     = uniform_bounds.y;

  float sum_weight = 0.0f;

  const uint32_t uniform_sample_count  = 32;
  const uint32_t distance_sample_count = 4;
  // TODO: Try out sampling technique from "Practical product sampling for single scattering in media" by Villeneuve et al.

  float random_resampling = quasirandom_sequence_1D(QUASI_RANDOM_TARGET_BRIDGE_VERTEX_RESAMPLING + seed, task.index);

  uint32_t sample_id = 0;

  for (uint32_t i = 0; i < uniform_sample_count; i++) {
    const float sample_random = quasirandom_sequence_1D(QUASI_RANDOM_TARGET_BRIDGE_VERTEX_DISTANCE + seed * 32 + sample_id, task.index);
    sample_id++;

    const float t = bridges_initial_vertex_uniform_sample(uniform_min, uniform_max, sample_random);

    const float uniform_sample_pdf  = bridges_initial_vertex_uniform_sample_pdf(uniform_min, uniform_max, t);
    const float distance_sample_pdf = volume_sample_intersection_pdf(volume, task.origin, task.ray, 0.0f, t);

    const float target_pdf = bridges_sample_initial_vertex_target_pdf(task, volume, light, light_center, params, t);

    const float weight = target_pdf / (distance_sample_pdf * distance_sample_count + uniform_sample_pdf * uniform_sample_count);

    sum_weight += weight;

    const float resampling_probability = weight / sum_weight;

    if (random_resampling < resampling_probability) {
      selected_t          = t;
      selected_target_pdf = target_pdf;

      random_resampling = random_resampling / resampling_probability;
    }
    else {
      random_resampling = (random_resampling - resampling_probability) / (1.0f - resampling_probability);
    }
  }

  for (uint32_t i = 0; i < distance_sample_count; i++) {
    const float sample_random = quasirandom_sequence_1D(QUASI_RANDOM_TARGET_BRIDGE_VERTEX_DISTANCE + seed * 32 + sample_id, task.index);
    sample_id++;

    const float t = volume_sample_intersection(volume, task.origin, task.ray, 0.0f, limit, sample_random);

    if (t == FLT_MAX)
      continue;

    const float uniform_sample_pdf  = bridges_initial_vertex_uniform_sample_pdf(uniform_min, uniform_max, t);
    const float distance_sample_pdf = volume_sample_intersection_pdf(volume, task.origin, task.ray, 0.0f, t);

    const float target_pdf = bridges_sample_initial_vertex_target_pdf(task, volume, light, light_center, params, t);

    const float weight = target_pdf / (distance_sample_pdf * distance_sample_count + uniform_sample_pdf * uniform_sample_count);

    sum_weight += weight;

    const float resampling_probability = weight / sum_weight;

    if (random_resampling < resampling_probability) {
      selected_t          = t;
      selected_target_pdf = target_pdf;

      random_resampling = random_resampling / resampling_probability;
    }
    else {
      random_resampling = (random_resampling - resampling_probability) / (1.0f - resampling_probability);
    }
  }

  pdf = (sum_weight > 0.0f) ? selected_target_pdf / sum_weight : FLT_MAX;

  transmittance.r = expf(-selected_t * (volume.scattering.r + volume.absorption.r));
  transmittance.g = expf(-selected_t * (volume.scattering.g + volume.absorption.g));
  transmittance.b = expf(-selected_t * (volume.scattering.b + volume.absorption.b));

  return add_vector(task.origin, scale_vector(task.ray, selected_t));
}

__device__ RGBF bridges_sample(const TraceTask task, const VolumeDescriptor volume, const float limit, const float ior) {
  uint32_t selected_seed                     = 0xFFFFFFFF;
  uint32_t selected_light_id                 = LIGHT_ID_NONE;
  Quaternion selected_rotation               = {0.0f, 0.0f, 0.0f, 1.0f};
  vec3 selected_initial_vertex               = get_vector(0.0f, 0.0f, 0.0f);
  RGBF selected_initial_vertex_transmittance = get_color(0.0f, 0.0f, 0.0f);
  float selected_scale                       = 0.0f;
  float selected_target_pdf                  = FLT_MAX;

  const JendersieEonParams params = jendersie_eon_phase_parameters(device.scene.fog.droplet_diameter);

  float sum_weight = 0.0f;

  float random_resampling = quasirandom_sequence_1D(QUASI_RANDOM_TARGET_BRIDGE_RESAMPLING, task.index);

  for (uint32_t i = 0; i < device.bridge_settings.num_ris_samples; i++) {
    float sample_pdf = 1.0f;

    ////////////////////////////////////////////////////////////////////
    // Sample light
    ////////////////////////////////////////////////////////////////////

    const float random_light_tree = quasirandom_sequence_1D(QUASI_RANDOM_TARGET_BRIDGE_LIGHT_TREE + i, task.index);

    uint32_t light_list_length;
    float light_list_pdf;
    const uint32_t light_list_ptr =
      light_tree_traverse(volume, task.origin, task.ray, limit, random_light_tree, light_list_length, light_list_pdf);

    sample_pdf *= light_list_pdf;

    const float random_light_list = quasirandom_sequence_1D(QUASI_RANDOM_TARGET_BRIDGE_LIGHT_LIST + i, task.index);

    const uint32_t light_id = uint32_t(__fmul_rd(random_light_list, light_list_length)) + light_list_ptr;

    sample_pdf *= 1.0f / light_list_length;

    const TriangleLight light = load_triangle_light(device.scene.triangle_lights, light_id);

    ////////////////////////////////////////////////////////////////////
    // Sample initial vertex
    ////////////////////////////////////////////////////////////////////

    float initial_vertex_pdf;
    RGBF sample_initial_vertex_transmittance;
    const vec3 sample_initial_vertex =
      bridges_sample_initial_vertex(task, volume, limit, light, i, task.index, initial_vertex_pdf, sample_initial_vertex_transmittance);

    sample_pdf *= initial_vertex_pdf;

    ////////////////////////////////////////////////////////////////////
    // Sample light point
    ////////////////////////////////////////////////////////////////////

    const float2 random_light_point = quasirandom_sequence_2D(QUASI_RANDOM_TARGET_BRIDGE_LIGHT_POINT + i, task.index);

    RGBF light_color;
    float solid_angle;
    float light_dist;
    const vec3 light_dir = light_sample_triangle(light, sample_initial_vertex, random_light_point, solid_angle, light_dist, light_color);

    // We sampled a point that emits no light, skip.
    if (color_importance(light_color) == 0.0f || solid_angle < eps)
      continue;

    const vec3 light_point = add_vector(sample_initial_vertex, scale_vector(light_dir, light_dist));

    if (light_point.y < volume.min_height || light_point.y > volume.max_height)
      continue;

    sample_pdf *= 1.0f / solid_angle;

    ////////////////////////////////////////////////////////////////////
    // Sample path
    ////////////////////////////////////////////////////////////////////

    float sample_path_pdf;
    float sample_path_scale;
    vec3 sample_path_end_vertex;
    RGBF sample_path_weight = bridges_sample_bridge(
      sample_initial_vertex, volume, light_point, i, task.index, sample_path_pdf, sample_path_end_vertex, sample_path_scale);

    sample_pdf *= sample_path_pdf;

    ////////////////////////////////////////////////////////////////////
    // Modify path
    ////////////////////////////////////////////////////////////////////

    const Quaternion sample_rotation = bridges_compute_rotation(sample_initial_vertex, light_point, sample_path_end_vertex);

    const vec3 rotation_initial_direction = rotate_vector_by_quaternion(light_dir, sample_rotation);
    const float cos_angle                 = dot_product(rotation_initial_direction, task.ray);

    float phase_function_weight;
    if (volume.type == VOLUME_TYPE_FOG) {
      phase_function_weight = jendersie_eon_phase_function(cos_angle, params);
    }
    else {
      phase_function_weight = ocean_phase(cos_angle);
    }

    sample_path_weight = scale_color(sample_path_weight, phase_function_weight);
    sample_path_weight = mul_color(sample_path_weight, sample_initial_vertex_transmittance);

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
      selected_seed                         = i;
      selected_light_id                     = light_id;
      selected_rotation                     = sample_rotation;
      selected_scale                        = sample_path_scale;
      selected_target_pdf                   = target_pdf;
      selected_initial_vertex               = sample_initial_vertex;
      selected_initial_vertex_transmittance = sample_initial_vertex_transmittance;

      random_resampling = random_resampling / resampling_probability;
    }
    else {
      random_resampling = (random_resampling - resampling_probability) / (1.0f - resampling_probability);
    }
  }

  sum_weight /= device.bridge_settings.num_ris_samples;

  ////////////////////////////////////////////////////////////////////
  // Evaluate sampled path
  ////////////////////////////////////////////////////////////////////

  RGBF bridge_color = bridges_evaluate_bridge(
    task, volume, selected_initial_vertex, selected_light_id, selected_seed, selected_rotation, selected_scale,
    selected_initial_vertex_transmittance, ior, task.index);

  bridge_color = scale_color(bridge_color, sum_weight / selected_target_pdf);

  return bridge_color;
}

#endif /* OPTIX_KERNEL */

#endif /* CU_BRIDGES_H */

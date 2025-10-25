#ifndef CU_LUMINARY_LIGHT_BRIDGES_H
#define CU_LUMINARY_LIGHT_BRIDGES_H

#include "light.cuh"
#include "light_common.cuh"
#include "light_triangle.cuh"
#include "math.cuh"
#include "memory.cuh"
#include "ocean_utils.cuh"
#include "random.cuh"
#include "ris.cuh"
#include "utils.cuh"
#include "utils.h"
#include "volume_utils.cuh"

LUMINARY_FUNCTION Quaternion bridges_compute_rotation(const vec3 initial_vertex, const vec3 light_point, const vec3 end_vertex) {
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

LUMINARY_FUNCTION float bridges_log_factorial(const uint32_t vertex_count) {
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
LUMINARY_FUNCTION float bridges_get_vertex_count_importance(const uint32_t vertex_count, const float effective_dist) {
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

LUMINARY_FUNCTION uint32_t
  bridges_sample_vertex_count(const VolumeDescriptor volume, const float light_dist, const uint32_t seed, const ushort2 pixel, float& pdf) {
  const float effective_dist = light_dist * volume.max_scattering;

  const float random         = random_1D(RANDOM_TARGET_LIGHT_GEO_BRIDGE_VERTEX_COUNT + seed, pixel);
  RISReservoir ris_reservoir = ris_reservoir_init(random);

  ////////////////////////////////////////////////////////////////////
  // Sample a vertex count based on importance LUT
  ////////////////////////////////////////////////////////////////////

  const uint32_t max_num_vertices = min(device.settings.bridge_max_num_vertices, BRIDGES_MAX_VERTEX_COUNT);

  // Fallback values. These come into play if sum_importance is 0.0f.
  // This happens when effective dist is too large.
  uint32_t selected_vertex_count = max_num_vertices - 1;

#pragma unroll
  for (uint32_t vertex_count = 0; vertex_count < max_num_vertices; vertex_count++) {
    // TODO: The paper uses some additional terms here for the importance
    const float importance = bridges_get_vertex_count_importance(vertex_count + 1, effective_dist);

    if (ris_reservoir_add_sample(ris_reservoir, importance, 1.0f)) {
      selected_vertex_count = vertex_count;
    }
  }

  pdf = ris_reservoir_get_sampling_prob(ris_reservoir);

  return 1 + selected_vertex_count;
}

LUMINARY_FUNCTION RGBF bridges_sample_bridge(
  MaterialContextVolume ctx, const vec3 light_point, const vec3 initial_vertex, const uint32_t seed, const ushort2 pixel, float& path_pdf,
  vec3& end_vertex, float& scale) {
  const vec3 light_vector  = sub_vector(light_point, initial_vertex);
  const float target_scale = get_length(light_vector);

  ////////////////////////////////////////////////////////////////////
  // Sample vertex count
  ////////////////////////////////////////////////////////////////////

  float vertex_count_pdf;
  const uint32_t vertex_count = bridges_sample_vertex_count(ctx.descriptor, target_scale, seed, pixel, vertex_count_pdf);

  ////////////////////////////////////////////////////////////////////
  // Sample path
  ////////////////////////////////////////////////////////////////////

  vec3 current_vertex    = initial_vertex;
  vec3 current_direction = normalize_vector(light_vector);

  float sum_dist = 0.0f;

  {
    const float random_dist = random_1D(RANDOM_TARGET_LIGHT_GEO_BRIDGE_DISTANCE + seed * LIGHT_GEO_MAX_BRIDGE_LENGTH + 0, pixel);
    const float dist        = -logf(random_dist);
    current_vertex          = add_vector(current_vertex, scale_vector(current_direction, dist));

    sum_dist += dist;
  }

  LUMINARY_ASSUME(vertex_count <= BRIDGES_MAX_VERTEX_COUNT);

  for (uint32_t i = 1; i < vertex_count; i++) {
    const float2 random_phase = random_2D(RANDOM_TARGET_LIGHT_GEO_BRIDGE_PHASE + seed * LIGHT_GEO_MAX_BRIDGE_LENGTH + i, pixel);

    current_direction = bridges_phase_sample(current_direction, random_phase);

    const float random_dist = random_1D(RANDOM_TARGET_LIGHT_GEO_BRIDGE_DISTANCE + seed * LIGHT_GEO_MAX_BRIDGE_LENGTH + i, pixel);
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

  const RGBF scattering = ctx.descriptor.scattering;
  const RGBF absorption = ctx.descriptor.absorption;

  RGBF path_weight = get_color(
    expf(vertex_count * logf(scattering.r) - sum_dist * (scattering.r + absorption.r)),
    expf(vertex_count * logf(scattering.g) - sum_dist * (scattering.g + absorption.g)),
    expf(vertex_count * logf(scattering.b) - sum_dist * (scattering.b + absorption.b)));

  const float log_path_pdf = bridges_log_factorial(vertex_count) - vertex_count * logf(sum_dist);

  path_pdf = vertex_count_pdf * expf(log_path_pdf) * target_scale * target_scale * target_scale;

  return path_weight;
}

LUMINARY_FUNCTION vec3 bridges_sample_initial_vertex(
  MaterialContextVolume ctx, const vec3 point_on_light, const ushort2 pixel, const uint32_t output_id, RGBF& attenuation, float& pdf) {
  float random_intersection = random_1D(RANDOM_TARGET_LIGHT_GEO_INITIAL_VERTEX + output_id, pixel);

  const vec3 PO             = sub_vector(point_on_light, ctx.position);
  const float dist_to_light = fmaxf(-dot_product(PO, ctx.V), 0.0f);

  // We sample points in front of the light with a very high probability, the rest is only for unbiasedness.
  const float forward_prob = (dist_to_light < ctx.max_dist) ? BRIDGES_INITIAL_VERTEX_FORWARD_PROB : 1.0f;

  float max_dist;
  float t_offset;

  if (random_intersection < forward_prob) {
    random_intersection = random_intersection / forward_prob;

    max_dist = clampf(dist_to_light, 0.0f, ctx.max_dist);
    t_offset = 0.0f;

    pdf = forward_prob;
  }
  else {
    random_intersection = (random_intersection - forward_prob) / (1.0f - forward_prob);

    // This path is only ever hit if ctx.max_dist > dist_to_light;
    max_dist = ctx.max_dist - dist_to_light;
    t_offset = dist_to_light;

    pdf = 1.0f - forward_prob;
  }

  const float t = t_offset + volume_sample_intersection_bounded(ctx.descriptor, max_dist, random_intersection);

  const RGBF volume_transmittance = volume_get_transmittance(ctx.descriptor);

  attenuation.r = expf(-t * volume_transmittance.r) * ctx.descriptor.scattering.r;
  attenuation.g = expf(-t * volume_transmittance.g) * ctx.descriptor.scattering.g;
  attenuation.b = expf(-t * volume_transmittance.b) * ctx.descriptor.scattering.b;

  pdf *= volume_sample_intersection_bounded_pdf(ctx.descriptor, max_dist, t - t_offset);

  return add_vector(ctx.position, scale_vector(ctx.V, -t));
}

LUMINARY_FUNCTION LightSampleResult<MATERIAL_VOLUME> bridges_sample(
  MaterialContextVolume ctx, TriangleLight light, const uint32_t light_id, const uint3 light_uv_packed, const ushort2 pixel,
  const uint32_t output_id, float2& target_and_weight) {
  const float2 random_light_point = random_2D(RANDOM_TARGET_LIGHT_GEO_BRIDGE_LIGHT_POINT + output_id, pixel);

  const vec3 point_on_light = light_triangle_sample_bridges(light, random_light_point);

  RGBF initial_attenuation;
  float initial_pdf;
  const vec3 initial_vertex = bridges_sample_initial_vertex(ctx, point_on_light, pixel, output_id, initial_attenuation, initial_pdf);

  LightSampleResult<MATERIAL_VOLUME> result;
  result.light_id = LIGHT_ID_INVALID;

  if (initial_pdf == 0.0f || color_importance(initial_attenuation) == 0.0f)
    return result;

  vec3 light_dir;
  float area, light_dist;
  light_triangle_sample_finalize_bridges(light, light_uv_packed, initial_vertex, point_on_light, light_dir, light_dist, area);

  target_and_weight.x = 0.0f;
  target_and_weight.y = 1.0f;

  if (light_dist == FLT_MAX || area < eps)
    return result;

  RGBF light_color = mul_color(light_get_color(light), initial_attenuation);

  // We sampled a point that emits no light, skip.
  if (color_importance(light_color) == 0.0f)
    return result;

  const vec3 light_point = add_vector(initial_vertex, scale_vector(light_dir, light_dist));

  if (light_point.y < ctx.descriptor.min_height || light_point.y > ctx.descriptor.max_height)
    return result;

  float sample_weight = area / initial_pdf;

  ////////////////////////////////////////////////////////////////////
  // Sample path
  ////////////////////////////////////////////////////////////////////

  float sample_path_pdf;
  float sample_path_scale;
  vec3 sample_path_end_vertex;
  RGBF sample_path_weight =
    bridges_sample_bridge(ctx, light_point, initial_vertex, output_id, pixel, sample_path_pdf, sample_path_end_vertex, sample_path_scale);

  if (sample_path_pdf == 0.0f)
    return result;

  sample_weight *= 1.0f / sample_path_pdf;

  ////////////////////////////////////////////////////////////////////
  // Modify path
  ////////////////////////////////////////////////////////////////////

  const Quaternion sample_rotation = bridges_compute_rotation(initial_vertex, light_point, sample_path_end_vertex);

  const vec3 rotation_initial_direction = quaternion_apply(sample_rotation, light_dir);
  const float cos_angle                 = -dot_product(rotation_initial_direction, ctx.V);

  const float phase_function_weight = bridges_phase_function(cos_angle);
  light_color                       = scale_color(light_color, phase_function_weight);

  ////////////////////////////////////////////////////////////////////
  // Finalize sample
  ////////////////////////////////////////////////////////////////////

  sample_path_weight = mul_color(sample_path_weight, light_color);

  target_and_weight.x = color_importance(sample_path_weight);
  target_and_weight.y = sample_weight;

  result.light_id    = light_id;
  result.light_color = sample_path_weight;
  result.rotation    = sample_rotation;
  result.scale       = sample_path_scale;
  result.seed        = output_id;

  return result;
}

#ifdef OPTIX_KERNEL

#include "optix_include.cuh"

LUMINARY_FUNCTION RGBF bridges_sample_apply_shadowing(
  const MaterialContextVolume ctx, const DeviceTaskDirectLightBridges& direct_light_task, const ushort2 pixel, const bool sample_is_valid) {
  const uint32_t seed = direct_light_task.seed;

  bool bridge_is_valid = sample_is_valid;
  bridge_is_valid &= seed != 0xFFFFFFFF;

  ////////////////////////////////////////////////////////////////////
  // Get light sample
  ////////////////////////////////////////////////////////////////////

  OptixTraceStatus trace_status = (bridge_is_valid) ? OPTIX_TRACE_STATUS_EXECUTE : OPTIX_TRACE_STATUS_ABORT;
  vec3 light_vector             = get_vector(0.0f, 0.0f, 1.0f);
  vec3 initial_vertex           = get_vector(0.0f, 0.0f, 0.0f);
  TriangleHandle light_handle   = TRIANGLE_HANDLE_INVALID;

  if (bridge_is_valid) {
    light_handle = device.ptrs.light_tree_tri_handle_map[direct_light_task.light_id];

    const DeviceTransform light_transform = load_transform(light_handle.instance_id);

    uint3 light_uv_packed;
    TriangleLight light = light_triangle_sample_init(light_handle, light_transform, light_uv_packed);

    const float2 random_light_point = random_2D(RANDOM_TARGET_LIGHT_GEO_BRIDGE_LIGHT_POINT + seed, pixel);

    const vec3 point_on_light = light_triangle_sample_bridges(light, random_light_point);

    RGBF initial_attenuation;
    float initial_pdf;
    initial_vertex = bridges_sample_initial_vertex(ctx, point_on_light, pixel, seed, initial_attenuation, initial_pdf);

    vec3 light_dir;
    float area, light_dist;
    light_triangle_sample_finalize_bridges(light, light_uv_packed, initial_vertex, point_on_light, light_dir, light_dist, area);

    light_vector = scale_vector(light_dir, light_dist);
  }

  ////////////////////////////////////////////////////////////////////
  // Get sampled vertex count
  ////////////////////////////////////////////////////////////////////

  uint32_t vertex_count = 1;
  if (bridge_is_valid) {
    float vertex_count_pdf;
    vertex_count = bridges_sample_vertex_count(ctx.descriptor, get_length(light_vector), seed, pixel, vertex_count_pdf);
  }

  ////////////////////////////////////////////////////////////////////
  // Compute visibility of path
  ////////////////////////////////////////////////////////////////////

  const Quaternion16 rotation = direct_light_task.rotation;
  const float scale           = direct_light_task.scale;

  vec3 current_vertex            = initial_vertex;
  vec3 current_direction_sampled = normalize_vector(light_vector);
  vec3 current_direction         = quaternion16_apply(rotation, current_direction_sampled);

  // We don't need to trace visibility to the initial vertex as it was sampled with an interval that we know has no intersections
  RGBF shadow_term = splat_color(1.0f);

  float dist;
  dist = -logf(random_1D(RANDOM_TARGET_LIGHT_GEO_BRIDGE_DISTANCE + seed * LIGHT_GEO_MAX_BRIDGE_LENGTH + 0, pixel)) * scale;

  const RGBF shadow_vertex0 =
    optix_geometry_shadowing(TRIANGLE_HANDLE_INVALID, current_vertex, current_direction, dist, light_handle, trace_status);
  shadow_term = mul_color(shadow_term, shadow_vertex0);

  for (uint32_t vertex_id = 1; vertex_id < BRIDGES_MAX_VERTEX_COUNT; vertex_id++) {
    const bool vertex_is_valid = (vertex_id < vertex_count);

    if (vertex_is_valid) {
      current_vertex = add_vector(current_vertex, scale_vector(current_direction, dist));

      const float2 random_phase = random_2D(RANDOM_TARGET_LIGHT_GEO_BRIDGE_PHASE + seed * LIGHT_GEO_MAX_BRIDGE_LENGTH + vertex_id, pixel);

      current_direction_sampled = bridges_phase_sample(current_direction_sampled, random_phase);

      current_direction = quaternion16_apply(rotation, current_direction_sampled);

      dist = -logf(random_1D(RANDOM_TARGET_LIGHT_GEO_BRIDGE_DISTANCE + seed * LIGHT_GEO_MAX_BRIDGE_LENGTH + vertex_id, pixel)) * scale;
    }

    // We always execute all shadow rays as Nvidia hardware prefers that over conditional branching.
    const OptixTraceStatus trace_status_vertex = (vertex_is_valid) ? trace_status : OPTIX_TRACE_STATUS_OPTIONAL_UNUSED;

    const RGBF shadow_vertex =
      optix_geometry_shadowing(TRIANGLE_HANDLE_INVALID, current_vertex, current_direction, dist, light_handle, trace_status_vertex);
    shadow_term = mul_color(shadow_term, shadow_vertex);
  }

  return (bridge_is_valid) ? mul_color(direct_light_task.light_color, shadow_term) : splat_color(0.0f);
}

#endif /* OPTIX_KERNEL */

#endif /* CU_LUMINARY_LIGHT_BRIDGES_H */

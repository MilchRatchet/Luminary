#ifndef CU_LUMINARY_LIGHT_BRIDGES_H
#define CU_LUMINARY_LIGHT_BRIDGES_H

#if defined(OPTIX_KERNEL)

#include "light.cuh"
#include "light_common.cuh"
#include "light_triangle.cuh"
#include "math.cuh"
#include "memory.cuh"
#include "ocean_utils.cuh"
#include "optix_include.cuh"
#include "random.cuh"
#include "ris.cuh"
#include "utils.cuh"
#include "utils.h"
#include "volume_utils.cuh"

// This must correspond to the G term used when computing the LUT.
#define BRIDGES_HG_G_TERM (0.85f)
#define BRIDGES_INITIAL_VERTEX_FORWARD_PROB (0.95f)

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

  const float random         = random_1D(RANDOM_TARGET_LIGHT_GEO_BRIDGE_VERTEX_COUNT + seed, pixel);
  RISReservoir ris_reservoir = ris_reservoir_init(random);

  ////////////////////////////////////////////////////////////////////
  // Sample a vertex count based on importance LUT
  ////////////////////////////////////////////////////////////////////

  // Fallback values. These come into play if sum_importance is 0.0f.
  // This happens when effective dist is too large.
  uint32_t selected_vertex_count = device.settings.bridge_max_num_vertices - 1;

  for (uint32_t i = 0; i < device.settings.bridge_max_num_vertices; i++) {
    // TODO: The paper uses some additional terms here for the importance
    const float importance = bridges_get_vertex_count_importance(i + 1, effective_dist);

    if (ris_reservoir_add_sample(ris_reservoir, importance, 1.0f)) {
      selected_vertex_count = i;
    }
  }

  pdf = ris_reservoir_get_sampling_prob(ris_reservoir);

  return 1 + selected_vertex_count;
}

__device__ RGBF bridges_sample_bridge(
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

__device__ vec3 bridges_sample_initial_vertex(
  MaterialContextVolume ctx, const vec3 point_on_light, const ushort2 pixel, const uint32_t output_id, RGBF& attenuation, float& pdf) {
  float random_intersection = random_1D(RANDOM_TARGET_LIGHT_GEO_INITIAL_VERTEX + output_id, pixel);

  const vec3 PO             = sub_vector(point_on_light, ctx.position);
  const float dist_to_light = -dot_product(PO, ctx.V);

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

__device__ LightSampleResult<MATERIAL_VOLUME> bridges_sample(
  MaterialContextVolume ctx, TriangleLight light, const TriangleHandle light_handle, const uint3 light_uv_packed, const ushort2 pixel,
  const uint32_t output_id, float2& target_and_weight) {
  const float2 random_light_point = random_2D(RANDOM_TARGET_LIGHT_GEO_BRIDGE_LIGHT_POINT + output_id, pixel);

  const vec3 point_on_light = light_triangle_sample_bridges(light, random_light_point);

  RGBF initial_attenuation;
  float initial_pdf;
  const vec3 initial_vertex = bridges_sample_initial_vertex(ctx, point_on_light, pixel, output_id, initial_attenuation, initial_pdf);

  LightSampleResult<MATERIAL_VOLUME> result;
  result.handle = triangle_handle_get(INSTANCE_ID_INVALID, 0);

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

  result.handle      = light_handle;
  result.light_color = sample_path_weight;
  result.rotation    = sample_rotation;
  result.scale       = sample_path_scale;
  result.seed        = output_id;

  return result;
}

__device__ RGBF
  bridges_sample_apply_shadowing(const MaterialContextVolume ctx, const LightSampleResult<MATERIAL_VOLUME> sample, const ushort2 pixel) {
  bool bridge_is_valid = true;
  bridge_is_valid &= sample.handle.instance_id != INSTANCE_ID_INVALID;
  bridge_is_valid &= sample.seed != 0xFFFFFFFF;

  ////////////////////////////////////////////////////////////////////
  // Get light sample
  ////////////////////////////////////////////////////////////////////

  OptixTraceStatus trace_status = (bridge_is_valid) ? OPTIX_TRACE_STATUS_EXECUTE : OPTIX_TRACE_STATUS_ABORT;
  vec3 light_vector             = get_vector(0.0f, 0.0f, 1.0f);
  vec3 initial_vertex           = get_vector(0.0f, 0.0f, 0.0f);

  if (bridge_is_valid) {
    const DeviceTransform light_transform = load_transform(sample.handle.instance_id);

    uint3 light_uv_packed;
    TriangleLight light = light_triangle_sample_init(sample.handle, light_transform, light_uv_packed);

    const float2 random_light_point = random_2D(RANDOM_TARGET_LIGHT_GEO_BRIDGE_LIGHT_POINT + sample.seed, pixel);

    const vec3 point_on_light = light_triangle_sample_bridges(light, random_light_point);

    RGBF initial_attenuation;
    float initial_pdf;
    initial_vertex = bridges_sample_initial_vertex(ctx, point_on_light, pixel, sample.seed, initial_attenuation, initial_pdf);

    vec3 light_dir;
    float area, light_dist;
    light_triangle_sample_finalize_bridges(light, light_uv_packed, initial_vertex, point_on_light, light_dir, light_dist, area);

    light_vector = scale_vector(light_dir, light_dist);
  }

  ////////////////////////////////////////////////////////////////////
  // Get sampled vertex count
  ////////////////////////////////////////////////////////////////////

  uint32_t vertex_count;
  if (bridge_is_valid) {
    float vertex_count_pdf;
    vertex_count = bridges_sample_vertex_count(ctx.descriptor, get_length(light_vector), sample.seed, pixel, vertex_count_pdf);
  }
  else {
    vertex_count = 1;
  }

  ////////////////////////////////////////////////////////////////////
  // Compute visibility of path
  ////////////////////////////////////////////////////////////////////

  vec3 current_vertex            = initial_vertex;
  vec3 current_direction_sampled = normalize_vector(light_vector);
  vec3 current_direction         = quaternion_apply(sample.rotation, current_direction_sampled);

  // We don't need to trace visibility to the initial vertex as it was sampled with an interval that we know has no intersections
  RGBF shadow_term = splat_color(1.0f);

  float sum_dist = 0.0f;

  float dist;
  dist = -logf(random_1D(RANDOM_TARGET_LIGHT_GEO_BRIDGE_DISTANCE + sample.seed * LIGHT_GEO_MAX_BRIDGE_LENGTH + 0, pixel)) * sample.scale;

  shadow_term = mul_color(shadow_term, optix_geometry_shadowing(current_vertex, current_direction, dist, sample.handle, trace_status));

  sum_dist += dist;

  for (int i = 1; i < vertex_count; i++) {
    current_vertex = add_vector(current_vertex, scale_vector(current_direction, dist));

    const float2 random_phase = random_2D(RANDOM_TARGET_LIGHT_GEO_BRIDGE_PHASE + sample.seed * LIGHT_GEO_MAX_BRIDGE_LENGTH + i, pixel);

    current_direction_sampled = bridges_phase_sample(current_direction_sampled, random_phase);

    current_direction = quaternion_apply(sample.rotation, current_direction_sampled);

    dist = -logf(random_1D(RANDOM_TARGET_LIGHT_GEO_BRIDGE_DISTANCE + sample.seed * LIGHT_GEO_MAX_BRIDGE_LENGTH + i, pixel)) * sample.scale;

    shadow_term = mul_color(shadow_term, optix_geometry_shadowing(current_vertex, current_direction, dist, sample.handle, trace_status));

    sum_dist += dist;
  }

  return mul_color(sample.light_color, shadow_term);
}

#endif /* OPTIX_KERNEL */

#endif /* CU_LUMINARY_LIGHT_BRIDGES_H */

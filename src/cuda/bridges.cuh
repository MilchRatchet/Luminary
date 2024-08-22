#ifndef CU_BRIDGES_H
#define CU_BRIDGES_H

#include "light.cuh"
#include "math.cuh"
#include "memory.cuh"
#include "random.cuh"
#include "utils.cuh"
#include "utils.h"
#include "volume_utils.cuh"

struct BridgeSample {
  uint32_t seed;
  vec3 end_vertex;
  RGBF weight;
} typedef BridgeSample;

// TODO: I wrote this for fog but obviously it needs to work for all volumes.
__device__ BridgeSample bridges_sample_bridge(const GBufferData data, const vec3 light_point, const uint32_t seed, const ushort2 pixel) {
  const vec3 initial_direction = sub_vector(light_point, data.position);

  vec3 current_point     = data.origin;
  vec3 current_direction = normalize_vector(initial_direction);

  for (int i = 0; i < device.bridge_settings.max_num_vertices; i++) {
    const float random_dist = quasirandom_sequence_1D(QUASI_RANDOM_TARGET_BRIDGE_DISTANCE + seed * 32 + i, pixel);

    const float dist = -logf(random_dist);

    current_point = add_vector(current_point, scale_vector(current_direction, dist));

    const float2 random_phase = quasirandom_sequence_2D(QUASI_RANDOM_TARGET_BRIDGE_PHASE + seed * 32 + i, pixel);
    const float random_method = quasirandom_sequence_1D(QUASI_RANDOM_TARGET_BRIDGE_PHASE_METHOD + seed * 32 + i, pixel);

    current_direction = jendersie_eon_phase_sample(current_direction, device.scene.fog.droplet_diameter, random_phase, random_method);
  }

  const float target_scale  = get_length(initial_direction);
  const float actual_scale  = get_length(sub_vector(current_point, data.position));
  const float scale         = target_scale / actual_scale;
  const float inverse_scale = actual_scale / target_scale;

  // TODO: This will actually have to be a RGBF.
  const float weight = 1.0f;

  for (int i = 0; i < device.bridge_settings.max_num_vertices; i++) {
    const float random_dist = quasirandom_sequence_1D(QUASI_RANDOM_TARGET_BRIDGE_DISTANCE + seed * 32 + i, pixel);

    const float dist = -logf(random_dist) * scale;

    // TODO: I can probably simplify this.
    const float extinction = expf(-dist * FOG_DENSITY);
    const float pdf        = expf(-dist * inverse_scale);

    weight *= extinction / pdf;
  }

  BridgeSample sample;

  sample.seed       = seed;
  sample.end_vertex = current_point;
  sample.weight     = get_color(weight, weight, weight);

  return sample;
}

__device__ RGBF bridges_sample(const GBufferData data) {
  ////////////////////////////////////////////////////////////////////
  // Sample light
  ////////////////////////////////////////////////////////////////////

  const float random_light_tree = quasirandom_sequence_1D(QUASI_RANDOM_TARGET_BRIDGE_LIGHT_TREE, task.index);

  uint32_t light_list_length;
  float light_list_pdf;
  const uint32_t light_list_ptr =
    light_tree_traverse(data.position, scale_vector(data.V, -1.0f), random_light_tree, light_list_length, light_list_pdf);

  const float random_light_list = quasirandom_sequence_1D(QUASI_RANDOM_TARGET_BRIDGE_LIGHT_LIST, task.index);

  const uint32_t light_id = uint32_t(__fmul_rd(random_light_list, light_list_length)) + light_list_ptr;

  const TriangleLight light = load_triangle_light(device.scene.triangle_lights, light_id);

  const float2 random_light_point = quasirandom_sequence_2D(QUASI_RANDOM_TARGET_BRIDGE_LIGHT_POINT, task.index);

  RGBF light_color;
  const vec3 light_point = light_sample_triangle_bridge(light, random_light_point, light_color);

  const vec3 initial_direction = sub_vector(data.position, light_point);

  // We sampled a point that emits no light, skip.
  if (color_importance(light_color) == 0.0f)
    continue;

  ////////////////////////////////////////////////////////////////////
  // Sample vertex count
  ////////////////////////////////////////////////////////////////////

  ////////////////////////////////////////////////////////////////////
  // Sample path
  ////////////////////////////////////////////////////////////////////

  ////////////////////////////////////////////////////////////////////
  // Modify path
  ////////////////////////////////////////////////////////////////////
}

#endif /* CU_BRIDGES_H */

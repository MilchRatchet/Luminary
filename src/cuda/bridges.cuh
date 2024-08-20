#ifndef CU_BRIDGES_H
#define CU_BRIDGES_H

#include "light.cuh"
#include "math.cuh"
#include "memory.cuh"
#include "random.cuh"
#include "utils.cuh"
#include "utils.h"

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

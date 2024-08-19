#define UTILS_NO_DEVICE_TABLE

#define SHADING_KERNEL
#define OPTIX_KERNEL
#define VOLUME_KERNEL
#define VOLUME_DL_KERNEL

#include "utils.h"

extern "C" static __constant__ DeviceConstantMemory device;

#include "light.cuh"
#include "memory.cuh"
#include "random.cuh"
#include "utils.cuh"

extern "C" __global__ void __raygen__optix() {
  const int task_count = device.ptrs.trace_counts[THREAD_ID];

  for (int i = 0; i < task_count; i++) {
    const int offset = get_task_address(i);

    ////////////////////////////////////////////////////////////////////
    // Gather ray info
    ////////////////////////////////////////////////////////////////////

    const TraceTask task = load_trace_task(device.ptrs.trace_tasks + offset);
    const uint32_t pixel = task.index.y * device.width + task.index.x;

    const float2 result = __ldcs((float2*) (device.ptrs.trace_results + offset));
    const float limit   = result.x;

    ////////////////////////////////////////////////////////////////////
    // Sample light
    ////////////////////////////////////////////////////////////////////

    const float random_light_tree = quasirandom_sequence_1D(QUASI_RANDOM_TARGET_BRIDGE_LIGHT_TREE, task.index);

    uint32_t light_list_length;
    float light_list_pdf;
    const uint32_t light_list_ptr = light_tree_traverse(task.origin, task.ray, limit, random_light_tree, light_list_length, light_list_pdf);

    const float random_light_list = quasirandom_sequence_1D(QUASI_RANDOM_TARGET_BRIDGE_LIGHT_LIST, task.index);

    const uint32_t light_id = uint32_t(__fmul_rd(random_light_list, light_list_length)) + light_list_ptr;

    const TriangleLight light = load_triangle_light(device.scene.triangle_lights, light_id);

    const float2 random_light_point = quasirandom_sequence_2D(QUASI_RANDOM_TARGET_BRIDGE_LIGHT_POINT, task.index);

    RGBF light_color;
    const vec3 light_point = light_sample_triangle_bridge(light, random_light_point, light_color);

    const float light_dist = get_length(sub_vector(task.origin, light_point));

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
}

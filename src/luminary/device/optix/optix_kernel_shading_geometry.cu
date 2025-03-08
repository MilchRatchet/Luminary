// Functions work differently when executed from this kernel
// This emulates the old device.iteration_type == TYPE_LIGHT checks.
#define SHADING_KERNEL
#define OPTIX_KERNEL

#include "bsdf.cuh"
#include "direct_lighting.cuh"
#include "geometry_utils.cuh"
#include "ior_stack.cuh"
#include "math.cuh"
#include "memory.cuh"
#include "shading_kernel.cuh"
#include "utils.cuh"

extern "C" __global__ void __raygen__optix() {
  HANDLE_DEVICE_ABORT();

  const int task_count  = device.ptrs.task_counts[TASK_ADDRESS_OFFSET_GEOMETRY];
  const int task_offset = device.ptrs.task_offsets[TASK_ADDRESS_OFFSET_GEOMETRY];

  for (int i = 0; i < task_count; i++) {
    const uint32_t offset                = get_task_address(task_offset + i);
    DeviceTask task                      = task_load(offset);
    const TriangleHandle triangle_handle = triangle_handle_load(offset);
    const float depth                    = trace_depth_load(offset);
    const uint32_t pixel                 = get_pixel_id(task.index);

    task.origin = add_vector(task.origin, scale_vector(task.ray, depth));

    GBufferData data = geometry_generate_g_buffer(task, triangle_handle, pixel);

    // We have to clamp due to numerical precision issues in the microfacet models.
    data.roughness = fmaxf(data.roughness, BSDF_ROUGHNESS_CLAMP);

    ////////////////////////////////////////////////////////////////////
    // Light Ray Sampling
    ////////////////////////////////////////////////////////////////////

    RGBF accumulated_light = splat_color(0.0f);
    accumulated_light      = add_color(accumulated_light, direct_lighting_sun(data, task.index));
    accumulated_light      = add_color(accumulated_light, direct_lighting_geometry(data, task.index));
    accumulated_light      = add_color(accumulated_light, direct_lighting_ambient(data, task.index));

    const RGBF record = load_RGBF(device.ptrs.records + pixel);

    accumulated_light = mul_color(accumulated_light, record);

    write_beauty_buffer(accumulated_light, pixel, task.state);
  }
}

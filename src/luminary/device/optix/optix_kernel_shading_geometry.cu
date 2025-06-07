// OptiX translation unit setup
#include "optix_compile_defines.cuh"
//

#include "bsdf.cuh"
#include "direct_lighting.cuh"
#include "geometry_utils.cuh"
#include "ior_stack.cuh"
#include "math.cuh"
#include "memory.cuh"
#include "utils.cuh"

extern "C" __global__ void __raygen__optix() {
  HANDLE_DEVICE_ABORT();

#ifdef OPTIX_ENABLE_GEOMETRY_DL
  if (LIGHTS_ARE_PRESENT == false)
    return;
#endif

  const uint32_t task_count  = device.ptrs.task_counts[TASK_ADDRESS_OFFSET_GEOMETRY];
  const uint32_t task_offset = device.ptrs.task_offsets[TASK_ADDRESS_OFFSET_GEOMETRY];
  const uint32_t task_id     = TASK_ID;

  if (task_id >= task_count)
    return;

  const uint32_t task_base_address      = task_get_base_address(task_offset + task_id, TASK_STATE_BUFFER_INDEX_POSTSORT);
  DeviceTask task                       = task_load(task_base_address);
  const DeviceTaskTrace trace           = task_trace_load(task_base_address);
  const DeviceTaskThroughput throughput = task_throughput_load(task_base_address);

#ifdef OPTIX_ENABLE_GEOMETRY_DL
  if (direct_lighting_geometry_is_valid(task) == false)
    return;
#endif

  task.origin = add_vector(task.origin, scale_vector(task.ray, trace.depth));

  DeviceIORStack ior_stack          = trace.ior_stack;
  const MaterialContextGeometry ctx = geometry_get_context(task, trace.handle, ior_stack, throughput.payload);

  ////////////////////////////////////////////////////////////////////
  // Light Ray Sampling
  ////////////////////////////////////////////////////////////////////

  RGBF accumulated_light = splat_color(0.0f);

#ifdef OPTIX_ENABLE_GEOMETRY_DL
  accumulated_light = add_color(accumulated_light, direct_lighting_geometry(ctx, task.index));
#endif

#ifdef OPTIX_ENABLE_SKY_DL
  accumulated_light = add_color(accumulated_light, direct_lighting_sun(ctx, task.index));
  accumulated_light = add_color(accumulated_light, direct_lighting_ambient(ctx, task.index));
#endif

  accumulated_light = mul_color(accumulated_light, record_unpack(throughput.record));

  const uint32_t pixel = get_pixel_id(task.index);
  write_beauty_buffer(accumulated_light, pixel, task.state);
}

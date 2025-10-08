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

  // TODO: Combine particle and geometry tasks into one as they have the same shadow logic
  const uint32_t task_count  = device.ptrs.task_counts[TASK_ADDRESS_OFFSET_GEOMETRY];
  const uint32_t task_offset = device.ptrs.task_offsets[TASK_ADDRESS_OFFSET_GEOMETRY];
  const uint32_t task_id     = TASK_ID;

  if (task_id >= task_count)
    return;

  const uint32_t task_base_address = task_get_base_address(task_offset + task_id, TASK_STATE_BUFFER_INDEX_POSTSORT);
  DeviceTask task                  = task_load(task_base_address);
  const DeviceTaskTrace trace      = task_trace_load(task_base_address);

  task.origin = add_vector(task.origin, scale_vector(task.ray, trace.depth));

  const uint32_t direct_light_task_base_address = task_get_base_address(task_offset + task_id, TASK_STATE_BUFFER_INDEX_DIRECT_LIGHT);

  RGBF accumulated_light = splat_color(0.0f);

  ////////////////////////////////////////////////////////////////////
  // Shadow Geometry
  ////////////////////////////////////////////////////////////////////

  {
    const DeviceTaskDirectLightGeo direct_light_task = task_direct_light_geo_load(direct_light_task_base_address);
    const RGBF direct_light_result                   = direct_lighting_geometry_evaluate_task(task, trace, direct_light_task);

    accumulated_light = add_color(accumulated_light, direct_light_result);
  }

  ////////////////////////////////////////////////////////////////////
  // Shadow Sun
  ////////////////////////////////////////////////////////////////////

  {
    const DeviceTaskDirectLightSun direct_light_task = task_direct_light_sun_load(direct_light_task_base_address);
    const RGBF direct_light_result                   = direct_lighting_sun_evaluate_task(task, trace, direct_light_task);

    accumulated_light = add_color(accumulated_light, direct_light_result);
  }

  ////////////////////////////////////////////////////////////////////
  // Shadow Ambient
  ////////////////////////////////////////////////////////////////////

  {
    const DeviceTaskDirectLightAmbient direct_light_task = task_direct_light_ambient_load(direct_light_task_base_address);
    const RGBF direct_light_result                       = direct_lighting_ambient_evaluate_task(task, trace, direct_light_task);

    accumulated_light = add_color(accumulated_light, direct_light_result);
  }

  ////////////////////////////////////////////////////////////////////
  // Store Result
  ////////////////////////////////////////////////////////////////////

  const DeviceTaskThroughput throughput = task_throughput_load(task_base_address);

  accumulated_light = mul_color(accumulated_light, record_unpack(throughput.record));

  const uint32_t pixel = get_pixel_id(task.index);
  write_beauty_buffer(accumulated_light, pixel, task.state);
}

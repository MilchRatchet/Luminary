// OptiX translation unit setup
#define OPTIX_KERNEL
//

#include "bsdf.cuh"
#include "direct_lighting.cuh"
#include "geometry_utils.cuh"
#include "ior_stack.cuh"
#include "math.cuh"
#include "memory.cuh"
#include "utils.cuh"

static_assert(
  SHADING_TASK_INDEX_PARTICLE == SHADING_TASK_INDEX_GEOMETRY + 1, "This assumes that particle tasks come directly after geometry tasks.");

extern "C" __global__ void __raygen__optix() {
  HANDLE_DEVICE_ABORT();

  const uint32_t task_count = device.ptrs.task_counts[TASK_ADDRESS_OFFSET_GEOMETRY] + device.ptrs.task_counts[TASK_ADDRESS_OFFSET_PARTICLE];
  const uint32_t task_offset = device.ptrs.task_offsets[TASK_ADDRESS_OFFSET_GEOMETRY];
  const uint32_t task_id     = TASK_ID;

  if (task_id >= task_count)
    return;

  const uint32_t task_base_address = task_get_base_address(task_offset + task_id, TASK_STATE_BUFFER_INDEX_POSTSORT);
  DeviceTask task                  = task_load(task_base_address);
  const DeviceTaskTrace trace      = task_trace_load(task_base_address);

  task.origin = add_vector(task.origin, scale_vector(task.ray, trace.depth));

  const uint32_t direct_light_task_base_address =
    task_get_base_address<DeviceTaskDirectLight>(task_offset + task_id, TASK_STATE_BUFFER_INDEX_DIRECT_LIGHT);

  RGBF accumulated_light = splat_color(0.0f);

  ////////////////////////////////////////////////////////////////////
  // Shadow Geometry
  ////////////////////////////////////////////////////////////////////

  // TODO: Dont load if not allowed
  {
    const DeviceTaskDirectLightGeo direct_light_task = task_direct_light_geo_load(direct_light_task_base_address);

    const bool is_allowed          = direct_lighting_geometry_is_allowed(task);
    const RGBF direct_light_result = direct_lighting_geometry_evaluate_task(task, trace, direct_light_task, is_allowed);

    accumulated_light = add_color(accumulated_light, direct_light_result);
  }

  ////////////////////////////////////////////////////////////////////
  // Shadow BSDF
  ////////////////////////////////////////////////////////////////////

  {
    const DeviceTaskDirectLightBSDF direct_light_task = task_direct_light_bsdf_load(direct_light_task_base_address);

    const bool is_allowed          = direct_lighting_bsdf_is_allowed(task, trace);
    const RGBF direct_light_result = direct_lighting_bsdf_evaluate_task(task, trace, direct_light_task, is_allowed);

    accumulated_light = add_color(accumulated_light, direct_light_result);
  }

  ////////////////////////////////////////////////////////////////////
  // Shadow Sun
  ////////////////////////////////////////////////////////////////////

  {
    const DeviceTaskDirectLightSun direct_light_task = task_direct_light_sun_load(direct_light_task_base_address);

    const bool is_allowed          = direct_lighting_sun_is_allowed(task);
    const RGBF direct_light_result = direct_lighting_sun_evaluate_task(task, trace, direct_light_task, is_allowed);

    accumulated_light = add_color(accumulated_light, direct_light_result);
  }

  ////////////////////////////////////////////////////////////////////
  // Shadow Ambient
  ////////////////////////////////////////////////////////////////////

  {
    const DeviceTaskDirectLightAmbient direct_light_task = task_direct_light_ambient_load(direct_light_task_base_address);

    const bool is_allowed          = direct_lighting_ambient_is_allowed(task);
    const RGBF direct_light_result = direct_lighting_ambient_evaluate_task(task, trace, direct_light_task, is_allowed);

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

// OptiX translation unit setup
#define OPTIX_KERNEL
//

#include "bsdf.cuh"
#include "direct_lighting.cuh"
#include "geometry_utils.cuh"
#include "math.cuh"
#include "medium_stack.cuh"
#include "memory.cuh"
#include "utils.cuh"

extern "C" __global__ void __raygen__optix() {
  HANDLE_DEVICE_ABORT();

  const uint32_t task_count = device.ptrs.trace_counts[THREAD_ID];
  const uint32_t task_id    = TASK_ID;

  if (task_id >= task_count)
    return;

  const uint32_t task_base_address   = task_get_base_address(task_id, TASK_STATE_BUFFER_INDEX_PRESORT);
  DeviceTask task                    = task_load(task_base_address);
  DeviceTaskTrace trace              = task_trace_load(task_base_address);
  const DeviceTaskMediumStack medium = task_medium_load(task_base_address);

  // The trace handle contains the actual handle of the ray's end point but the direct lighting functions assume that it is
  // the handle of the starting point so we must invalidate it.
  trace.handle = TRIANGLE_HANDLE_INVALID;

  const VolumeType volume_type  = (VolumeType) medium_stack_volume_peek(medium, false);
  const VolumeDescriptor volume = volume_get_descriptor_preset(volume_type);

  MaterialContextVolume ctx = volume_get_context(task, volume, trace.depth);

  const uint32_t direct_light_task_base_address =
    task_get_base_address<DeviceTaskDirectLight>(task_id, TASK_STATE_BUFFER_INDEX_DIRECT_LIGHT);

  RGBF accumulated_light = splat_color(0.0f);

  ////////////////////////////////////////////////////////////////////
  // Shadow Geometry
  ////////////////////////////////////////////////////////////////////

  {
    const DeviceTaskDirectLightBridges direct_light_task = task_direct_light_bridges_load(direct_light_task_base_address);

    const bool is_allowed          = direct_lighting_bridges_is_allowed(ctx);
    const RGBF direct_light_result = direct_lighting_bridges_evaluate_task(ctx, direct_light_task, task.path_id, is_allowed);

    accumulated_light = add_color(accumulated_light, direct_light_result);
  }

  ////////////////////////////////////////////////////////////////////
  // Initial vertex sampling for Sun and Ambient
  ////////////////////////////////////////////////////////////////////

  RGBF initial_vertex_weight;
  volume_sample_sky_dl_initial_vertex(ctx, task.path_id, initial_vertex_weight);

  task.origin = ctx.position;

  ////////////////////////////////////////////////////////////////////
  // Shadow Sun
  ////////////////////////////////////////////////////////////////////

  {
    const DeviceTaskDirectLightSun direct_light_task = task_direct_light_sun_load(direct_light_task_base_address);

    const bool is_allowed          = direct_lighting_sun_is_allowed(ctx);
    const RGBF direct_light_result = direct_lighting_sun_evaluate_task(task, trace, medium, direct_light_task, is_allowed);

    accumulated_light = add_color(accumulated_light, mul_color(direct_light_result, initial_vertex_weight));
  }

  ////////////////////////////////////////////////////////////////////
  // Shadow Ambient
  ////////////////////////////////////////////////////////////////////

  {
    const DeviceTaskDirectLightAmbient direct_light_task = task_direct_light_ambient_load(direct_light_task_base_address);

    const bool is_allowed          = direct_lighting_ambient_is_allowed(ctx);
    const RGBF direct_light_result = direct_lighting_ambient_evaluate_task(task, trace, medium, direct_light_task, is_allowed);

    accumulated_light = add_color(accumulated_light, mul_color(direct_light_result, initial_vertex_weight));
  }

  ////////////////////////////////////////////////////////////////////
  // Store Result
  ////////////////////////////////////////////////////////////////////

  const DeviceTaskThroughput throughput = task_throughput_load(task_base_address);

  accumulated_light = mul_color(accumulated_light, record_unpack(throughput.record));

  const uint32_t index = path_id_get_pixel_index(task.path_id);
  write_beauty_buffer(accumulated_light, index, task.state);
}

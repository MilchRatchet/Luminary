// OptiX translation unit setup
#include "optix_compile_defines.cuh"
//

#include "bsdf.cuh"
#include "direct_lighting.cuh"
#include "directives.cuh"
#include "ior_stack.cuh"
#include "math.cuh"
#include "memory.cuh"
#include "utils.cuh"
#include "volume_utils.cuh"

extern "C" __global__ void __raygen__optix() {
  HANDLE_DEVICE_ABORT();

#ifdef OPTIX_ENABLE_GEOMETRY_DL
  if (LIGHTS_ARE_PRESENT == false)
    return;
#endif /* OPTIX_ENABLE_GEOMETRY_DL */

  const uint32_t task_count = device.ptrs.trace_counts[THREAD_ID];
  const uint32_t task_id    = TASK_ID;

  if (task_id >= task_count)
    return;

  const uint32_t task_base_address = task_get_base_address(task_id, TASK_STATE_BUFFER_INDEX_PRESORT);
  DeviceTask task                  = task_load(task_base_address);
  const DeviceTaskTrace trace      = task_trace_load(task_base_address);
  DeviceTaskThroughput throughput  = task_throughput_load(task_base_address);

  const VolumeType volume_type = VolumeType(task.volume_id);

#ifdef OPTIX_ENABLE_GEOMETRY_DL
  if (volume_should_do_geometry_direct_lighting(volume_type, task.state) == false)
    return;
#endif /* OPTIX_ENABLE_GEOMETRY_DL */

#ifdef OPTIX_ENABLE_SKY_DL
  if (volume_should_do_sky_direct_lighting(volume_type, task.state) == false)
    return;
#endif /* OPTIX_ENABLE_SKY_DL */

  const VolumeDescriptor volume = volume_get_descriptor_preset(volume_type);

  MaterialContextVolume ctx = volume_get_context(task, volume, trace.depth);

  RGBF accumulated_light = get_color(0.0f, 0.0f, 0.0f);

#ifdef OPTIX_ENABLE_GEOMETRY_DL
  accumulated_light = add_color(accumulated_light, direct_lighting_geometry(ctx, task.index));
#endif /* OPTIX_ENABLE_GEOMETRY_DL */

#ifdef OPTIX_ENABLE_SKY_DL
  volume_sample_sky_dl_initial_vertex(ctx, task.index, throughput);

  accumulated_light = add_color(accumulated_light, direct_lighting_sun(ctx, task.index));
  accumulated_light = add_color(accumulated_light, direct_lighting_ambient(ctx, task.index));
#endif /* OPTIX_ENABLE_SKY_DL */

  accumulated_light = mul_color(accumulated_light, record_unpack(throughput.record));

  const uint32_t pixel = get_pixel_id(task.index);
  write_beauty_buffer_indirect(accumulated_light, pixel);
}

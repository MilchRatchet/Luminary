// OptiX translation unit setup
#include "optix_compile_defines.cuh"
//

#include "bsdf.cuh"
#include "direct_lighting.cuh"
#include "directives.cuh"
#include "ior_stack.cuh"
#include "math.cuh"
#include "memory.cuh"
#include "particle_utils.cuh"
#include "utils.cuh"
#include "volume_utils.cuh"

extern "C" __global__ void __raygen__optix() {
  HANDLE_DEVICE_ABORT();

  const uint32_t task_count  = device.ptrs.task_counts[TASK_ADDRESS_OFFSET_PARTICLE];
  const uint32_t task_offset = device.ptrs.task_offsets[TASK_ADDRESS_OFFSET_PARTICLE];
  const uint32_t task_id     = TASK_ID;

  if (task_id >= task_count)
    return;

  const uint32_t offset       = get_task_address(task_offset + task_id);
  DeviceTask task             = task_load(offset);
  const TriangleHandle handle = triangle_handle_load(offset);
  const float depth           = trace_depth_load(offset);
  const uint32_t pixel        = get_pixel_id(task.index);

  task.origin = add_vector(task.origin, scale_vector(task.ray, depth));

  const VolumeType volume_type  = VOLUME_HIT_TYPE(handle.instance_id);
  const VolumeDescriptor volume = volume_get_descriptor_preset(volume_type);

  const GBufferData data = particle_generate_g_buffer(task, handle.instance_id, pixel);

  const RGBF record = load_RGBF(device.ptrs.records + pixel);

  RGBF accumulated_light = get_color(0.0f, 0.0f, 0.0f);

  accumulated_light = add_color(accumulated_light, direct_lighting_geometry(data, task.index));
  accumulated_light = add_color(accumulated_light, direct_lighting_sun(data, task.index));
  accumulated_light = add_color(accumulated_light, direct_lighting_ambient(data, task.index));

  accumulated_light = mul_color(accumulated_light, record);

  write_beauty_buffer(accumulated_light, pixel, task.state);
}

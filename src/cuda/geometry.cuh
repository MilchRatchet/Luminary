#ifndef CU_GEOMETRY_H
#define CU_GEOMETRY_H

#include "geometry_utils.cuh"
#include "math.cuh"
#include "memory.cuh"
#include "state.cuh"

LUMINARY_KERNEL void process_debug_geometry_tasks() {
  const int task_count = device.ptrs.task_counts[THREAD_ID * TASK_ADDRESS_COUNT_STRIDE + TASK_ADDRESS_OFFSET_GEOMETRY];

  for (int i = 0; i < task_count; i++) {
    GeometryTask task = load_geometry_task(device.ptrs.trace_tasks + get_task_address(i));
    const int pixel   = task.index.y * device.width + task.index.x;

    if (device.shading_mode == SHADING_ALBEDO) {
      const GBufferData data = geometry_generate_g_buffer(task, pixel);

      write_beauty_buffer(add_color(opaque_color(data.albedo), data.emission), pixel, true);
    }
    else if (device.shading_mode == SHADING_DEPTH) {
      const float dist  = get_length(sub_vector(device.scene.camera.pos, task.position));
      const float value = __saturatef((1.0f / dist) * 2.0f);
      write_beauty_buffer(get_color(value, value, value), pixel, true);
    }
    else if (device.shading_mode == SHADING_NORMAL) {
      const GBufferData data = geometry_generate_g_buffer(task, pixel);

      const vec3 normal = data.normal;

      write_beauty_buffer(get_color(__saturatef(normal.x), __saturatef(normal.y), __saturatef(normal.z)), pixel, true);
    }
    else if (device.shading_mode == SHADING_HEAT) {
      const float cost  = device.ptrs.trace_result_buffer[pixel].depth;
      const float value = 0.1f * cost;
      const float red   = __saturatef(2.0f * value);
      const float green = __saturatef(2.0f * (value - 0.5f));
      const float blue  = __saturatef((value > 0.5f) ? 4.0f * (0.25f - fabsf(value - 1.0f)) : 4.0f * (0.25f - fabsf(value - 0.25f)));
      write_beauty_buffer(get_color(red, green, blue), pixel, true);
    }
    else if (device.shading_mode == SHADING_IDENTIFICATION) {
      const uint32_t v = random_uint32_t_base(0, task.hit_id);

      const uint16_t r = v & 0x7ff;
      const uint16_t g = (v >> 10) & 0x7ff;
      const uint16_t b = (v >> 20) & 0x7ff;

      const float cr = ((float) r) / 0x7ff;
      const float cg = ((float) g) / 0x7ff;
      const float cb = ((float) b) / 0x7ff;

      const RGBF color = get_color(cr, cg, cb);

      write_beauty_buffer(color, pixel, true);
    }
    else if (device.shading_mode == SHADING_LIGHTS) {
      const GBufferData data = geometry_generate_g_buffer(task, pixel);

      RGBF color = add_color(scale_color(opaque_color(data.albedo), 0.5f), scale_color(data.emission, 100.0f));

      write_beauty_buffer(color, pixel, true);
    }
  }
}

#endif /* CU_GEOMETRY_H */

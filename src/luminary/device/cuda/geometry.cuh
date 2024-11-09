#ifndef CU_GEOMETRY_H
#define CU_GEOMETRY_H

#include "geometry_utils.cuh"
#include "math.cuh"
#include "memory.cuh"
#include "ocean_utils.cuh"
#include "toy_utils.cuh"

LUMINARY_KERNEL void process_debug_geometry_tasks() {
  const int task_count = device.ptrs.task_counts[THREAD_ID * TASK_ADDRESS_COUNT_STRIDE + TASK_ADDRESS_OFFSET_GEOMETRY];

  for (int i = 0; i < task_count; i++) {
    const uint32_t offset       = get_task_address(i);
    ShadingTask task            = load_shading_task(offset);
    ShadingTaskAuxData aux_data = load_shading_task_aux_data(offset);
    const uint32_t pixel        = get_pixel_id(task.index);

    switch (task.instance_id) {
      case HIT_TYPE_TOY: {
        switch (device.settings.shading_mode) {
          case LUMINARY_SHADING_MODE_ALBEDO: {
            write_beauty_buffer_forced(opaque_color(device.toy.albedo), pixel);
          } break;
          case LUMINARY_SHADING_MODE_DEPTH: {
            const float dist  = get_length(sub_vector(device.camera.pos, task.position));
            const float value = __saturatef((1.0f / dist) * 2.0f);
            write_beauty_buffer_forced(get_color(value, value, value), pixel);
          } break;
          case LUMINARY_SHADING_MODE_NORMAL: {
            vec3 normal = get_toy_normal(task.position);

            if (dot_product(normal, task.ray) > 0.0f) {
              normal = scale_vector(normal, -1.0f);
            }

            normal.x = 0.5f * normal.x + 0.5f;
            normal.y = 0.5f * normal.y + 0.5f;
            normal.z = 0.5f * normal.z + 0.5f;

            write_beauty_buffer_forced(get_color(__saturatef(normal.x), __saturatef(normal.y), __saturatef(normal.z)), pixel);
          } break;
          case LUMINARY_SHADING_MODE_IDENTIFICATION: {
            write_beauty_buffer_forced(get_color(1.0f, 0.63f, 0.0f), pixel);
          } break;
          case LUMINARY_SHADING_MODE_LIGHTS: {
            RGBF color;
            if (device.toy.emissive) {
              color = get_color(100.0f, 100.0f, 100.0f);
            }
            else {
              color = scale_color(opaque_color(device.toy.albedo), 0.1f);
            }

            write_beauty_buffer_forced(color, pixel);
          } break;
          default:
            break;
        }
      } break;
      case HIT_TYPE_OCEAN: {
        switch (device.settings.shading_mode) {
          case LUMINARY_SHADING_MODE_DEPTH: {
            const float dist  = get_length(sub_vector(device.camera.pos, task.position));
            const float value = __saturatef((1.0f / dist) * 2.0f);
            write_beauty_buffer_forced(get_color(value, value, value), pixel);
          } break;
          case LUMINARY_SHADING_MODE_NORMAL: {
            vec3 normal = ocean_get_normal(task.position);

            normal.x = 0.5f * normal.x + 0.5f;
            normal.y = 0.5f * normal.y + 0.5f;
            normal.z = 0.5f * normal.z + 0.5f;

            write_beauty_buffer_forced(get_color(__saturatef(normal.x), __saturatef(normal.y), __saturatef(normal.z)), pixel);
          } break;
          case LUMINARY_SHADING_MODE_IDENTIFICATION: {
            write_beauty_buffer_forced(get_color(0.0f, 0.0f, 1.0f), pixel);
          } break;
          default:
            break;
        }
      } break;
      default: {
        switch (device.settings.shading_mode) {
          case LUMINARY_SHADING_MODE_ALBEDO: {
            const GBufferData data = geometry_generate_g_buffer(task, aux_data, pixel);

            write_beauty_buffer_forced(add_color(opaque_color(data.albedo), data.emission), pixel);
          } break;
          case LUMINARY_SHADING_MODE_DEPTH: {
            const float dist  = get_length(sub_vector(device.camera.pos, task.position));
            const float value = __saturatef((1.0f / dist) * 2.0f);
            write_beauty_buffer_forced(get_color(value, value, value), pixel);
          } break;
          case LUMINARY_SHADING_MODE_NORMAL: {
            const GBufferData data = geometry_generate_g_buffer(task, aux_data, pixel);

            const vec3 normal = data.normal;

            write_beauty_buffer_forced(get_color(__saturatef(normal.x), __saturatef(normal.y), __saturatef(normal.z)), pixel);
          } break;
          case LUMINARY_SHADING_MODE_IDENTIFICATION: {
            const uint32_t v = random_uint32_t_base(0x55555555, (task.instance_id << 16) | aux_data.tri_id);

            const uint16_t r = v & 0x7ff;
            const uint16_t g = (v >> 10) & 0x7ff;
            const uint16_t b = (v >> 20) & 0x7ff;

            const float cr = ((float) r) / 0x7ff;
            const float cg = ((float) g) / 0x7ff;
            const float cb = ((float) b) / 0x7ff;

            const RGBF color = get_color(cr, cg, cb);

            write_beauty_buffer_forced(color, pixel);
          } break;
          case LUMINARY_SHADING_MODE_LIGHTS: {
            const GBufferData data = geometry_generate_g_buffer(task, aux_data, pixel);
            const RGBF color       = add_color(scale_color(opaque_color(data.albedo), 0.025f), data.emission);

            write_beauty_buffer_forced(color, pixel);
          } break;
          default:
            break;
        }
      } break;
    }
  }
}

#endif /* CU_GEOMETRY_H */

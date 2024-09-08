#ifndef CU_GEOMETRY_H
#define CU_GEOMETRY_H

#include "geometry_utils.cuh"
#include "math.cuh"
#include "memory.cuh"
#include "ocean_utils.cuh"
#include "state.cuh"
#include "toy_utils.cuh"

LUMINARY_KERNEL void process_debug_geometry_tasks() {
  const int task_count = device.ptrs.task_counts[THREAD_ID * TASK_ADDRESS_COUNT_STRIDE + TASK_ADDRESS_OFFSET_GEOMETRY];

  for (int i = 0; i < task_count; i++) {
    const uint32_t offset = get_task_address(i);
    ShadingTask task      = load_shading_task(device.ptrs.trace_tasks + offset);
    const uint32_t pixel  = get_pixel_id(task.index);

    switch (task.hit_id) {
      case HIT_TYPE_TOY: {
        switch (device.shading_mode) {
          case SHADING_ALBEDO: {
            write_beauty_buffer(opaque_color(device.scene.toy.albedo), pixel, true);
          } break;
          case SHADING_DEPTH: {
            const float dist  = get_length(sub_vector(device.scene.camera.pos, task.position));
            const float value = __saturatef((1.0f / dist) * 2.0f);
            write_beauty_buffer(get_color(value, value, value), pixel, true);
          } break;
          case SHADING_NORMAL: {
            vec3 normal = get_toy_normal(task.position);

            if (dot_product(normal, task.ray) > 0.0f) {
              normal = scale_vector(normal, -1.0f);
            }

            normal.x = 0.5f * normal.x + 0.5f;
            normal.y = 0.5f * normal.y + 0.5f;
            normal.z = 0.5f * normal.z + 0.5f;

            write_beauty_buffer(get_color(__saturatef(normal.x), __saturatef(normal.y), __saturatef(normal.z)), pixel, true);
          } break;
          case SHADING_IDENTIFICATION: {
            write_beauty_buffer(get_color(1.0f, 0.63f, 0.0f), pixel, true);
          } break;
          case SHADING_LIGHTS: {
            RGBF color;
            if (device.scene.toy.emissive) {
              color = get_color(100.0f, 100.0f, 100.0f);
            }
            else {
              color = scale_color(opaque_color(device.scene.toy.albedo), 0.1f);
            }

            write_beauty_buffer(color, pixel, true);
          } break;
          default:
            break;
        }
      } break;
      case HIT_TYPE_OCEAN: {
        switch (device.shading_mode) {
          case SHADING_DEPTH: {
            const float dist  = get_length(sub_vector(device.scene.camera.pos, task.position));
            const float value = __saturatef((1.0f / dist) * 2.0f);
            write_beauty_buffer(get_color(value, value, value), pixel, true);
          } break;
          case SHADING_NORMAL: {
            vec3 normal = ocean_get_normal(task.position);

            normal.x = 0.5f * normal.x + 0.5f;
            normal.y = 0.5f * normal.y + 0.5f;
            normal.z = 0.5f * normal.z + 0.5f;

            write_beauty_buffer(get_color(__saturatef(normal.x), __saturatef(normal.y), __saturatef(normal.z)), pixel, true);
          } break;
          case SHADING_IDENTIFICATION: {
            write_beauty_buffer(get_color(0.0f, 0.0f, 1.0f), pixel, true);
          } break;
          default:
            break;
        }
      } break;
      default: {
        switch (device.shading_mode) {
          case SHADING_ALBEDO: {
            const GBufferData data = geometry_generate_g_buffer(task, pixel);

            write_beauty_buffer(add_color(opaque_color(data.albedo), data.emission), pixel, true);
          } break;
          case SHADING_DEPTH: {
            const float dist  = get_length(sub_vector(device.scene.camera.pos, task.position));
            const float value = __saturatef((1.0f / dist) * 2.0f);
            write_beauty_buffer(get_color(value, value, value), pixel, true);
          } break;
          case SHADING_NORMAL: {
            const GBufferData data = geometry_generate_g_buffer(task, pixel);

            const vec3 normal = data.normal;

            write_beauty_buffer(get_color(__saturatef(normal.x), __saturatef(normal.y), __saturatef(normal.z)), pixel, true);
          } break;
          case SHADING_HEAT: {
            const float cost  = device.ptrs.trace_results[offset].depth;
            const float value = 0.1f * cost;
            const float red   = __saturatef(2.0f * value);
            const float green = __saturatef(2.0f * (value - 0.5f));
            const float blue  = __saturatef((value > 0.5f) ? 4.0f * (0.25f - fabsf(value - 1.0f)) : 4.0f * (0.25f - fabsf(value - 0.25f)));
            write_beauty_buffer(get_color(red, green, blue), pixel, true);
          } break;
          case SHADING_IDENTIFICATION: {
            const uint32_t v = random_uint32_t_base(0x55555555, task.hit_id);

            const uint16_t r = v & 0x7ff;
            const uint16_t g = (v >> 10) & 0x7ff;
            const uint16_t b = (v >> 20) & 0x7ff;

            const float cr = ((float) r) / 0x7ff;
            const float cg = ((float) g) / 0x7ff;
            const float cb = ((float) b) / 0x7ff;

            const RGBF color = get_color(cr, cg, cb);

            write_beauty_buffer(color, pixel, true);
          } break;
          case SHADING_LIGHTS: {
            const GBufferData data = geometry_generate_g_buffer(task, pixel);

            const uint32_t light_id = load_triangle_light_id(task.hit_id);

            RGBF color = scale_color(opaque_color(data.albedo), 0.025f);

            if (light_id != LIGHT_ID_NONE) {
              const TriangleLight tri_light = load_triangle_light(device.scene.triangle_lights, light_id);

#if 0
              const float power = tri_light.power;

              color = add_color(color, get_color(power, power, power));
#else
              const float value = 5.0f * tri_light.power;
              const float red   = __saturatef(2.0f * value);
              const float green = __saturatef(2.0f * (value - 0.5f));
              const float blue = __saturatef((value > 0.5f) ? 4.0f * (0.25f - fabsf(value - 1.0f)) : 4.0f * (0.25f - fabsf(value - 0.25f)));

              color = get_color(red, green, blue);
#endif
            }

            write_beauty_buffer(color, pixel, true);
          } break;
          default:
            break;
        }
      } break;
    }
  }
}

#endif /* CU_GEOMETRY_H */

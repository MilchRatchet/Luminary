#include "brdf.cuh"
#include "directives.cuh"
#include "geometry.cuh"
#include "math.cuh"
#include "memory.cuh"
#include "restir.cuh"
#include "utils.cuh"

__global__ __launch_bounds__(THREADS_PER_BLOCK, 7) void process_geometry_tasks() {
  const int task_count   = device.ptrs.task_counts[THREAD_ID * TASK_ADDRESS_COUNT_STRIDE + TASK_ADDRESS_OFFSET_GEOMETRY];
  const int task_offset  = device.ptrs.task_offsets[THREAD_ID * TASK_ADDRESS_OFFSET_STRIDE + TASK_ADDRESS_OFFSET_GEOMETRY];
  int light_trace_count  = device.ptrs.light_trace_count[THREAD_ID];
  int bounce_trace_count = device.ptrs.bounce_trace_count[THREAD_ID];

  for (int i = 0; i < task_count; i++) {
    GeometryTask task = load_geometry_task(device.trace_tasks + get_task_address(task_offset + i));
    const int pixel   = task.index.y * device.width + task.index.x;

    GBufferData data = geometry_generate_g_buffer(task, pixel);

    RGBF record = load_RGBF(device.records + pixel);

    if (data.albedo.a > 0.0f && color_any(data.emission)) {
      write_albedo_buffer(add_color(data.emission, opaque_color(data.albedo)), pixel);

      RGBF emission = mul_color(data.emission, record);

      if (device.iteration_type == TYPE_BOUNCE) {
        const float mis_weight = device.ptrs.mis_buffer[pixel];
        emission               = scale_color(emission, mis_weight);
      }

      const uint32_t light             = device.ptrs.light_sample_history[pixel];
      const uint32_t triangle_light_id = load_triangle_light_id(data.hit_id);

      if (proper_light_sample(light, triangle_light_id)) {
        write_beauty_buffer(emission, pixel);
      }
    }

    write_normal_buffer(data.normal, pixel);

    if (data.flags & G_BUFFER_TRANSPARENT_PASS && !device.scene.material.colored_transparency) {
      data.albedo.r = 1.0f;
      data.albedo.g = 1.0f;
      data.albedo.b = 1.0f;
    }

    BRDFInstance brdf = brdf_get_instance(data.albedo, data.V, data.normal, data.roughness, data.metallic);

    if (data.flags & G_BUFFER_TRANSPARENT_PASS) {
      if (device.iteration_type != TYPE_LIGHT) {
        const float ambient_index_of_refraction = geometry_get_ambient_index_of_refraction(data.position);

        const float refraction_index = (data.flags & G_BUFFER_REFRACTION_IS_INSIDE) ? data.refraction_index / ambient_index_of_refraction
                                                                                    : ambient_index_of_refraction / data.refraction_index;

        brdf = brdf_sample_ray_refraction(brdf, refraction_index, pixel);
      }
      else {
        brdf.term = mul_color(brdf.term, opaque_color(data.albedo));
        brdf.L    = task.ray;
      }

      record = mul_color(record, brdf.term);

      TraceTask new_task;
      new_task.origin = data.position;
      new_task.ray    = brdf.L;
      new_task.index  = task.index;

      switch (device.iteration_type) {
        case TYPE_CAMERA:
        case TYPE_BOUNCE:
          if (validate_trace_task(new_task, pixel, record)) {
            device.ptrs.mis_buffer[pixel] = 1.0f;
            store_RGBF(device.ptrs.bounce_records + pixel, record);
            store_trace_task(device.ptrs.bounce_trace + get_task_address(bounce_trace_count++), new_task);
          }
          break;
        case TYPE_LIGHT:
          if (quasirandom_sequence_1D(QUASI_RANDOM_TARGET_LIGHT_TRANSPARENCY, pixel) > data.albedo.a) {
            if (state_consume(pixel, STATE_FLAG_LIGHT_OCCUPIED)) {
              store_RGBF(device.ptrs.light_records + pixel, record);
              store_trace_task(device.ptrs.light_trace + get_task_address(light_trace_count++), new_task);
            }
          }
          break;
      }
    }
    else if (device.iteration_type != TYPE_LIGHT) {
      if (!material_is_mirror(data.roughness, data.metallic))
        write_albedo_buffer(opaque_color(data.albedo), pixel);

      bool bounce_is_specular;
      BRDFInstance bounce_brdf = brdf_sample_ray(brdf, pixel, bounce_is_specular);

      float bounce_mis_weight = 1.0f;

      if (!state_peek(pixel, STATE_FLAG_LIGHT_OCCUPIED)) {
        uint32_t light_history_buffer_entry = LIGHT_ID_ANY;
        LightSample light                   = restir_sample_reservoir(data, record, pixel);

        if (light.weight > 0.0f) {
          const BRDFInstance brdf_sample = brdf_apply_sample_weight(brdf_apply_sample(brdf, light, data.position, pixel));

          const RGBF light_record = mul_color(record, brdf_sample.term);

          TraceTask light_task;
          light_task.origin = data.position;
          light_task.ray    = brdf_sample.L;
          light_task.index  = task.index;

          if (luminance(light_record) > 0.0f && state_consume(pixel, STATE_FLAG_LIGHT_OCCUPIED)) {
            const float light_mis_weight = (bounce_is_specular) ? data.roughness * data.roughness : 1.0f;
            bounce_mis_weight            = 1.0f - light_mis_weight;

            store_RGBF(device.ptrs.light_records + pixel, scale_color(light_record, light_mis_weight));
            light_history_buffer_entry = light.id;
            store_trace_task(device.ptrs.light_trace + get_task_address(light_trace_count++), light_task);
          }
        }

        device.ptrs.light_sample_history[pixel] = light_history_buffer_entry;
      }

      RGBF bounce_record = mul_color(record, bounce_brdf.term);

      TraceTask bounce_task;
      bounce_task.origin = data.position;
      bounce_task.ray    = bounce_brdf.L;
      bounce_task.index  = task.index;

      if (validate_trace_task(bounce_task, pixel, bounce_record)) {
        device.ptrs.mis_buffer[pixel] = bounce_mis_weight;
        store_RGBF(device.ptrs.bounce_records + pixel, bounce_record);
        store_trace_task(device.ptrs.bounce_trace + get_task_address(bounce_trace_count++), bounce_task);
      }
    }
  }

  device.ptrs.light_trace_count[THREAD_ID]  = light_trace_count;
  device.ptrs.bounce_trace_count[THREAD_ID] = bounce_trace_count;
}

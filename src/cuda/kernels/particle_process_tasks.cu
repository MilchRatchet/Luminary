#include "brdf.cuh"
#include "directives.cuh"
#include "math.cuh"
#include "memory.cuh"
#include "particle.cuh"
#include "restir.cuh"
#include "utils.cuh"

__global__ void particle_process_tasks() {
  const int task_count   = device.ptrs.task_counts[THREAD_ID * TASK_ADDRESS_COUNT_STRIDE + TASK_ADDRESS_OFFSET_PARTICLE];
  const int task_offset  = device.ptrs.task_offsets[THREAD_ID * TASK_ADDRESS_OFFSET_STRIDE + TASK_ADDRESS_OFFSET_PARTICLE];
  int light_trace_count  = device.ptrs.light_trace_count[THREAD_ID];
  int bounce_trace_count = device.ptrs.bounce_trace_count[THREAD_ID];

  for (int i = 0; i < task_count; i++) {
    ParticleTask task = load_particle_task(device.trace_tasks + get_task_address(task_offset + i));
    const int pixel   = task.index.y * device.width + task.index.x;

    const GBufferData data = particle_generate_g_buffer(task, pixel);

    RGBF record = load_RGBF(device.records + pixel);

    write_normal_buffer(data.normal, pixel);

    const float random_choice = quasirandom_sequence_1D(QUASI_RANDOM_TARGET_BOUNCE_DIR_CHOICE, pixel);
    const float2 random_dir   = quasirandom_sequence_2D(QUASI_RANDOM_TARGET_BOUNCE_DIR, pixel);

    const vec3 bounce_ray = jendersie_eon_phase_sample(task.ray, device.scene.particles.phase_diameter, random_dir, random_choice);

    VolumeType volume_type = VOLUME_TYPE_PARTICLE;

    record = mul_color(record, opaque_color(data.albedo));

    RGBF bounce_record = record;

    TraceTask bounce_task;
    bounce_task.origin = task.position;
    bounce_task.ray    = bounce_ray;
    bounce_task.index  = task.index;

    device.ptrs.mis_buffer[pixel] = 0.0f;
    if (validate_trace_task(bounce_task, pixel, bounce_record)) {
      store_RGBF(device.ptrs.bounce_records + pixel, bounce_record);
      store_trace_task(device.ptrs.bounce_trace + get_task_address(bounce_trace_count++), bounce_task);
    }

    if (!state_peek(pixel, STATE_FLAG_LIGHT_OCCUPIED)) {
      LightSample light = restir_sample_reservoir(data, record, pixel);

      uint32_t light_history_buffer_entry = LIGHT_ID_ANY;

      if (light.weight > 0.0f) {
        BRDFInstance brdf = brdf_get_instance_scattering(scale_vector(task.ray, -1.0f));
        const BRDFInstance brdf_sample =
          brdf_apply_sample_weight_scattering(brdf_apply_sample(brdf, light, task.position, pixel), volume_type);

        const RGBF light_record = mul_color(record, brdf_sample.term);

        TraceTask light_task;
        light_task.origin = task.position;
        light_task.ray    = brdf_sample.L;
        light_task.index  = task.index;

        if (luminance(light_record) > 0.0f && state_consume(pixel, STATE_FLAG_LIGHT_OCCUPIED)) {
          store_RGBF(device.ptrs.light_records + pixel, light_record);
          light_history_buffer_entry = light.id;
          store_trace_task(device.ptrs.light_trace + get_task_address(light_trace_count++), light_task);
        }
      }

      device.ptrs.light_sample_history[pixel] = light_history_buffer_entry;
    }
  }

  device.ptrs.light_trace_count[THREAD_ID]  = light_trace_count;
  device.ptrs.bounce_trace_count[THREAD_ID] = bounce_trace_count;
}

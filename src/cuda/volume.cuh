#ifndef CU_VOLUME_H
#define CU_VOLUME_H

#include "directives.cuh"
#include "math.cuh"
#include "ocean_utils.cuh"
#include "restir.cuh"
#include "state.cuh"
#include "utils.cuh"
#include "volume_utils.cuh"

//
// This implements a volume renderer. These volumes are homogenous and bound by a disk-box (A horizontal disk with a width in the vertical
// axis). Closed form tracking is used to solve any light interaction with such a volume. While this implementation is mostly generic, some
// details are handled simply with the constraint that Luminary only has two types of volumes, fog and ocean water. Fog does not use
// absorption and only has scalar scattering values while ocean scattering and absorption is performed using three color channels.
//

////////////////////////////////////////////////////////////////////
// Literature
////////////////////////////////////////////////////////////////////

// [FonWKH17]
// J. Fong, M. Wrenninge, C. Kulla, R. Habel, "Production Volume Rendering", SIGGRAPH 2017 Course, 2017
// https://graphics.pixar.com/library/ProductionVolumeRendering/

////////////////////////////////////////////////////////////////////
// Kernel
////////////////////////////////////////////////////////////////////

__device__ GBufferData volume_generate_g_buffer(const VolumeTask task, const int pixel) {
  uint32_t flags = (!state_peek(pixel, STATE_FLAG_LIGHT_OCCUPIED)) ? G_BUFFER_REQUIRES_SAMPLING : 0;
  flags |= G_BUFFER_VOLUME_HIT;

  GBufferData data;
  data.hit_id    = task.hit_id;
  data.albedo    = RGBAF_set(0.0f, 0.0f, 0.0f, 0.0f);
  data.emission  = get_color(0.0f, 0.0f, 0.0f);
  data.normal    = get_vector(0.0f, 0.0f, 0.0f);
  data.position  = task.position;
  data.V         = scale_vector(task.ray, -1.0f);
  data.roughness = 1.0f;
  data.metallic  = 0.0f;
  data.flags     = flags;

  return data;
}

__global__ void volume_process_events() {
  const int task_count = device.trace_count[THREAD_ID];

  for (int i = 0; i < task_count; i++) {
    const int offset    = get_task_address(i);
    TraceTask task      = load_trace_task(device.trace_tasks + offset);
    const float2 result = __ldcs((float2*) (device.ptrs.trace_results + offset));
    const int pixel     = task.index.y * device.width + task.index.x;

    float depth     = result.x;
    uint32_t hit_id = __float_as_uint(result.y);

    RGBF record = load_RGBF(device.records + pixel);

    if (device.iteration_type == TYPE_LIGHT) {
      if (device.scene.fog.active) {
        const VolumeDescriptor volume = volume_get_descriptor_preset_fog();
        const float2 path             = volume_compute_path(volume, task.origin, task.ray, depth);

        if (path.x >= 0.0f) {
          record = scale_color(record, expf(-path.y * volume.max_scattering));
        }
      }

      if (device.scene.ocean.active) {
        const VolumeDescriptor volume = volume_get_descriptor_preset_ocean();
        const float2 path             = volume_compute_path(volume, task.origin, task.ray, depth);

        if (path.x >= 0.0f) {
          RGBF volume_transmittance = volume_get_transmittance(volume);

          record.r *= expf(-path.y * volume_transmittance.r);
          record.g *= expf(-path.y * volume_transmittance.g);
          record.b *= expf(-path.y * volume_transmittance.b);
        }
      }
    }
    else {
      const float random = quasirandom_sequence_1D(QUASI_RANDOM_TARGET_VOLUME_DIST, task.index);

      if (device.scene.fog.active) {
        const VolumeDescriptor volume = volume_get_descriptor_preset_fog();
        const float2 path             = volume_compute_path(volume, task.origin, task.ray, depth);

        if (path.x >= 0.0f) {
          const float volume_dist = volume_sample_intersection(volume, task.origin, task.ray, path.x, path.y, random);

          if (volume_dist < depth) {
            depth  = volume_dist;
            hit_id = HIT_TYPE_VOLUME_FOG;
          }
        }
      }

      if (device.scene.ocean.active) {
        bool ocean_intersection_possible = true;
        if (task.origin.y < OCEAN_MIN_HEIGHT || task.origin.y > OCEAN_MAX_HEIGHT) {
          const float short_distance  = ocean_short_distance(task.origin, task.ray);
          ocean_intersection_possible = (short_distance != FLT_MAX) && (short_distance <= depth);
        }

        if (ocean_intersection_possible) {
          const float ocean_depth = ocean_intersection_distance(task.origin, task.ray, depth);

          if (ocean_depth < depth) {
            depth  = ocean_depth;
            hit_id = HIT_TYPE_OCEAN;
          }
        }

        const VolumeDescriptor volume = volume_get_descriptor_preset_ocean();
        const float2 path             = volume_compute_path(volume, task.origin, task.ray, depth);

        if (path.x >= 0.0f) {
          const float volume_dist = volume_sample_intersection(volume, task.origin, task.ray, path.x, path.y, random);

          float integration_depth = path.y;

          if (volume_dist < depth) {
            depth  = volume_dist;
            hit_id = HIT_TYPE_VOLUME_OCEAN;

            integration_depth = depth - path.x;

            const float sampling_pdf = volume.max_scattering * expf(-integration_depth * volume.max_scattering);
            const RGBF target_pdf    = get_color(
              volume.scattering.r * expf(-integration_depth * volume.scattering.r),
              volume.scattering.g * expf(-integration_depth * volume.scattering.g),
              volume.scattering.b * expf(-integration_depth * volume.scattering.b));

            record = mul_color(record, scale_color(target_pdf, 1.0f / sampling_pdf));
          }

          record.r *= expf(-integration_depth * volume.absorption.r);
          record.g *= expf(-integration_depth * volume.absorption.g);
          record.b *= expf(-integration_depth * volume.absorption.b);
        }
      }

      __stcs((float2*) (device.ptrs.trace_results + offset), make_float2(depth, __uint_as_float(hit_id)));
    }

    store_RGBF(device.records + pixel, record);
  }
}

__global__ __launch_bounds__(THREADS_PER_BLOCK, 9) void volume_process_tasks() {
  const int task_count   = device.ptrs.task_counts[THREAD_ID * TASK_ADDRESS_COUNT_STRIDE + TASK_ADDRESS_OFFSET_VOLUME];
  const int task_offset  = device.ptrs.task_offsets[THREAD_ID * TASK_ADDRESS_OFFSET_STRIDE + TASK_ADDRESS_OFFSET_VOLUME];
  int light_trace_count  = device.ptrs.light_trace_count[THREAD_ID];
  int bounce_trace_count = device.ptrs.bounce_trace_count[THREAD_ID];

  for (int i = 0; i < task_count; i++) {
    VolumeTask task = load_volume_task(device.trace_tasks + get_task_address(task_offset + i));
    const int pixel = task.index.y * device.width + task.index.x;

    VolumeType volume_type = VOLUME_HIT_TYPE(task.hit_id);

    const VolumeDescriptor volume = volume_get_descriptor_preset(volume_type);

    RGBF record = device.records[pixel];

    write_albedo_buffer(get_color(0.0f, 0.0f, 0.0f), pixel);

    const float random_choice = quasirandom_sequence_1D(QUASI_RANDOM_TARGET_BOUNCE_DIR_CHOICE, task.index);
    const float2 random_dir   = quasirandom_sequence_2D(QUASI_RANDOM_TARGET_BOUNCE_DIR, task.index);

    const vec3 bounce_ray = (volume.type == VOLUME_TYPE_FOG)
                              ? jendersie_eon_phase_sample(task.ray, device.scene.fog.droplet_diameter, random_dir, random_choice)
                              : ocean_phase_sampling(task.ray, random_dir, random_choice);

    RGBF bounce_record = record;

    TraceTask bounce_task;
    bounce_task.origin = task.position;
    bounce_task.ray    = bounce_ray;
    bounce_task.index  = task.index;

    device.ptrs.mis_buffer[pixel] = 0.0f;
    if (validate_trace_task(bounce_task, bounce_record)) {
      store_RGBF(device.ptrs.bounce_records + pixel, bounce_record);
      store_trace_task(device.ptrs.bounce_trace + get_task_address(bounce_trace_count++), bounce_task);
    }

    if (!state_peek(pixel, STATE_FLAG_LIGHT_OCCUPIED)) {
      const GBufferData data = volume_generate_g_buffer(task, pixel);
      LightSample light      = restir_sample_reservoir(data, record, task.index);

      uint32_t light_history_buffer_entry = LIGHT_ID_ANY;

      BRDFInstance brdf = brdf_get_instance_scattering(scale_vector(task.ray, -1.0f));

      if (light.weight > 0.0f) {
        RGBF light_weight;
        const vec3 light_ray = restir_apply_sample_shading(data, light, task.index, light_weight);

        const RGBF light_record = mul_color(record, light_weight);

        TraceTask light_task;
        light_task.origin = task.position;
        light_task.ray    = light_ray;
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

#endif /* CU_VOLUME_H */

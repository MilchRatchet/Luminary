#ifndef CU_VOLUME_H
#define CU_VOLUME_H

#include "directives.cuh"
#include "math.cuh"
#include "ocean_utils.cuh"
#include "sky.cuh"
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

LUMINARY_KERNEL void volume_process_events() {
  HANDLE_DEVICE_ABORT();

  const int task_count = device.ptrs.trace_counts[THREAD_ID];

  uint16_t geometry_task_count = 0;
  uint16_t ocean_task_count    = 0;
  uint16_t volume_task_count   = 0;
  uint16_t particle_task_count = 0;
  uint16_t sky_task_count      = 0;

  for (int i = 0; i < task_count; i++) {
    HANDLE_DEVICE_ABORT();

    const uint32_t task_base_address = task_get_base_address(i, TASK_STATE_BUFFER_INDEX_PRESORT);
    DeviceTask task                  = task_load(task_base_address);
    const DeviceTaskTrace trace      = task_trace_load(task_base_address);

    TriangleHandle handle = trace.handle;
    float depth           = trace.depth;

    if (device.ocean.active) {
      const float ocean_depth = ocean_intersection_distance(task.origin, task.ray, depth);

      if (ocean_depth < depth) {
        depth              = ocean_depth;
        handle.instance_id = HIT_TYPE_OCEAN;
        handle.tri_id      = 0;
      }
    }

    const uint32_t pixel                  = get_pixel_id(task.index);
    const DeviceTaskThroughput throughput = task_throughput_load(task_base_address);
    RGBF record                           = record_unpack(throughput.record);

    VolumeDescriptor volume;
    volume.type = VolumeType(task.volume_id);

    if (volume.type != VOLUME_TYPE_NONE) {
      volume          = volume_get_descriptor_preset(volume.type);
      VolumePath path = volume_compute_path(volume, task.origin, task.ray, depth, true);

      float volume_intersection_probability = (task.state & STATE_FLAG_CAMERA_DIRECTION) ? 0.5f : 1.0f;

      const bool sky_fast_path = handle.instance_id == HIT_TYPE_SKY && device.sky.mode != LUMINARY_SKY_MODE_DEFAULT;

      if (sky_fast_path) {
        RGBF sky_color = sky_color_no_compute(task.ray, task.state);
        sky_color      = mul_color(sky_color, record);

        sky_color = mul_color(sky_color, volume_integrate_transmittance_precomputed(volume, path.length));

        if (device.state.depth <= 1) {
          write_beauty_buffer_direct(sky_color, pixel);
        }
        else {
          write_beauty_buffer(sky_color, pixel, task.state);
        }

        volume_intersection_probability = 1.0f;
        handle.instance_id              = HIT_TYPE_INVALID;
      }

      const float2 randoms = random_2D(RANDOM_TARGET_VOLUME_INTERSECTION, task.index);

      bool sampled_volume_intersection = false;

      float pdf = 1.0f;
      if (randoms.y < volume_intersection_probability) {
        const float volume_dist = volume_sample_intersection(volume, path.start, path.length, randoms.x);

        if (volume_dist < depth) {
          const float sample_pdf = volume_sample_intersection_pdf(volume, path.start, volume_dist);

          depth              = volume_dist;
          handle.instance_id = VOLUME_TYPE_TO_HIT(volume.type);
          handle.tri_id      = 0;

          record = mul_color(record, volume.scattering);

          pdf *= volume_intersection_probability;
          pdf *= sample_pdf;

          sampled_volume_intersection = true;

          path.length = depth - path.start;
        }
      }

      if (sampled_volume_intersection == false && sky_fast_path == false) {
        const float miss_probability = volume_sample_intersection_miss_probability(volume, path.length);

        pdf *= (1.0f - volume_intersection_probability) + volume_intersection_probability * miss_probability;
      }

      record = mul_color(record, volume_integrate_transmittance_precomputed(volume, path.length));
      record = scale_color(record, 1.0f / pdf);
    }

    task_trace_handle_store(task_base_address, handle);
    task_trace_depth_store(task_base_address, depth);
    task_throughput_record_store(task_base_address, record_pack(record));

    ////////////////////////////////////////////////////////////////////
    // Increment counts
    ////////////////////////////////////////////////////////////////////

    if (handle.instance_id == HIT_TYPE_SKY) {
      sky_task_count++;
    }
    else if (handle.instance_id == HIT_TYPE_OCEAN) {
      ocean_task_count++;
    }
    else if (VOLUME_HIT_CHECK(handle.instance_id)) {
      volume_task_count++;
    }
    else if (handle.instance_id <= HIT_TYPE_PARTICLE_MAX && handle.instance_id >= HIT_TYPE_PARTICLE_MIN) {
      particle_task_count++;
    }
    else if (handle.instance_id <= HIT_TYPE_TRIANGLE_ID_LIMIT) {
      geometry_task_count++;
    }
  }

  device.ptrs.task_counts[TASK_ADDRESS_OFFSET_GEOMETRY] = geometry_task_count;
  device.ptrs.task_counts[TASK_ADDRESS_OFFSET_OCEAN]    = ocean_task_count;
  device.ptrs.task_counts[TASK_ADDRESS_OFFSET_VOLUME]   = volume_task_count;
  device.ptrs.task_counts[TASK_ADDRESS_OFFSET_PARTICLE] = particle_task_count;
  device.ptrs.task_counts[TASK_ADDRESS_OFFSET_SKY]      = sky_task_count;
}

LUMINARY_KERNEL void volume_process_tasks() {
  HANDLE_DEVICE_ABORT();

  const int task_count  = device.ptrs.task_counts[TASK_ADDRESS_OFFSET_VOLUME];
  const int task_offset = device.ptrs.task_offsets[TASK_ADDRESS_OFFSET_VOLUME];
  int trace_count       = device.ptrs.trace_counts[THREAD_ID];

  for (int i = 0; i < task_count; i++) {
    HANDLE_DEVICE_ABORT();

    const uint32_t task_base_address      = task_get_base_address(task_offset + i, TASK_STATE_BUFFER_INDEX_POSTSORT);
    DeviceTask task                       = task_load(task_base_address);
    const DeviceTaskTrace trace           = task_trace_load(task_base_address);
    const DeviceTaskThroughput throughput = task_throughput_load(task_base_address);

    const uint32_t pixel = get_pixel_id(task.index);

    task.origin = add_vector(task.origin, scale_vector(task.ray, trace.depth));

    const VolumeType volume_type  = VOLUME_HIT_TYPE(trace.handle.instance_id);
    const VolumeDescriptor volume = volume_get_descriptor_preset(volume_type);

    MaterialContextVolume ctx = volume_get_context(task, volume, 0.0f);

    const vec3 bounce_ray = volume_sample_ray<MATERIAL_VOLUME>(ctx, task.index);

    uint16_t new_state = task.state;

    new_state &= ~STATE_FLAG_DELTA_PATH;
    new_state &= ~STATE_FLAG_CAMERA_DIRECTION;
    new_state &= ~STATE_FLAG_ALLOW_EMISSION;
    new_state &= ~STATE_FLAG_MIS_EMISSION;
    new_state &= ~STATE_FLAG_ALLOW_AMBIENT;

    new_state |= STATE_FLAG_VOLUME_SCATTERED;

    const uint32_t dst_task_base_address = task_get_base_address(trace_count++, TASK_STATE_BUFFER_INDEX_PRESORT);

    task_trace_ior_stack_store(dst_task_base_address, trace.ior_stack);
    task_throughput_record_store(dst_task_base_address, throughput.record);

    DeviceTask bounce_task;
    bounce_task.state     = new_state;
    bounce_task.origin    = ctx.position;
    bounce_task.ray       = bounce_ray;
    bounce_task.index     = task.index;
    bounce_task.volume_id = task.volume_id;

    task_store(dst_task_base_address, bounce_task);
  }

  device.ptrs.trace_counts[THREAD_ID] = trace_count;
}

#endif /* CU_VOLUME_H */

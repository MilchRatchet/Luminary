#ifndef CU_VOLUME_H
#define CU_VOLUME_H

#include "directives.cuh"
#include "math.cuh"
#include "ocean_utils.cuh"
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
    const int offset      = get_task_address(i);
    DeviceTask task       = task_load(offset);
    float depth           = trace_depth_load(offset);
    TriangleHandle handle = triangle_handle_load(offset);

    if (device.ocean.active) {
      const float ocean_depth = ocean_intersection_distance(task.origin, task.ray, depth);

      if (ocean_depth < depth) {
        depth              = ocean_depth;
        handle.instance_id = HIT_TYPE_OCEAN;
        handle.tri_id      = 0;
      }
    }

    VolumePath path = make_volume_path(FLT_MAX, 0.0f);
    VolumeDescriptor volume;
    volume.type = VOLUME_TYPE_NONE;

    if (device.fog.active) {
      const VolumeDescriptor fog_volume = volume_get_descriptor_preset_fog();
      const VolumePath fog_path         = volume_compute_path(fog_volume, task.origin, task.ray, depth);

      if (fog_path.length > 0.0f && fog_path.start < path.start) {
        volume = fog_volume;
        path   = fog_path;
      }
    }

    if (device.ocean.active) {
      const VolumeDescriptor fog_volume = volume_get_descriptor_preset_ocean();
      const VolumePath fog_path         = volume_compute_path(fog_volume, task.origin, task.ray, depth);

      if (fog_path.length > 0.0f && fog_path.start < path.start) {
        volume = fog_volume;
        path   = fog_path;
      }
    }

    const uint32_t pixel = get_pixel_id(task.index);
    RGBF record          = load_RGBF(device.ptrs.records + pixel);

    if (volume.type != VOLUME_TYPE_NONE) {
      const float choice_random = quasirandom_sequence_1D(QUASI_RANDOM_TARGET_VOLUME_STRATEGY, task.index);

      float pdf = 0.5f;
      if (choice_random < 0.5f) {
        const float random = quasirandom_sequence_1D(QUASI_RANDOM_TARGET_VOLUME_DIST, task.index);

        const float volume_dist = volume_sample_intersection(volume, path.start, path.length, random);

        if (volume_dist < depth) {
          depth              = volume_dist;
          handle.instance_id = VOLUME_TYPE_TO_HIT(volume.type);
          handle.tri_id      = 0;

          record = mul_color(record, volume.scattering);

          pdf *= volume_sample_intersection_pdf(volume, path.start, volume_dist);
        }
        else {
          handle.instance_id = HIT_TYPE_INVALID;
        }
      }

      record = mul_color(record, volume_integrate_transmittance_precomputed(volume, path.start, depth));

      record = scale_color(record, 1.0f / pdf);
    }

    trace_depth_store(depth, offset);
    triangle_handle_store(handle, offset);
    store_RGBF(device.ptrs.records, pixel, record);

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
    const uint32_t offset       = get_task_address(task_offset + i);
    DeviceTask task             = task_load(offset);
    const TriangleHandle handle = triangle_handle_load(offset);
    const float depth           = trace_depth_load(offset);
    const uint32_t pixel        = get_pixel_id(task.index);

    task.origin = add_vector(task.origin, scale_vector(task.ray, depth));

    const VolumeType volume_type  = VOLUME_HIT_TYPE(handle.instance_id);
    const VolumeDescriptor volume = volume_get_descriptor_preset(volume_type);

    GBufferData data = volume_generate_g_buffer(task, handle.instance_id, pixel, volume);

    const vec3 bounce_ray = bsdf_sample_volume(data, task.index);

    uint8_t new_state = task.state & ~(STATE_FLAG_DELTA_PATH | STATE_FLAG_CAMERA_DIRECTION | STATE_FLAG_ALLOW_EMISSION);

    new_state |= STATE_FLAG_VOLUME_SCATTERED;

    DeviceTask bounce_task;
    bounce_task.state  = new_state;
    bounce_task.origin = data.position;
    bounce_task.ray    = bounce_ray;
    bounce_task.index  = task.index;

    task_store(bounce_task, get_task_address(trace_count++));
  }

  device.ptrs.trace_counts[THREAD_ID] = trace_count;
}

#endif /* CU_VOLUME_H */

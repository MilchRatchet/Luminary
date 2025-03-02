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

  for (int i = 0; i < task_count; i++) {
    const int offset      = get_task_address(i);
    DeviceTask task       = task_load(offset);
    float depth           = trace_depth_load(offset);
    TriangleHandle handle = triangle_handle_load(offset);

    const uint32_t pixel = get_pixel_id(task.index);

    RGBF record = load_RGBF(device.ptrs.records + pixel);

    const float random = quasirandom_sequence_1D(QUASI_RANDOM_TARGET_VOLUME_DIST, task.index);

    if (device.fog.active) {
      const VolumeDescriptor volume = volume_get_descriptor_preset_fog();
      const float2 path             = volume_compute_path(volume, task.origin, task.ray, depth);

      if (path.x >= 0.0f) {
        const float volume_dist = volume_sample_intersection(volume, path.x, path.y, random);

        if (volume_dist < depth) {
          depth              = volume_dist;
          handle.instance_id = HIT_TYPE_VOLUME_FOG;
          handle.tri_id      = 0;
        }
      }
    }

    if (device.ocean.active) {
      const float ocean_depth = ocean_intersection_distance(task.origin, task.ray, depth);

      if (ocean_depth < depth) {
        depth              = ocean_depth;
        handle.instance_id = HIT_TYPE_OCEAN;
        handle.tri_id      = 0;
      }

      const VolumeDescriptor volume = volume_get_descriptor_preset_ocean();
      const float2 path             = volume_compute_path(volume, task.origin, task.ray, depth);

      if (path.x >= 0.0f) {
        float integration_depth = path.y;

        const bool allow_ocean_volume_hit = device.ocean.multiscattering || !(task.state & STATE_FLAG_OCEAN_SCATTERED);

        if (allow_ocean_volume_hit) {
          const float volume_dist = volume_sample_intersection(volume, path.x, path.y, random);

          if (volume_dist < depth) {
            depth              = volume_dist;
            handle.instance_id = HIT_TYPE_VOLUME_OCEAN;
            handle.tri_id      = 0;

            integration_depth = depth - path.x;

            const float sampling_pdf = volume.max_scattering * expf(-integration_depth * volume.max_scattering);
            record                   = mul_color(record, scale_color(volume.scattering, 1.0f / sampling_pdf));
          }
        }

        record.r *= expf(-integration_depth * (volume.absorption.r + volume.scattering.r));
        record.g *= expf(-integration_depth * (volume.absorption.g + volume.scattering.g));
        record.b *= expf(-integration_depth * (volume.absorption.b + volume.scattering.b));
      }
    }

    trace_depth_store(depth, offset);
    triangle_handle_store(handle, offset);
    store_RGBF(device.ptrs.records, pixel, record);
  }
}

#endif /* CU_VOLUME_H */

#ifndef CU_VOLUME_H
#define CU_VOLUME_H

#include "directives.cuh"
#include "math.cuh"
#include "ocean_utils.cuh"
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

LUMINARY_KERNEL void volume_process_events() {
  const int task_count = device.ptrs.trace_counts[THREAD_ID];

  for (int i = 0; i < task_count; i++) {
    const int offset    = get_task_address(i);
    TraceTask task      = load_trace_task(device.ptrs.trace_tasks + offset);
    const float2 result = __ldcs((float2*) (device.ptrs.trace_results + offset));
    const int pixel     = task.index.y * device.width + task.index.x;

    float depth     = result.x;
    uint32_t hit_id = __float_as_uint(result.y);

    RGBF record = load_RGBF(device.ptrs.records + pixel);

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

    store_RGBF(device.ptrs.records + pixel, record);
  }
}

#endif /* CU_VOLUME_H */

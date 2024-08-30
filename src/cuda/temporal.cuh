#ifndef CU_TEMPORAL_H
#define CU_TEMPORAL_H

#include "structs.h"
#include "utils.cuh"

// TODO: Refactor this into a separate file, the idea of "temporal" is long gone now that reprojection is gone. While we are at it, get rid
// of the legacy naming of "temporal frames".

__device__ RGBF temporal_reject_invalid_sample(RGBF sample) {
  if (isnan(luminance(sample)) || isinf(luminance(sample))) {
    // Debug code to identify paths that cause NaNs and INFs
#if 0
      ushort2 pixel;
      pixel.y = (uint16_t) (offset / device.width);
      pixel.x = (uint16_t) (offset - pixel.y * device.width);
      printf(
        "Path at (%u, %u) on frame %u ran into a NaN or INF: (%f %f %f)\n", pixel.x, pixel.y, device.temporal_frames, sample.r, sample.g,
        sample.b);
#endif

    sample = get_color(0.0f, 0.0f, 0.0f);
  }

  return sample;
}

LUMINARY_KERNEL void temporal_accumulation() {
  const int amount = device.width * device.height;

  for (int offset = THREAD_ID; offset < amount; offset += blockDim.x * gridDim.x) {
    // Direct Lighting
    RGBF direct_buffer = load_RGBF(device.ptrs.frame_direct_buffer + offset);
    RGBF direct_output = load_RGBF(device.ptrs.frame_direct_accumulate + offset);

    direct_buffer = temporal_reject_invalid_sample(direct_buffer);

    direct_output = scale_color(direct_output, device.temporal_frames);
    direct_output = add_color(direct_buffer, direct_output);
    direct_output = scale_color(direct_output, 1.0f / (device.temporal_frames + 1));

    store_RGBF(device.ptrs.frame_direct_accumulate + offset, direct_output);

    // Indirect Lighting
    RGBF indirect_buffer = load_RGBF(device.ptrs.frame_indirect_buffer + offset);
    RGBF indirect_output = load_RGBF(device.ptrs.frame_indirect_accumulate + offset);

    indirect_buffer = temporal_reject_invalid_sample(indirect_buffer);

    float variance = (device.temporal_frames == 0) ? 1.0f : __ldcs(device.ptrs.frame_variance + offset);

    if (device.scene.camera.do_firefly_clamping) {
      float luminance_buffer = color_importance(indirect_buffer);
      float luminance_output = color_importance(indirect_output);

      const float deviation = fminf(0.1f, sqrtf(fmaxf(variance, eps)));

      if (device.temporal_frames) {
        variance *= device.temporal_frames - 1.0f;

        float diff = luminance_buffer - luminance_output;
        diff       = diff * diff;

        // Hard firefly rejection.
        // Fireflies that appear during the first frame are accepted by our method since there is
        // no reference yet to reject them from. This trick here hard rejects them by taking the
        // dimmer of the two frames. This is not unbiased but since it only happens exactly once
        // the bias will decrease with the number of frames.
        // Taking neighbouring pixels as reference is not the target since I want to consider each
        // pixel as its own independent entity to preserve fine details.
        // TODO: Improve this method to remove the visible dimming during the second frame.
        if (device.temporal_frames == 1) {
          float min_luminance = fminf(luminance_buffer, luminance_output);

          indirect_output =
            (luminance_output > eps) ? scale_color(indirect_output, min_luminance / luminance_output) : get_color(0.0f, 0.0f, 0.0f);
          indirect_buffer =
            (luminance_buffer > eps) ? scale_color(indirect_buffer, min_luminance / luminance_buffer) : get_color(0.0f, 0.0f, 0.0f);
        }

        variance += diff;
        variance *= 1.0f / device.temporal_frames;
      }

      __stcs(device.ptrs.frame_variance + offset, variance);

      const float firefly_rejection    = 0.1f + luminance_output + deviation * 4.0f;
      const float new_luminance_buffer = fminf(luminance_buffer, firefly_rejection);

      indirect_buffer =
        (luminance_buffer > eps) ? scale_color(indirect_buffer, new_luminance_buffer / luminance_buffer) : get_color(0.0f, 0.0f, 0.0f);
    }

    indirect_output = scale_color(indirect_output, device.temporal_frames);
    indirect_output = add_color(indirect_buffer, indirect_output);
    indirect_output = scale_color(indirect_output, 1.0f / (device.temporal_frames + 1));

    store_RGBF(device.ptrs.frame_indirect_accumulate + offset, indirect_output);

    RGBF output = add_color(direct_output, indirect_output);

    store_RGBF(device.ptrs.frame_accumulate + offset, output);
  }
}

LUMINARY_KERNEL void temporal_accumulation_aov(const RGBF* buffer, RGBF* accumulate) {
  const int amount = device.width * device.height;

  for (int offset = THREAD_ID; offset < amount; offset += blockDim.x * gridDim.x) {
    RGBF input  = load_RGBF(buffer + offset);
    RGBF output = (device.temporal_frames == 0) ? input : load_RGBF(accumulate + offset);

    output = scale_color(output, device.temporal_frames);
    output = add_color(input, output);
    output = scale_color(output, 1.0f / (device.temporal_frames + 1));

    store_RGBF(accumulate + offset, output);
  }
}

#endif /* CU_TEMPORAL_H */

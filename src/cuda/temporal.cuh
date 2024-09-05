#ifndef CU_TEMPORAL_H
#define CU_TEMPORAL_H

#include "math.cuh"
#include "structs.h"
#include "utils.cuh"

// TODO: Refactor this into a separate file, the idea of "temporal" is long gone now that reprojection is gone. While we are at it, get rid
// of the legacy naming of "temporal frames".

// This is a gaussian filter with a sigma of 0.4 and a radius of 2. I cut out a lot of the terms
// because they were negligible.
__device__ float temporal_gather_pixel_weight(const RGBF pixel, const float x, const float y) {
  const float weight_x = expf(-(x * x) * 3.125f);
  const float weight_y = expf(-(y * y) * 3.125f);

  return weight_x * weight_y;
}

__device__ RGBF temporal_gather_pixel_load(
  const RGBF* image, const int width, const int height, const float sx, const float sy, const int i, const int j) {
  const int index_x = ((int) sx) + i;
  const int index_y = ((int) sy) + j;

  if (index_x < 0 || index_x >= width || index_y < 0 || index_y >= height)
    return get_color(0.0f, 0.0f, 0.0f);

  const int index = index_x + index_y * width;

  const RGBF pixel = load_RGBF(image + index);

  const float rx = fabsf(sx - index_x);
  const float ry = fabsf(sy - index_y);

  return scale_color(pixel, temporal_gather_pixel_weight(pixel, rx, ry));
}

__device__ RGBF temporal_gather_pixel(const RGBF* image, const float x, const float y, const int width, const int height) {
  RGBF result = get_color(0.0f, 0.0f, 0.0f);

  for (int j = -1; j <= 2; j++) {
    for (int i = -1; i <= 2; i++) {
      result = add_color(result, temporal_gather_pixel_load(image, width, height, x, y, i, j));
    }
  }

  return result;
}

__device__ RGBF temporal_reject_invalid_sample(RGBF sample, const uint32_t offset) {
  if (isnan(luminance(sample)) || isinf(luminance(sample))) {
    // Debug code to identify paths that cause NaNs and INFs
#if 0
      ushort2 pixel;
      pixel.y = (uint16_t) (offset / device.internal_width);
      pixel.x = (uint16_t) (offset - pixel.y * device.internal_width);
      printf(
        "Path at (%u, %u) on frame %u ran into a NaN or INF: (%f %f %f)\n", pixel.x, pixel.y, (uint32_t) device.temporal_frames, sample.r, sample.g,
        sample.b);
#endif

    sample = UTILS_DEBUG_NAN_COLOR;
  }

  return sample;
}

__device__ float temporal_increment() {
  const uint32_t undersampling_scale = (1 << device.undersampling) * (1 << device.undersampling);

  return 1.0f / undersampling_scale;
}

LUMINARY_KERNEL void temporal_accumulation() {
  const uint32_t amount = device.internal_width * device.internal_height;

  const float2 jitter = quasirandom_sequence_2D_global(QUASI_RANDOM_TARGET_CAMERA_JITTER);

  const float increment = temporal_increment();

  for (uint32_t offset = THREAD_ID; offset < amount; offset += blockDim.x * gridDim.x) {
    const uint32_t y = offset / device.internal_width;
    const uint32_t x = offset - y * device.internal_width;

    const float sx = x + jitter.x;
    const float sy = y + jitter.y;

    // Direct Lighting
    RGBF direct_buffer = temporal_gather_pixel(device.ptrs.frame_direct_buffer, sx, sy, device.internal_width, device.internal_height);
    RGBF direct_output = load_RGBF(device.ptrs.frame_direct_accumulate + offset);

    direct_buffer = temporal_reject_invalid_sample(direct_buffer, offset);

    direct_output = scale_color(direct_output, ceilf(device.temporal_frames));
    direct_output = add_color(direct_buffer, direct_output);
    direct_output = scale_color(direct_output, 1.0f / (device.temporal_frames + increment));

    store_RGBF(device.ptrs.frame_direct_accumulate + offset, direct_output);

    // Indirect Lighting
    RGBF indirect_buffer = temporal_gather_pixel(device.ptrs.frame_indirect_buffer, sx, sy, device.internal_width, device.internal_height);
    RGBF indirect_output = load_RGBF(device.ptrs.frame_indirect_accumulate + offset);

    indirect_buffer = temporal_reject_invalid_sample(indirect_buffer, offset);

    float variance = (device.temporal_frames == 0.0f) ? 1.0f : __ldcs(device.ptrs.frame_variance + offset);

#if 0
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
        // TODO: Adapt to undersampling.
#if 0
        if (device.temporal_frames == 1) {
          float min_luminance = fminf(luminance_buffer, luminance_output);

          indirect_output =
            (luminance_output > eps) ? scale_color(indirect_output, min_luminance / luminance_output) : get_color(0.0f, 0.0f, 0.0f);
          indirect_buffer =
            (luminance_buffer > eps) ? scale_color(indirect_buffer, min_luminance / luminance_buffer) : get_color(0.0f, 0.0f, 0.0f);
        }
#endif

        variance += diff;
        variance *= 1.0f / device.temporal_frames;
      }

      __stcs(device.ptrs.frame_variance + offset, variance);

      const float firefly_rejection    = 0.1f + luminance_output + deviation * 4.0f;
      const float new_luminance_buffer = fminf(luminance_buffer, firefly_rejection);

      indirect_buffer =
        (luminance_buffer > eps) ? scale_color(indirect_buffer, new_luminance_buffer / luminance_buffer) : get_color(0.0f, 0.0f, 0.0f);
    }
#endif

    indirect_output = scale_color(indirect_output, ceilf(device.temporal_frames));
    indirect_output = add_color(indirect_buffer, indirect_output);
    indirect_output = scale_color(indirect_output, 1.0f / (device.temporal_frames + increment));

    store_RGBF(device.ptrs.frame_indirect_accumulate + offset, indirect_output);

    RGBF output = add_color(direct_output, indirect_output);

    store_RGBF(device.ptrs.frame_accumulate + offset, output);
  }
}

LUMINARY_KERNEL void temporal_accumulation_aov(const RGBF* buffer, RGBF* accumulate) {
  const uint32_t amount = device.internal_width * device.internal_height;

  const float increment = temporal_increment();

  for (uint32_t offset = THREAD_ID; offset < amount; offset += blockDim.x * gridDim.x) {
    RGBF input  = load_RGBF(buffer + offset);
    RGBF output = (device.temporal_frames == 0.0f) ? input : load_RGBF(accumulate + offset);

    output = scale_color(output, ceilf(device.temporal_frames));
    output = add_color(input, output);
    output = scale_color(output, 1.0f / (device.temporal_frames + increment));

    store_RGBF(accumulate + offset, output);
  }
}

#endif /* CU_TEMPORAL_H */

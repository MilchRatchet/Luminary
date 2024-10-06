#ifndef CU_TEMPORAL_H
#define CU_TEMPORAL_H

#include "math.cuh"
#include "structs.h"
#include "utils.cuh"

// TODO: Refactor this into a separate file, the idea of "temporal" is long gone now that reprojection is gone. While we are at it, get rid
// of the legacy naming of "temporal frames".

// Simple tent filter.
__device__ float temporal_gather_pixel_weight(const float x, const float y) {
  return (1.0f - x) * (1.0f - y);
}

__device__ RGBF temporal_gather_pixel_load(
  const RGBF* image, const uint32_t width, const uint32_t height, const float pixel_x, const float pixel_y, const float sample_x,
  const float sample_y) {
  const uint32_t index_x = (uint32_t) max(min((int32_t) sample_x, width - 1), 0);
  const uint32_t index_y = (uint32_t) max(min((int32_t) sample_y, height - 1), 0);

  const uint32_t index = index_x + index_y * width;

  const RGBF pixel = load_RGBF(image + index);

  const float rx = fabsf(pixel_x - sample_x);
  const float ry = fabsf(pixel_y - sample_y);

  return scale_color(pixel, temporal_gather_pixel_weight(rx, ry));
}

__device__ RGBF temporal_gather_pixel(
  const RGBF* image, const float pixel_x, const float pixel_y, const float base_x, const float base_y, const uint32_t width,
  const uint32_t height) {
  RGBF result = get_color(0.0f, 0.0f, 0.0f);

  result = add_color(result, temporal_gather_pixel_load(image, width, height, pixel_x, pixel_y, base_x, base_y));
  result = add_color(result, temporal_gather_pixel_load(image, width, height, pixel_x, pixel_y, base_x + 1.0f, base_y));
  result = add_color(result, temporal_gather_pixel_load(image, width, height, pixel_x, pixel_y, base_x, base_y + 1.0f));
  result = add_color(result, temporal_gather_pixel_load(image, width, height, pixel_x, pixel_y, base_x + 1.0f, base_y + 1.0f));

  return result;
}

__device__ RGBF temporal_reject_invalid_sample(RGBF sample, const uint32_t offset) {
  if (isnan(luminance(sample)) || isinf(luminance(sample))) {
    // Debug code to identify paths that cause NaNs and INFs
#if 0
      ushort2 pixel;
      pixel.y = (uint16_t) (offset / device.settings.width);
      pixel.x = (uint16_t) (offset - pixel.y * device.settings.width);
      printf(
        "Path at (%u, %u) on frame %u ran into a NaN or INF: (%f %f %f)\n", pixel.x, pixel.y, (uint32_t) device.sample_id, sample.r, sample.g,
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
  const uint32_t amount = device.settings.width * device.settings.height;

  const float2 jitter = quasirandom_sequence_2D_global(QUASI_RANDOM_TARGET_CAMERA_JITTER);

  const float increment = temporal_increment();

  const bool load_accumulate = (device.sample_id >= 1.0f);
  const float prev_scale     = device.sample_id;
  const float curr_inv_scale = 1.0f / fmaxf(1.0f, device.sample_id + increment);

  for (uint32_t offset = THREAD_ID; offset < amount; offset += blockDim.x * gridDim.x) {
    const uint32_t y = offset / device.settings.width;
    const uint32_t x = offset - y * device.settings.width;

    const float pixel_x = x + 0.5f;
    const float pixel_y = y + 0.5f;

    const float base_x = floorf(x - (jitter.x - 0.5f)) + jitter.x;
    const float base_y = floorf(y - (jitter.y - 0.5f)) + jitter.y;

    // Direct Lighting
    RGBF direct_buffer = temporal_gather_pixel(
      device.ptrs.frame_direct_buffer, pixel_x, pixel_y, base_x, base_y, device.settings.width, device.settings.height);
    RGBF direct_output = (load_accumulate) ? load_RGBF(device.ptrs.frame_direct_accumulate + offset) : get_color(0.0f, 0.0f, 0.0f);

    direct_buffer = temporal_reject_invalid_sample(direct_buffer, offset);

    direct_output = scale_color(direct_output, prev_scale);
    direct_output = add_color(direct_buffer, direct_output);
    direct_output = scale_color(direct_output, curr_inv_scale);

    store_RGBF(device.ptrs.frame_direct_accumulate + offset, direct_output);

    // Indirect Lighting
    RGBF indirect_buffer = temporal_gather_pixel(
      device.ptrs.frame_indirect_buffer, pixel_x, pixel_y, base_x, base_y, device.settings.width, device.settings.height);
    RGBF indirect_output = (load_accumulate) ? load_RGBF(device.ptrs.frame_indirect_accumulate + offset) : get_color(0.0f, 0.0f, 0.0f);

    indirect_buffer = temporal_reject_invalid_sample(indirect_buffer, offset);

    // Firefly clamping only takes effect after undersampling has stopped.
    if (device.camera.do_firefly_clamping && device.sample_id >= 1.0f) {
      float variance = (device.sample_id < 2.0f) ? 1.0f : __ldcs(device.ptrs.frame_variance + offset);

      float luminance_buffer = color_importance(indirect_buffer);
      float luminance_output = color_importance(indirect_output);

      const float deviation = fminf(0.1f, sqrtf(fmaxf(variance, eps)));

      variance *= device.sample_id - 1.0f;

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
      if (device.sample_id < 2.0f) {
        const float min_luminance = fminf(luminance_buffer, luminance_output);

        indirect_output =
          (luminance_output > eps) ? scale_color(indirect_output, min_luminance / luminance_output) : get_color(0.0f, 0.0f, 0.0f);
        indirect_buffer =
          (luminance_buffer > eps) ? scale_color(indirect_buffer, min_luminance / luminance_buffer) : get_color(0.0f, 0.0f, 0.0f);
      }

      variance += diff;
      variance *= 1.0f / device.sample_id;

      __stcs(device.ptrs.frame_variance + offset, variance);

      const float firefly_rejection    = 0.1f + luminance_output + deviation * 4.0f;
      const float new_luminance_buffer = fminf(luminance_buffer, firefly_rejection);

      indirect_buffer =
        (luminance_buffer > eps) ? scale_color(indirect_buffer, new_luminance_buffer / luminance_buffer) : get_color(0.0f, 0.0f, 0.0f);
    }

    indirect_output = scale_color(indirect_output, prev_scale);
    indirect_output = add_color(indirect_buffer, indirect_output);
    indirect_output = scale_color(indirect_output, curr_inv_scale);

    store_RGBF(device.ptrs.frame_indirect_accumulate + offset, indirect_output);

    RGBF output = add_color(direct_output, indirect_output);

    store_RGBF(device.ptrs.frame_accumulate + offset, output);
  }
}

LUMINARY_KERNEL void temporal_accumulation_aov(const RGBF* buffer, RGBF* accumulate) {
  const uint32_t amount = device.settings.width * device.settings.height;

  const float increment = temporal_increment();

  for (uint32_t offset = THREAD_ID; offset < amount; offset += blockDim.x * gridDim.x) {
    RGBF input  = load_RGBF(buffer + offset);
    RGBF output = (device.sample_id == 0.0f) ? input : load_RGBF(accumulate + offset);

    output = scale_color(output, ceilf(device.sample_id));
    output = add_color(input, output);
    output = scale_color(output, 1.0f / (device.sample_id + increment));

    store_RGBF(accumulate + offset, output);
  }
}

#endif /* CU_TEMPORAL_H */

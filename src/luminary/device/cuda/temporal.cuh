#ifndef CU_TEMPORAL_H
#define CU_TEMPORAL_H

#include "math.cuh"
#include "utils.cuh"

// TODO: Refactor this into a separate file, the idea of "temporal" is long gone now that reprojection is gone. While we are at it, get rid
// of the legacy naming of "temporal frames".

// Simple tent filter.
__device__ float temporal_gather_pixel_weight(const float x, const float y) {
  return (1.0f - x) * (1.0f - y);
}

__device__ RGBF temporal_gather_pixel_load(
  const RGB_E6M20* image, const uint32_t width, const uint32_t height, const float pixel_x, const float pixel_y, const float sample_x,
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
  const RGB_E6M20* image, const float pixel_x, const float pixel_y, const float base_x, const float base_y, const uint32_t width,
  const uint32_t height) {
  RGBF result = get_color(0.0f, 0.0f, 0.0f);

  result = add_color(result, temporal_gather_pixel_load(image, width, height, pixel_x, pixel_y, base_x, base_y));
  result = add_color(result, temporal_gather_pixel_load(image, width, height, pixel_x, pixel_y, base_x + 1.0f, base_y));
  result = add_color(result, temporal_gather_pixel_load(image, width, height, pixel_x, pixel_y, base_x, base_y + 1.0f));
  result = add_color(result, temporal_gather_pixel_load(image, width, height, pixel_x, pixel_y, base_x + 1.0f, base_y + 1.0f));

  return result;
}

LUMINARY_KERNEL void temporal_accumulation_first_sample() {
  HANDLE_DEVICE_ABORT();

  const uint32_t amount = device.settings.width * device.settings.height;

  const float2 jitter = quasirandom_sequence_2D_global(QUASI_RANDOM_TARGET_CAMERA_JITTER);

  const uint32_t bucket_id       = device.state.sample_id % MAX_NUM_INDIRECT_BUCKETS;
  RGB_E6M20* indirect_bucket_ptr = device.ptrs.frame_indirect_accumulate[bucket_id];

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

    store_RGBF(device.ptrs.frame_direct_accumulate, offset, direct_buffer);

    // Indirect Lighting
    RGBF indirect_buffer = temporal_gather_pixel(
      device.ptrs.frame_indirect_buffer, pixel_x, pixel_y, base_x, base_y, device.settings.width, device.settings.height);

    store_RGBF(indirect_bucket_ptr, offset, indirect_buffer);

    RGBF output = add_color(direct_buffer, indirect_buffer);

    store_RGBF(device.ptrs.frame_current_result, offset, output);
  }
}

__device__ uint32_t temporal_get_bucket_sample_count(const uint32_t bucket_id) {
  const uint32_t overall_sample_count           = device.state.sample_id / MAX_NUM_INDIRECT_BUCKETS;
  const uint32_t buckets_with_additional_sample = device.state.sample_id - overall_sample_count * MAX_NUM_INDIRECT_BUCKETS;

  return overall_sample_count + ((bucket_id < buckets_with_additional_sample) ? 1 : 0);
}

LUMINARY_KERNEL void temporal_accumulation_update() {
  HANDLE_DEVICE_ABORT();

  const uint32_t amount = device.settings.width * device.settings.height;

  const float2 jitter = quasirandom_sequence_2D_global(QUASI_RANDOM_TARGET_CAMERA_JITTER);

  const float prev_scale     = device.state.sample_id;
  const float curr_inv_scale = 1.0f / (device.state.sample_id + 1.0f);

  const uint32_t bucket_id       = device.state.sample_id % MAX_NUM_INDIRECT_BUCKETS;
  RGB_E6M20* indirect_bucket_ptr = device.ptrs.frame_indirect_accumulate[bucket_id];

  const uint32_t bucket_sample_count = temporal_get_bucket_sample_count(bucket_id);

  const float prev_indirect_scale     = bucket_sample_count;
  const float curr_indirect_inv_scale = 1.0f / (bucket_sample_count + 1.0f);

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
    RGBF direct_output = load_RGBF(device.ptrs.frame_direct_accumulate + offset);

    direct_output = scale_color(direct_output, prev_scale);
    direct_output = add_color(direct_output, direct_buffer);
    direct_output = scale_color(direct_output, curr_inv_scale);

    store_RGBF(device.ptrs.frame_direct_accumulate, offset, direct_output);

    // Indirect Lighting
    RGBF indirect_buffer = temporal_gather_pixel(
      device.ptrs.frame_indirect_buffer, pixel_x, pixel_y, base_x, base_y, device.settings.width, device.settings.height);
    RGBF indirect_output = load_RGBF(indirect_bucket_ptr + offset);

    indirect_output = scale_color(indirect_output, prev_indirect_scale);
    indirect_output = add_color(indirect_output, indirect_buffer);
    indirect_output = scale_color(indirect_output, curr_indirect_inv_scale);

    store_RGBF(indirect_bucket_ptr, offset, indirect_output);
  }
}

__device__ void temporal_load_buckets(
  uint32_t bucket_id, uint32_t x, uint32_t y, float red[9 * MAX_NUM_INDIRECT_BUCKETS], float green[9 * MAX_NUM_INDIRECT_BUCKETS],
  float blue[9 * MAX_NUM_INDIRECT_BUCKETS], uint32_t& num_buckets) {
  if (device.state.sample_id > bucket_id) {
    const RGB_E6M20* src = device.ptrs.frame_indirect_accumulate[bucket_id];

    if (y) {
      if (x) {
        const RGBF color00 = load_RGBF(src + (x - 1) + (y - 1) * device.settings.width);

        red[num_buckets]   = color00.r;
        green[num_buckets] = color00.g;
        blue[num_buckets]  = color00.b;
        num_buckets++;
      }

      const RGBF color01 = load_RGBF(src + x + (y - 1) * device.settings.width);

      red[num_buckets]   = color01.r;
      green[num_buckets] = color01.g;
      blue[num_buckets]  = color01.b;
      num_buckets++;

      if (x + 1 != device.settings.width) {
        const RGBF color02 = load_RGBF(src + (x + 1) + (y - 1) * device.settings.width);

        red[num_buckets]   = color02.r;
        green[num_buckets] = color02.g;
        blue[num_buckets]  = color02.b;
        num_buckets++;
      }
    }

    if (x) {
      const RGBF color10 = load_RGBF(src + (x - 1) + y * device.settings.width);

      red[num_buckets]   = color10.r;
      green[num_buckets] = color10.g;
      blue[num_buckets]  = color10.b;
      num_buckets++;
    }

    const RGBF color11 = load_RGBF(src + x + y * device.settings.width);

    red[num_buckets]   = color11.r;
    green[num_buckets] = color11.g;
    blue[num_buckets]  = color11.b;
    num_buckets++;

    if (x + 1 != device.settings.width) {
      const RGBF color12 = load_RGBF(src + (x + 1) + y * device.settings.width);

      red[num_buckets]   = color12.r;
      green[num_buckets] = color12.g;
      blue[num_buckets]  = color12.b;
      num_buckets++;
    }

    if (y + 1 != device.settings.height) {
      if (x) {
        const RGBF color20 = load_RGBF(src + (x - 1) + (y + 1) * device.settings.width);

        red[num_buckets]   = color20.r;
        green[num_buckets] = color20.g;
        blue[num_buckets]  = color20.b;
        num_buckets++;
      }

      const RGBF color21 = load_RGBF(src + x + (y + 1) * device.settings.width);

      red[num_buckets]   = color21.r;
      green[num_buckets] = color21.g;
      blue[num_buckets]  = color21.b;
      num_buckets++;

      if (x + 1 != device.settings.width) {
        const RGBF color22 = load_RGBF(src + (x + 1) + (y + 1) * device.settings.width);

        red[num_buckets]   = color22.r;
        green[num_buckets] = color22.g;
        blue[num_buckets]  = color22.b;
        num_buckets++;
      }
    }
  }
}

__device__ float temporal_apply_median_of_means(float buckets[], const uint32_t num_buckets) {
  // Sort
  {
    uint32_t i = 1;

    while (i < num_buckets) {
      const float x = buckets[i];
      uint32_t j    = i;
      while (j > 0 && buckets[j - 1] > x) {
        buckets[j] = buckets[j - 1];
        j--;
      }

      buckets[j] = x;
      i++;
    }
  }

  float output;

#if 1

  // Gini - Median of Means

  float num   = 0.0f;
  float denom = 0.0f;

  for (uint32_t bucket_id = 0; bucket_id < num_buckets; bucket_id++) {
    const float value = buckets[bucket_id];
    num += bucket_id * value;
    denom += value;
  }

  num *= 2.0f;
  denom *= num_buckets;

  const float G = (num / denom) - (num_buckets + 1.0f) / num_buckets;

  const uint32_t k = num_buckets >> 1;
  const uint32_t c = k - (1.0f - G) * k;

  output = 0.0f;

  for (uint32_t bucket_id = c; bucket_id < num_buckets - c; bucket_id++) {
    output += buckets[bucket_id];
  }

  output /= num_buckets - 2 * c;

#else

  // Median of Means
  output = buckets[num_buckets >> 1];

#endif

  return output;
}

LUMINARY_KERNEL void temporal_accumulation_output() {
  HANDLE_DEVICE_ABORT();

  const uint32_t width  = device.settings.width;
  const uint32_t height = device.settings.height;

  const uint32_t amount = width * height;

  for (uint32_t offset = THREAD_ID; offset < amount; offset += blockDim.x * gridDim.x) {
    const uint32_t y = offset / width;
    const uint32_t x = offset - y * width;

    float red[9 * MAX_NUM_INDIRECT_BUCKETS];
    float green[9 * MAX_NUM_INDIRECT_BUCKETS];
    float blue[9 * MAX_NUM_INDIRECT_BUCKETS];
    uint32_t num_buckets = 0;

    temporal_load_buckets(0, x, y, red, green, blue, num_buckets);
    temporal_load_buckets(1, x, y, red, green, blue, num_buckets);
    temporal_load_buckets(2, x, y, red, green, blue, num_buckets);

    RGBF output;
    output.r = temporal_apply_median_of_means(red, num_buckets);
    output.g = temporal_apply_median_of_means(green, num_buckets);
    output.b = temporal_apply_median_of_means(blue, num_buckets);

    if (device.camera.indirect_only == false) {
      const RGBF direct = load_RGBF(device.ptrs.frame_direct_accumulate + x + y * device.settings.width);

      output = add_color(output, direct);
    }

    store_RGBF(device.ptrs.frame_current_result, x + y * device.settings.width, output);
  }
}

LUMINARY_KERNEL void temporal_accumulation_output_raw() {
  HANDLE_DEVICE_ABORT();

  const uint32_t width  = device.settings.width;
  const uint32_t height = device.settings.height;

  const uint32_t amount = width * height;

  const uint32_t num_buckets = min(MAX_NUM_INDIRECT_BUCKETS, device.state.sample_id + 1);
  const float bucket_norm    = 1.0f / num_buckets;

  for (uint32_t offset = THREAD_ID; offset < amount; offset += blockDim.x * gridDim.x) {
    const uint32_t y = offset / width;
    const uint32_t x = offset - y * width;

    RGBF output = splat_color(0.0f);

    for (uint32_t bucket_id = 0; bucket_id < num_buckets; bucket_id++) {
      output = add_color(output, load_RGBF(device.ptrs.frame_indirect_accumulate[bucket_id] + x + y * device.settings.width));
    }

    output = scale_color(output, bucket_norm);

    if (device.camera.indirect_only == false) {
      const RGBF direct = load_RGBF(device.ptrs.frame_direct_accumulate + x + y * device.settings.width);

      output = add_color(output, direct);
    }

    store_RGBF(device.ptrs.frame_current_result, x + y * device.settings.width, output);
  }
}

LUMINARY_KERNEL void temporal_accumulation_aov() {
  HANDLE_DEVICE_ABORT();

  const uint32_t amount = device.settings.width * device.settings.height;

  const float2 jitter = quasirandom_sequence_2D_global(QUASI_RANDOM_TARGET_CAMERA_JITTER);

  const float prev_scale     = device.state.sample_id;
  const float curr_inv_scale = 1.0f / (device.state.sample_id + 1.0f);

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
    RGBF direct_output = load_RGBF(device.ptrs.frame_current_result + offset);

    direct_output = scale_color(direct_output, prev_scale);
    direct_output = add_color(direct_output, direct_buffer);
    direct_output = scale_color(direct_output, curr_inv_scale);

    store_RGBF(device.ptrs.frame_current_result, offset, direct_output);
  }
}

#endif /* CU_TEMPORAL_H */

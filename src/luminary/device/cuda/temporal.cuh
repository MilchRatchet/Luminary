#ifndef CU_TEMPORAL_H
#define CU_TEMPORAL_H

#include "math.cuh"
#include "utils.cuh"

// TODO: Refactor this into a separate file, the idea of "temporal" is long gone now that reprojection is gone. While we are at it, get rid
// of the legacy naming of "temporal frames".

// Simple tent filter.
LUMINARY_FUNCTION float temporal_gather_pixel_weight(const float x, const float y) {
  return (1.0f - x) * (1.0f - y);
}

LUMINARY_FUNCTION RGBF temporal_gather_pixel_load(
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

LUMINARY_FUNCTION RGBF temporal_gather_pixel(
  const RGBF* image, const float pixel_x, const float pixel_y, const float base_x, const float base_y, const uint32_t width,
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

  const uint32_t width  = device.settings.window_width;
  const uint32_t height = device.settings.window_height;

  const uint32_t amount = width * height;

  const float2 jitter = camera_get_jitter();

  const uint32_t bucket_sample_count = device.state.sample_id / MAX_NUM_INDIRECT_BUCKETS;
  const uint32_t bucket_id           = device.state.sample_id - bucket_sample_count * MAX_NUM_INDIRECT_BUCKETS;

  float* indirect_bucket_ptr_red   = device.ptrs.frame_indirect_accumulate_red[bucket_id];
  float* indirect_bucket_ptr_green = device.ptrs.frame_indirect_accumulate_green[bucket_id];
  float* indirect_bucket_ptr_blue  = device.ptrs.frame_indirect_accumulate_blue[bucket_id];

  for (uint32_t offset = THREAD_ID; offset < amount; offset += blockDim.x * gridDim.x) {
    HANDLE_DEVICE_ABORT();

    const uint32_t y = device.settings.window_y + offset / width;
    const uint32_t x = device.settings.window_x + offset - (y - device.settings.window_y) * width;

    const uint32_t index = path_id_get_pixel_index(path_id_get(x, y, 0));

    const float pixel_x = x + 0.5f;
    const float pixel_y = y + 0.5f;

    const float base_x = floorf(x - (jitter.x - 0.5f)) + jitter.x;
    const float base_y = floorf(y - (jitter.y - 0.5f)) + jitter.y;

    // Direct Lighting
    RGBF direct_buffer = temporal_gather_pixel(
      device.ptrs.frame_direct_buffer, pixel_x, pixel_y, base_x, base_y, device.settings.width, device.settings.height);

    store_RGBF(device.ptrs.frame_direct_accumulate, index, direct_buffer);

    // Indirect Lighting
    RGBF indirect_buffer = temporal_gather_pixel(
      device.ptrs.frame_indirect_buffer, pixel_x, pixel_y, base_x, base_y, device.settings.width, device.settings.height);

    __stcs(indirect_bucket_ptr_red + index, indirect_buffer.r);
    __stcs(indirect_bucket_ptr_green + index, indirect_buffer.g);
    __stcs(indirect_bucket_ptr_blue + index, indirect_buffer.b);

    RGBF output = add_color(direct_buffer, indirect_buffer);

    store_RGBF(device.ptrs.frame_current_result, index, output);
  }
}

LUMINARY_KERNEL void temporal_accumulation_update() {
  HANDLE_DEVICE_ABORT();

  const uint32_t width  = device.settings.window_width;
  const uint32_t height = device.settings.window_height;

  const uint32_t amount = width * height;

  const float2 jitter = camera_get_jitter();

  const uint32_t bucket_sample_count = device.state.sample_id / MAX_NUM_INDIRECT_BUCKETS;
  const uint32_t bucket_id           = device.state.sample_id - bucket_sample_count * MAX_NUM_INDIRECT_BUCKETS;

  const bool load_indirect_bucket = device.state.aggregate_sample_count >= MAX_NUM_INDIRECT_BUCKETS;

  float* indirect_bucket_ptr_red   = device.ptrs.frame_indirect_accumulate_red[bucket_id];
  float* indirect_bucket_ptr_green = device.ptrs.frame_indirect_accumulate_green[bucket_id];
  float* indirect_bucket_ptr_blue  = device.ptrs.frame_indirect_accumulate_blue[bucket_id];

  for (uint32_t offset = THREAD_ID; offset < amount; offset += blockDim.x * gridDim.x) {
    HANDLE_DEVICE_ABORT();

    const uint32_t y = device.settings.window_y + offset / width;
    const uint32_t x = device.settings.window_x + offset - (y - device.settings.window_y) * width;

    const uint32_t index = path_id_get_pixel_index(path_id_get(x, y, 0));

    const float pixel_x = x + 0.5f;
    const float pixel_y = y + 0.5f;

    const float base_x = floorf(x - (jitter.x - 0.5f)) + jitter.x;
    const float base_y = floorf(y - (jitter.y - 0.5f)) + jitter.y;

    // Direct Lighting
    RGBF direct_buffer = temporal_gather_pixel(
      device.ptrs.frame_direct_buffer, pixel_x, pixel_y, base_x, base_y, device.settings.width, device.settings.height);
    RGBF direct_output = load_RGBF(device.ptrs.frame_direct_accumulate + index);

    direct_output = add_color(direct_output, direct_buffer);

    store_RGBF(device.ptrs.frame_direct_accumulate, index, direct_output);

    // Indirect Lighting
    RGBF indirect_buffer = temporal_gather_pixel(
      device.ptrs.frame_indirect_buffer, pixel_x, pixel_y, base_x, base_y, device.settings.width, device.settings.height);

    RGBF indirect_output = splat_color(0.0f);
    if (load_indirect_bucket) {
      indirect_output.r = __ldcs(indirect_bucket_ptr_red + index);
      indirect_output.g = __ldcs(indirect_bucket_ptr_green + index);
      indirect_output.b = __ldcs(indirect_bucket_ptr_blue + index);
    }

    indirect_output = add_color(indirect_output, indirect_buffer);

    __stcs(indirect_bucket_ptr_red + index, indirect_output.r);
    __stcs(indirect_bucket_ptr_green + index, indirect_output.g);
    __stcs(indirect_bucket_ptr_blue + index, indirect_output.b);
  }
}

LUMINARY_FUNCTION void temporal_load_buckets(
  const float* src, uint32_t x, uint32_t y, float values[9 * MAX_NUM_INDIRECT_BUCKETS], uint32_t& num_buckets) {
  if (y) {
    if (x) {
      values[num_buckets++] = fabsf(__ldcs(src + (x - 1) + (y - 1) * device.settings.width));
    }

    values[num_buckets++] = fabsf(__ldcs(src + x + (y - 1) * device.settings.width));

    if (x + 1 != device.settings.width) {
      values[num_buckets++] = fabsf(__ldcs(src + (x + 1) + (y - 1) * device.settings.width));
    }
  }

  if (x) {
    values[num_buckets++] = fabsf(__ldcs(src + (x - 1) + y * device.settings.width));
  }

  values[num_buckets++] = fabsf(__ldcs(src + x + y * device.settings.width));

  if (x + 1 != device.settings.width) {
    values[num_buckets++] = fabsf(__ldcs(src + (x + 1) + y * device.settings.width));
  }

  if (y + 1 != device.settings.height) {
    if (x) {
      values[num_buckets++] = fabsf(__ldcs(src + (x - 1) + (y + 1) * device.settings.width));
    }

    values[num_buckets++] = fabsf(__ldcs(src + x + (y + 1) * device.settings.width));

    if (x + 1 != device.settings.width) {
      values[num_buckets++] = fabsf(__ldcs(src + (x + 1) + (y + 1) * device.settings.width));
    }
  }
}

LUMINARY_FUNCTION float temporal_apply_median_of_means(float buckets[], const uint32_t num_buckets) {
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

  const uint32_t width  = device.settings.window_width;
  const uint32_t height = device.settings.window_height;

  const uint32_t amount = width * height;

  float values[9 * MAX_NUM_INDIRECT_BUCKETS];
  const uint32_t num_buckets_to_load = min(MAX_NUM_INDIRECT_BUCKETS, device.state.aggregate_sample_count + 1);

  const float normalization = 1.0f / (device.state.aggregate_sample_count + 1);

  for (uint32_t offset = THREAD_ID; offset < amount; offset += blockDim.x * gridDim.x) {
    HANDLE_DEVICE_ABORT();

    const uint32_t y = device.settings.window_y + offset / width;
    const uint32_t x = device.settings.window_x + offset - (y - device.settings.window_y) * width;

    RGBF output;
    uint32_t num_buckets;

    num_buckets = 0;
    for (uint32_t bucket_id = 0; bucket_id < num_buckets_to_load; bucket_id++) {
      temporal_load_buckets(device.ptrs.frame_indirect_accumulate_red[bucket_id], x, y, values, num_buckets);
    }
    output.r = temporal_apply_median_of_means(values, num_buckets);

    num_buckets = 0;
    for (uint32_t bucket_id = 0; bucket_id < num_buckets_to_load; bucket_id++) {
      temporal_load_buckets(device.ptrs.frame_indirect_accumulate_green[bucket_id], x, y, values, num_buckets);
    }
    output.g = temporal_apply_median_of_means(values, num_buckets);

    num_buckets = 0;
    for (uint32_t bucket_id = 0; bucket_id < num_buckets_to_load; bucket_id++) {
      temporal_load_buckets(device.ptrs.frame_indirect_accumulate_blue[bucket_id], x, y, values, num_buckets);
    }
    output.b = temporal_apply_median_of_means(values, num_buckets);

    // We have a mean estimate of the buckets, but the buckets are a partition of the samples so we need to scale the result.
    output = scale_color(output, MAX_NUM_INDIRECT_BUCKETS);

    if (device.camera.indirect_only == false) {
      const RGBF direct = load_RGBF(device.ptrs.frame_direct_accumulate + x + y * device.settings.width);

      output = add_color(output, direct);
    }

    output = scale_color(output, normalization);

    store_RGBF(device.ptrs.frame_current_result, x + y * device.settings.width, output);
  }
}

LUMINARY_KERNEL void temporal_accumulation_output_raw() {
  HANDLE_DEVICE_ABORT();

  const uint32_t width  = device.settings.window_width;
  const uint32_t height = device.settings.window_height;

  const uint32_t amount = width * height;

  const uint32_t num_buckets = min(MAX_NUM_INDIRECT_BUCKETS, device.state.aggregate_sample_count + 1);
  const float normalization  = 1.0f / (device.state.aggregate_sample_count + 1);

  for (uint32_t offset = THREAD_ID; offset < amount; offset += blockDim.x * gridDim.x) {
    HANDLE_DEVICE_ABORT();

    const uint32_t y = device.settings.window_y + offset / width;
    const uint32_t x = device.settings.window_x + offset - (y - device.settings.window_y) * width;

    const uint32_t index = path_id_get_pixel_index(path_id_get(x, y, 0));

    RGBF output = splat_color(0.0f);

    for (uint32_t bucket_id = 0; bucket_id < num_buckets; bucket_id++) {
      RGBF indirect_output;
      indirect_output.r = __ldcs(device.ptrs.frame_indirect_accumulate_red[bucket_id] + index);
      indirect_output.g = __ldcs(device.ptrs.frame_indirect_accumulate_green[bucket_id] + index);
      indirect_output.b = __ldcs(device.ptrs.frame_indirect_accumulate_blue[bucket_id] + index);

      output = add_color(output, indirect_output);
    }

    if (device.camera.indirect_only == false) {
      const RGBF direct = load_RGBF(device.ptrs.frame_direct_accumulate + index);

      output = add_color(output, direct);
    }

    output = scale_color(output, normalization);

    store_RGBF(device.ptrs.frame_current_result, index, output);
  }
}

LUMINARY_KERNEL void temporal_accumulation_aov() {
  HANDLE_DEVICE_ABORT();

  const uint32_t width  = device.settings.window_width;
  const uint32_t height = device.settings.window_height;

  const uint32_t amount = width * height;

  const float2 jitter = camera_get_jitter();

  const float prev_scale     = device.state.aggregate_sample_count;
  const float curr_inv_scale = 1.0f / (device.state.aggregate_sample_count + 1.0f);

  for (uint32_t offset = THREAD_ID; offset < amount; offset += blockDim.x * gridDim.x) {
    HANDLE_DEVICE_ABORT();

    const uint32_t y = device.settings.window_y + offset / width;
    const uint32_t x = device.settings.window_x + offset - (y - device.settings.window_y) * width;

    const uint32_t index = path_id_get_pixel_index(path_id_get(x, y, 0));

    const float pixel_x = x + 0.5f;
    const float pixel_y = y + 0.5f;

    const float base_x = floorf(x - (jitter.x - 0.5f)) + jitter.x;
    const float base_y = floorf(y - (jitter.y - 0.5f)) + jitter.y;

    // Direct Lighting
    RGBF direct_buffer = temporal_gather_pixel(
      device.ptrs.frame_direct_buffer, pixel_x, pixel_y, base_x, base_y, device.settings.width, device.settings.height);
    RGBF direct_output = load_RGBF(device.ptrs.frame_current_result + index);

    direct_output = scale_color(direct_output, prev_scale);
    direct_output = add_color(direct_output, direct_buffer);
    direct_output = scale_color(direct_output, curr_inv_scale);

    store_RGBF(device.ptrs.frame_current_result, index, direct_output);
  }
}

#endif /* CU_TEMPORAL_H */

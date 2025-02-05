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

  const uint32_t bucket_id       = device.state.sample_id % device.settings.num_indirect_buckets;
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
  const uint32_t overall_sample_count           = device.state.sample_id / device.settings.num_indirect_buckets;
  const uint32_t buckets_with_additional_sample = device.state.sample_id - overall_sample_count * device.settings.num_indirect_buckets;

  return overall_sample_count + ((bucket_id < buckets_with_additional_sample) ? 1 : 0);
}

LUMINARY_KERNEL void temporal_accumulation_update() {
  HANDLE_DEVICE_ABORT();

  const uint32_t amount = device.settings.width * device.settings.height;

  const float2 jitter = quasirandom_sequence_2D_global(QUASI_RANDOM_TARGET_CAMERA_JITTER);

  const float prev_scale     = device.state.sample_id;
  const float curr_inv_scale = 1.0f / (device.state.sample_id + 1.0f);

  const uint32_t bucket_id       = device.state.sample_id % device.settings.num_indirect_buckets;
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

__device__ void temporal_load_buckets_0(uint32_t bucket_id, uint32_t x, uint32_t y, RGBF buckets[], uint32_t& num_buckets) {
  if (device.settings.num_indirect_buckets > bucket_id && device.state.sample_id > bucket_id) {
    const RGB_E6M20* src   = device.ptrs.frame_indirect_accumulate[bucket_id];
    buckets[num_buckets++] = load_RGBF(src + x + y * device.settings.width);
  }
}

__device__ void temporal_load_buckets_1(uint32_t bucket_id, uint32_t x, uint32_t y, RGBF buckets[], uint32_t& num_buckets) {
  if (device.settings.num_indirect_buckets > bucket_id && device.state.sample_id > bucket_id) {
    const RGB_E6M20* src = device.ptrs.frame_indirect_accumulate[bucket_id];
    RGBF_load_pair(src, x, y, device.settings.width, buckets[num_buckets + 0], buckets[num_buckets + 1]);
    RGBF_load_pair(src, x, y + 1, device.settings.width, buckets[num_buckets + 2], buckets[num_buckets + 3]);
    num_buckets += 4;
  }
}

__device__ RGBF temporal_apply_median_of_means(RGBF buckets[], const uint32_t num_buckets) {
  // Sort red component
  {
    uint32_t i = 1;

    while (i < num_buckets) {
      const float x = buckets[i].r;
      uint32_t j    = i;
      while (j > 0 && buckets[j - 1].r > x) {
        buckets[j].r = buckets[j - 1].r;
        j--;
      }

      buckets[j].r = x;
      i++;
    }
  }

  // Sort green component
  {
    uint32_t i = 1;

    while (i < num_buckets) {
      const float x = buckets[i].g;
      uint32_t j    = i;
      while (j > 0 && buckets[j - 1].g > x) {
        buckets[j].g = buckets[j - 1].g;
        j--;
      }

      buckets[j].g = x;
      i++;
    }
  }

  // Sort blue component
  {
    uint32_t i = 1;

    while (i < num_buckets) {
      const float x = buckets[i].b;
      uint32_t j    = i;
      while (j > 0 && buckets[j - 1].b > x) {
        buckets[j].b = buckets[j - 1].b;
        j--;
      }

      buckets[j].b = x;
      i++;
    }
  }

  RGBF output;

#if 1

  // Gini - Median of Means

  // MoN Red component
  {
    float num   = 0.0f;
    float denom = 0.0f;

    for (uint32_t bucket_id = 0; bucket_id < num_buckets; bucket_id++) {
      const float value = buckets[bucket_id].r;
      num += bucket_id * value;
      denom += value;
    }

    num *= 2.0f;
    denom *= num_buckets;

    const float G = (num / denom) - (num_buckets + 1.0f) / num_buckets;

    const uint32_t k = num_buckets >> 1;
    const uint32_t c = k - (1.0f - G) * k;

    float result = 0.0f;

    for (uint32_t bucket_id = c; bucket_id < num_buckets - c; bucket_id++) {
      result += buckets[bucket_id].r;
    }

    result /= num_buckets - 2 * c;

    output.r = result;
  }

  // MoN Green component
  {
    float num   = 0.0f;
    float denom = 0.0f;

    for (uint32_t bucket_id = 0; bucket_id < num_buckets; bucket_id++) {
      const float value = buckets[bucket_id].g;
      num += bucket_id * value;
      denom += value;
    }

    num *= 2.0f;
    denom *= num_buckets;

    const float G = (num / denom) - (num_buckets + 1.0f) / num_buckets;

    const uint32_t k = num_buckets >> 1;
    const uint32_t c = k - (1.0f - G) * k;

    float result = 0.0f;

    for (uint32_t bucket_id = c; bucket_id < num_buckets - c; bucket_id++) {
      result += buckets[bucket_id].g;
    }

    result /= num_buckets - 2 * c;

    output.g = result;
  }

  // MoN Blue component
  {
    float num   = 0.0f;
    float denom = 0.0f;

    for (uint32_t bucket_id = 0; bucket_id < num_buckets; bucket_id++) {
      const float value = buckets[bucket_id].b;
      num += bucket_id * value;
      denom += value;
    }

    num *= 2.0f;
    denom *= num_buckets;

    const float G = (num / denom) - (num_buckets + 1.0f) / num_buckets;

    const uint32_t k = num_buckets >> 1;
    const uint32_t c = k - (1.0f - G) * k;

    float result = 0.0f;

    for (uint32_t bucket_id = c; bucket_id < num_buckets - c; bucket_id++) {
      result += buckets[bucket_id].b;
    }

    result /= num_buckets - 2 * c;

    output.b = result;
  }

#else

  // Median of Means
  output = get_color(buckets[num_buckets >> 1].r, buckets[num_buckets >> 1].g, buckets[num_buckets >> 1].b);

#endif

  return output;
}

LUMINARY_KERNEL void temporal_accumulation_output_0() {
  HANDLE_DEVICE_ABORT();

  const uint32_t width  = device.settings.width;
  const uint32_t height = device.settings.height;

  const uint32_t amount = width * height;

  for (uint32_t offset = THREAD_ID; offset < amount; offset += blockDim.x * gridDim.x) {
    const uint32_t y = offset / width;
    const uint32_t x = offset - y * width;

    uint32_t num_buckets = 0;
    RGBF buckets[4 * 8];

    temporal_load_buckets_0(0, x, y, buckets, num_buckets);
    temporal_load_buckets_0(1, x, y, buckets, num_buckets);
    temporal_load_buckets_0(2, x, y, buckets, num_buckets);
    temporal_load_buckets_0(3, x, y, buckets, num_buckets);
    temporal_load_buckets_0(4, x, y, buckets, num_buckets);
    temporal_load_buckets_0(5, x, y, buckets, num_buckets);
    temporal_load_buckets_0(6, x, y, buckets, num_buckets);
    temporal_load_buckets_0(7, x, y, buckets, num_buckets);

    RGBF output;

    if (device.camera.do_firefly_rejection) {
      output = temporal_apply_median_of_means(buckets, num_buckets);
    }
    else {
      output = get_color(0.0f, 0.0f, 0.0f);

      for (uint32_t bucket_id = 0; bucket_id < num_buckets; bucket_id++) {
        output = add_color(output, buckets[bucket_id]);
      }

      output = scale_color(output, 1.0f / num_buckets);
    }

    if (device.camera.indirect_only == false) {
      const RGBF direct = load_RGBF(device.ptrs.frame_direct_accumulate + x + y * device.settings.width);

      output = add_color(output, direct);
    }

    store_RGBF(device.ptrs.frame_current_result, x + y * device.settings.width, output);
  }
}

LUMINARY_KERNEL void temporal_accumulation_output_1() {
  HANDLE_DEVICE_ABORT();

  const uint32_t width  = device.settings.width >> device.settings.supersampling;
  const uint32_t height = device.settings.height >> device.settings.supersampling;

  const uint32_t amount = width * height;

  for (uint32_t offset = THREAD_ID; offset < amount; offset += blockDim.x * gridDim.x) {
    const uint32_t y = offset / width;
    const uint32_t x = offset - y * width;

    const uint32_t base_x = x << device.settings.supersampling;
    const uint32_t base_y = y << device.settings.supersampling;

    uint32_t num_buckets = 0;
    RGBF buckets[4 * 8];

    temporal_load_buckets_1(0, base_x, base_y, buckets, num_buckets);
    temporal_load_buckets_1(1, base_x, base_y, buckets, num_buckets);
    temporal_load_buckets_1(2, base_x, base_y, buckets, num_buckets);
    temporal_load_buckets_1(3, base_x, base_y, buckets, num_buckets);
    temporal_load_buckets_1(4, base_x, base_y, buckets, num_buckets);
    temporal_load_buckets_1(5, base_x, base_y, buckets, num_buckets);
    temporal_load_buckets_1(6, base_x, base_y, buckets, num_buckets);
    temporal_load_buckets_1(7, base_x, base_y, buckets, num_buckets);

    RGBF output00;
    RGBF output01;
    RGBF output10;
    RGBF output11;

    if (device.camera.do_firefly_rejection) {
      const RGBF indirect_output = temporal_apply_median_of_means(buckets, num_buckets);

      output00 = indirect_output;
      output01 = indirect_output;
      output10 = indirect_output;
      output11 = indirect_output;
    }
    else {
      output00 = get_color(0.0f, 0.0f, 0.0f);
      output01 = get_color(0.0f, 0.0f, 0.0f);
      output10 = get_color(0.0f, 0.0f, 0.0f);
      output11 = get_color(0.0f, 0.0f, 0.0f);

      for (uint32_t bucket_id = 0; bucket_id < num_buckets; bucket_id += 4) {
        output00 = add_color(output00, buckets[bucket_id + 0]);
        output01 = add_color(output01, buckets[bucket_id + 1]);
        output10 = add_color(output10, buckets[bucket_id + 2]);
        output11 = add_color(output11, buckets[bucket_id + 3]);
      }

      output00 = scale_color(output00, 4.0f / num_buckets);
      output01 = scale_color(output01, 4.0f / num_buckets);
      output10 = scale_color(output10, 4.0f / num_buckets);
      output11 = scale_color(output11, 4.0f / num_buckets);
    }

    if (device.camera.indirect_only == false) {
      RGBF direct00;
      RGBF direct01;
      RGBF_load_pair(device.ptrs.frame_direct_accumulate, base_x, base_y, device.settings.width, direct00, direct01);

      output00 = add_color(output00, direct00);
      output01 = add_color(output01, direct01);

      RGBF direct10;
      RGBF direct11;
      RGBF_load_pair(device.ptrs.frame_direct_accumulate, base_x, base_y + 1, device.settings.width, direct10, direct11);

      output10 = add_color(output10, direct10);
      output11 = add_color(output11, direct11);
    }

    RGBF_store_pair(device.ptrs.frame_current_result, base_x, base_y, device.settings.width, output00, output01);
    RGBF_store_pair(device.ptrs.frame_current_result, base_x, base_y + 1, device.settings.width, output10, output11);
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

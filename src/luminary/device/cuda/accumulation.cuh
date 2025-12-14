#ifndef CU_LUMINARY_ACCUMULATION_H
#define CU_LUMINARY_ACCUMULATION_H

#include "adaptive_sampling.cuh"
#include "math.cuh"
#include "memory.cuh"
#include "utils.cuh"

LUMINARY_FUNCTION bool accumulation_reduce_sample(const RGBF value, const uint32_t index, RGBF& first_moment, RGBF& second_moment) {
  const uint32_t index_mask = __match_any_sync(__activemask(), index);

  const bool is_first_thread = (__ffs(index_mask) - 1) == THREAD_ID_IN_WARP;

  // TODO: Implement fast path for when index_mask == 0xFFFFFFFF

  first_moment  = value;
  second_moment = mul_color(value, value);

  for (uint32_t mask = index_mask & ~(1u << THREAD_ID_IN_WARP); mask != 0; mask &= mask - 1) {
    uint32_t other_thread_id = __ffs(mask) - 1;

    const float red   = __shfl_sync(index_mask, value.r, other_thread_id);
    const float green = __shfl_sync(index_mask, value.g, other_thread_id);
    const float blue  = __shfl_sync(index_mask, value.b, other_thread_id);

    const RGBF color = get_color(red, green, blue);

    first_moment  = add_color(first_moment, color);
    second_moment = add_color(second_moment, mul_color(color, color));
  }

  return is_first_thread;
}

LUMINARY_KERNEL void accumulation_collect_results() {
  HANDLE_DEVICE_ABORT();

  const int results_count = device.ptrs.results_counts[THREAD_ID];

  LUMINARY_ASSUME(results_count <= MAXIMUM_TASKS_PER_THREAD);

  for (int result_id = 0; result_id < results_count; result_id++) {
    const uint32_t task_result_base_address = task_get_base_address<DeviceTaskResult>(result_id, TASK_STATE_BUFFER_INDEX_RESULT);

    const DeviceTaskResult result = task_result_load(task_result_base_address);

    RGBF first_moment, second_moment;
    const bool is_first_thread = accumulation_reduce_sample(result.color, result.index, first_moment, second_moment);

    if (is_first_thread) {
      // TODO: Ensure this compiles to "red" instructions
      atomicAdd(device.ptrs.frame_first_moment[FRAME_CHANNEL_RED] + result.index, first_moment.r);
      atomicAdd(device.ptrs.frame_first_moment[FRAME_CHANNEL_GREEN] + result.index, first_moment.g);
      atomicAdd(device.ptrs.frame_first_moment[FRAME_CHANNEL_BLUE] + result.index, first_moment.b);

      atomicAdd(device.ptrs.frame_second_moment[FRAME_CHANNEL_RED] + result.index, second_moment.r);
      atomicAdd(device.ptrs.frame_second_moment[FRAME_CHANNEL_GREEN] + result.index, second_moment.g);
      atomicAdd(device.ptrs.frame_second_moment[FRAME_CHANNEL_BLUE] + result.index, second_moment.b);
    }
  }
}

LUMINARY_KERNEL void accumulation_collect_results_first_sample() {
  HANDLE_DEVICE_ABORT();

  const int results_count = device.ptrs.results_counts[THREAD_ID];

  LUMINARY_ASSUME(results_count <= MAXIMUM_TASKS_PER_THREAD);

  for (int result_id = 0; result_id < results_count; result_id++) {
    const uint32_t task_result_base_address = task_get_base_address<DeviceTaskResult>(result_id, TASK_STATE_BUFFER_INDEX_RESULT);

    const DeviceTaskResult result = task_result_load(task_result_base_address);

    const RGBF first_moment  = result.color;
    const RGBF second_moment = mul_color(result.color, result.color);

    __stcs(device.ptrs.frame_first_moment[FRAME_CHANNEL_RED] + result.index, first_moment.r);
    __stcs(device.ptrs.frame_first_moment[FRAME_CHANNEL_GREEN] + result.index, first_moment.g);
    __stcs(device.ptrs.frame_first_moment[FRAME_CHANNEL_BLUE] + result.index, first_moment.b);

    __stcs(device.ptrs.frame_second_moment[FRAME_CHANNEL_RED] + result.index, second_moment.r);
    __stcs(device.ptrs.frame_second_moment[FRAME_CHANNEL_GREEN] + result.index, second_moment.g);
    __stcs(device.ptrs.frame_second_moment[FRAME_CHANNEL_BLUE] + result.index, second_moment.b);
  }
}

LUMINARY_KERNEL void accumulation_generate_result() {
  HANDLE_DEVICE_ABORT();

  const uint32_t width  = device.settings.window_width;
  const uint32_t height = device.settings.window_height;

  const uint32_t offset_x = device.settings.window_x;
  const uint32_t offset_y = device.settings.window_y;

  const uint32_t amount = width * height;

  for (uint32_t window_index = THREAD_ID; window_index < amount; window_index += NUM_THREADS) {
    HANDLE_DEVICE_ABORT();

    const uint32_t y = offset_y + window_index / width;
    const uint32_t x = offset_x + window_index - (y - offset_y) * width;

    const uint32_t index = x + y * device.settings.width;

    const uint32_t sample_count = adaptive_sampling_get_sample_count(x, y);

    const float normalization = 1.0f / sample_count;

    RGBF result;
    switch (device.settings.adaptive_sampling_output_mode) {
      case LUMINARY_ADAPTIVE_SAMPLING_OUTPUT_MODE_BEAUTY: {
        result.r = __ldcs(device.ptrs.frame_first_moment[FRAME_CHANNEL_RED] + index) * normalization;
        result.g = __ldcs(device.ptrs.frame_first_moment[FRAME_CHANNEL_GREEN] + index) * normalization;
        result.b = __ldcs(device.ptrs.frame_first_moment[FRAME_CHANNEL_BLUE] + index) * normalization;
      } break;
      case LUMINARY_ADAPTIVE_SAMPLING_OUTPUT_MODE_VARIANCE: {
        float luminance;
        const float variance = adaptive_sampling_get_pixel_max_variance(x, y, normalization, luminance);

        const float rel_variance = (luminance > 0.0f) ? variance / luminance : 0.0f;

        result = splat_color(rel_variance);
      } break;
      case LUMINARY_ADAPTIVE_SAMPLING_OUTPUT_MODE_ERROR: {
        float luminance;
        const float variance = adaptive_sampling_get_pixel_max_variance(x, y, normalization, luminance);

        const float tonemap_compression = adaptive_sampling_compute_tonemap_compression_factor(luminance, device.camera.exposure);

        const float rel_variance = (luminance > 0.0f) ? variance / luminance : 0.0f;
        const float rel_mse      = rel_variance * normalization * tonemap_compression * tonemap_compression;

        const float value = 1024.0f * rel_mse;
        const float red   = __saturatef(2.0f * value);
        const float green = __saturatef(2.0f * (value - 0.5f));
        const float blue  = __saturatef((value > 0.5f) ? 4.0f * (0.25f - fabsf(value - 1.0f)) : 4.0f * (0.25f - fabsf(value - 0.25f)));

        result = get_color(red, green, blue);

      } break;
      case LUMINARY_ADAPTIVE_SAMPLING_OUTPUT_MODE_SAMPLE_DISTRIBUTION: {
        const uint32_t tasks_per_pixel = adaptive_sampling_get_current_tasks_per_pixel(x, y);

        const float rel_value = ((float) tasks_per_pixel) / ADAPTIVE_SAMPLING_MAX_SAMPLING_RATE;

        result = splat_color(rel_value);
      } break;
    }

    __stcs(device.ptrs.frame_result[FRAME_CHANNEL_RED] + index, result.r);
    __stcs(device.ptrs.frame_result[FRAME_CHANNEL_GREEN] + index, result.g);
    __stcs(device.ptrs.frame_result[FRAME_CHANNEL_BLUE] + index, result.b);
  }
}

LUMINARY_FUNCTION RGBF
  accumulation_load_undersampling_pixel(const uint16_t base_x, const uint16_t base_y, const uint32_t scale, const uint32_t sample_id) {
  const uint32_t offset_x = (sample_id & 0b01) ? 0 : scale >> 1;
  const uint32_t offset_y = (sample_id & 0b10) ? 0 : scale >> 1;

  const uint32_t pixel_x = min(base_x + offset_x, device.settings.width - 1);
  const uint32_t pixel_y = min(base_y + offset_y, device.settings.height - 1);

  const uint32_t index = pixel_x + pixel_y * device.settings.width;

  const float red   = __ldg(device.ptrs.frame_first_moment[FRAME_CHANNEL_RED] + index);
  const float green = __ldg(device.ptrs.frame_first_moment[FRAME_CHANNEL_GREEN] + index);
  const float blue  = __ldg(device.ptrs.frame_first_moment[FRAME_CHANNEL_BLUE] + index);

  return get_color(red, green, blue);
}

LUMINARY_KERNEL void accumulation_generate_result_undersampling() {
  HANDLE_DEVICE_ABORT();

  const uint32_t undersampling_stage     = (device.state.undersampling & UNDERSAMPLING_STAGE_MASK) >> UNDERSAMPLING_STAGE_SHIFT;
  const uint32_t undersampling_iteration = device.state.undersampling & UNDERSAMPLING_ITERATION_MASK;

  LUMINARY_ASSUME(undersampling_stage > 0);

  const uint32_t scale = 1u << undersampling_stage;

  // During undersampling we always copy the whole screen because
  // we dynamically change the memory layout based on undersampling stage.
  // We ignore render regions because they are not of interest here.
  const uint32_t width  = device.settings.width >> undersampling_stage;
  const uint32_t height = device.settings.height >> undersampling_stage;

  const uint32_t amount = width * height;

  const float color_scale = 1.0f / (4 - undersampling_iteration);

  for (uint32_t index = THREAD_ID; index < amount; index += NUM_THREADS) {
    HANDLE_DEVICE_ABORT();

    const uint32_t dst_y = index / width;
    const uint32_t dst_x = index - dst_y * width;

    const uint32_t base_x = dst_x << undersampling_stage;
    const uint32_t base_y = dst_y << undersampling_stage;

    RGBF result = splat_color(0.0f);
    for (uint32_t sample_id = undersampling_iteration; sample_id < 4; sample_id++) {
      const RGBF pixel = accumulation_load_undersampling_pixel(base_x, base_y, scale, sample_id);
      result           = add_color(result, pixel);
    }

    result = scale_color(result, color_scale);

    __stcs(device.ptrs.frame_result[FRAME_CHANNEL_RED] + index, result.r);
    __stcs(device.ptrs.frame_result[FRAME_CHANNEL_GREEN] + index, result.g);
    __stcs(device.ptrs.frame_result[FRAME_CHANNEL_BLUE] + index, result.b);
  }
}

#endif /* CU_LUMINARY_ACCUMULATION_H */

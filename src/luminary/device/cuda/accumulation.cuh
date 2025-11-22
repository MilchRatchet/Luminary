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

    DeviceTaskResult result = task_result_load(task_result_base_address);

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

    DeviceTaskResult result = task_result_load(task_result_base_address);

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

  const uint32_t amount = width * height;

  for (uint32_t index = THREAD_ID; index < amount; index += NUM_THREADS) {
    HANDLE_DEVICE_ABORT();

    const uint32_t y = device.settings.window_y + index / width;
    const uint32_t x = device.settings.window_x + index - (y - device.settings.window_y) * width;

    const uint32_t sample_count = adaptive_sampling_get_sample_count(x, y);

    const float normalization = 1.0f / sample_count;

    const float red   = __ldcs(device.ptrs.frame_first_moment[FRAME_CHANNEL_RED] + index) * normalization;
    const float green = __ldcs(device.ptrs.frame_first_moment[FRAME_CHANNEL_GREEN] + index) * normalization;
    const float blue  = __ldcs(device.ptrs.frame_first_moment[FRAME_CHANNEL_BLUE] + index) * normalization;

    __stcs(device.ptrs.frame_result[FRAME_CHANNEL_RED] + index, red);
    __stcs(device.ptrs.frame_result[FRAME_CHANNEL_GREEN] + index, green);
    __stcs(device.ptrs.frame_result[FRAME_CHANNEL_BLUE] + index, blue);
  }
}

#endif /* CU_LUMINARY_ACCUMULATION_H */

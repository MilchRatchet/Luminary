#ifndef CU_LUMINARY_ADAPTIVE_SAMPLING_H
#define CU_LUMINARY_ADAPTIVE_SAMPLING_H

#include "math.cuh"
#include "memory.cuh"
#include "utils.cuh"

LUMINARY_FUNCTION uint32_t adapative_sampling_get_sample_offset(const uint32_t x, const uint32_t y) {
  const uint32_t adaptive_sampling_width = device.settings.width >> ADAPTIVE_SAMPLING_BLOCK_SIZE_LOG;
  const uint32_t adaptive_sampling_x     = x >> ADAPTIVE_SAMPLING_BLOCK_SIZE_LOG;
  const uint32_t adaptive_sampling_y     = y >> ADAPTIVE_SAMPLING_BLOCK_SIZE_LOG;

  const uint32_t adaptive_sampling_block = adaptive_sampling_x + adaptive_sampling_y * adaptive_sampling_width;

  const uint32_t adaptive_sampling_counts = device.ptrs.stage_sample_counts[adaptive_sampling_block];

  uint32_t sample_id = device.state.sample_allocation.stage_sample_offsets[0];

  for (uint32_t stage_id = 0; stage_id < ADAPTIVE_SAMPLER_NUM_STAGES; stage_id++) {
    const uint32_t stage_sample_offset = device.state.sample_allocation.stage_sample_offsets[stage_id + 1];
    const uint32_t stage_sample_count  = ((adaptive_sampling_counts >> (stage_id * 8)) & 0xFF) + 1;

    sample_id += stage_sample_offset * stage_sample_count;
  }

  return sample_id;
}

LUMINARY_FUNCTION uint32_t adaptive_sampling_get_sample_count_from_block_index(const uint32_t adaptive_sampling_block) {
  // TODO: This has to be different because for multi-device the sample count is not computed from the sample allocation.

  const uint32_t adaptive_sampling_counts = device.ptrs.stage_sample_counts[adaptive_sampling_block];

  uint32_t count = device.state.sample_allocation.stage_sample_offsets[0];

  for (uint32_t stage_id = 0; stage_id < ADAPTIVE_SAMPLER_NUM_STAGES; stage_id++) {
    const uint32_t stage_sample_offset = device.state.sample_allocation.stage_sample_offsets[stage_id + 1];
    const uint32_t stage_sample_count  = ((adaptive_sampling_counts >> (stage_id * 8)) & 0xFF) + 1;

    count += stage_sample_offset * stage_sample_count;
  }

  const uint32_t this_stage_id = device.state.sample_allocation.stage_id;

  if (this_stage_id != 0) {
    count += ((adaptive_sampling_counts >> (this_stage_id * 8)) & 0xFF) + 1;
  }
  else {
    count++;
  }

  return count;
}

LUMINARY_FUNCTION uint32_t adaptive_sampling_get_sample_count(const uint32_t x, const uint32_t y) {
  const uint32_t adaptive_sampling_width = device.settings.width >> ADAPTIVE_SAMPLING_BLOCK_SIZE_LOG;
  const uint32_t adaptive_sampling_x     = x >> ADAPTIVE_SAMPLING_BLOCK_SIZE_LOG;
  const uint32_t adaptive_sampling_y     = y >> ADAPTIVE_SAMPLING_BLOCK_SIZE_LOG;

  const uint32_t adaptive_sampling_block = adaptive_sampling_x + adaptive_sampling_y * adaptive_sampling_width;

  return adaptive_sampling_get_sample_count_from_block_index(adaptive_sampling_block);
}

LUMINARY_FUNCTION float adaptive_sampling_get_pixel_max_variance(const uint32_t x, const uint32_t y, const float inv_n, float& max_value) {
  const bool pixel_in_frame = (x < device.settings.width) && (y < device.settings.height);
  if (pixel_in_frame == false) {
    max_value = 0.0f;
    return 0.0f;
  }

  const uint32_t index = x + y * device.settings.width;

  const float red1   = __ldg(device.ptrs.frame_first_moment[FRAME_CHANNEL_RED] + index) * inv_n;
  const float green1 = __ldg(device.ptrs.frame_first_moment[FRAME_CHANNEL_GREEN] + index) * inv_n;
  const float blue1  = __ldg(device.ptrs.frame_first_moment[FRAME_CHANNEL_BLUE] + index) * inv_n;

  max_value = fmaxf(red1, fmaxf(green1, blue1));

  const float red2   = __ldg(device.ptrs.frame_second_moment[FRAME_CHANNEL_RED] + index) * inv_n;
  const float green2 = __ldg(device.ptrs.frame_second_moment[FRAME_CHANNEL_GREEN] + index) * inv_n;
  const float blue2  = __ldg(device.ptrs.frame_second_moment[FRAME_CHANNEL_BLUE] + index) * inv_n;

  const float variance_red   = red2 - red1 * red1;
  const float variance_green = green2 - green1 * green1;
  const float variance_blue  = blue2 - blue1 * blue1;

  return fmaxf(variance_red, fmaxf(variance_green, variance_blue));
}

LUMINARY_KERNEL void adaptive_sampling_compute_stage_sample_counts(const KernelArgsAdaptiveSamplingComputeStageSampleCounts args) {
  const uint32_t adaptive_sampling_width =
    (device.settings.width + (1u << ADAPTIVE_SAMPLING_BLOCK_SIZE_LOG) - 1) >> ADAPTIVE_SAMPLING_BLOCK_SIZE_LOG;

  const uint32_t adaptive_sampling_block = WARP_ID;

  const uint32_t sample_count = adaptive_sampling_get_sample_count_from_block_index(adaptive_sampling_block);
  const float denominator     = 1.0f / sample_count;

  const uint32_t adaptive_sampling_y = adaptive_sampling_block / adaptive_sampling_width;
  const uint32_t adaptive_sampling_x = adaptive_sampling_block - adaptive_sampling_y * adaptive_sampling_width;

  const uint32_t x0 = (adaptive_sampling_x << ADAPTIVE_SAMPLING_BLOCK_SIZE_LOG) + (THREAD_ID_IN_WARP & ADAPTIVE_SAMPLING_BLOCK_SIZE_MASK);
  const uint32_t y0 = (adaptive_sampling_y << ADAPTIVE_SAMPLING_BLOCK_SIZE_LOG) + (THREAD_ID_IN_WARP >> ADAPTIVE_SAMPLING_BLOCK_SIZE_LOG);

  float max_value0;
  float max_variance = adaptive_sampling_get_pixel_max_variance(x0, y0, denominator, max_value0);

  const uint32_t x1 = x0;
  const uint32_t y1 = y0 + (1u << (ADAPTIVE_SAMPLING_BLOCK_SIZE_LOG - 1));

  float max_value1;
  max_variance = fmaxf(max_variance, adaptive_sampling_get_pixel_max_variance(x1, y1, denominator, max_value1));

  const float max_value = fmaxf(max_value0, max_value1);

  const float max_variance_block = warp_reduce_max(max_variance);
  const float max_value_block    = warp_reduce_max(max_value);

  if (THREAD_ID_IN_WARP == 0) {
    const float rel_variance = (max_value_block > 0.0f) ? max_variance_block / max_value_block : 0.0f;

    uint32_t adaptive_sampling_counts = device.ptrs.stage_sample_counts[adaptive_sampling_block];

    // Zero out all the bits not currently occupied by valid data.
    adaptive_sampling_counts &= (1u << (args.current_stage_id * 8)) - 1;

    uint32_t new_sample_count = floorf(rel_variance * 16.0f) * (1.0f / 32.0f);

    new_sample_count = max(new_sample_count, 1);
    new_sample_count = min(new_sample_count, 128);

    adaptive_sampling_counts |= new_sample_count << (args.current_stage_id * 8);

    device.ptrs.stage_sample_counts[adaptive_sampling_block] = adaptive_sampling_counts;
  }
}

LUMINARY_KERNEL void adaptive_sampling_compute_stage_total_task_counts(const KernelArgsAdaptiveSamplingComputeStageTotalTaskCounts args) {
  const uint32_t adaptive_sampling_width =
    (device.settings.width + (1u << ADAPTIVE_SAMPLING_BLOCK_SIZE_LOG) - 1) >> ADAPTIVE_SAMPLING_BLOCK_SIZE_LOG;
  const uint32_t adaptive_sampling_height =
    (device.settings.height + (1u << ADAPTIVE_SAMPLING_BLOCK_SIZE_LOG) - 1) >> ADAPTIVE_SAMPLING_BLOCK_SIZE_LOG;

  const uint32_t adaptive_sampling_amount = adaptive_sampling_width * adaptive_sampling_height;

  const uint32_t adaptive_sampling_block = THREAD_ID;

  uint32_t tasks_per_pixel = 0;
  if (adaptive_sampling_block < adaptive_sampling_amount) {
    tasks_per_pixel = device.ptrs.stage_sample_counts[adaptive_sampling_block];
  }

  tasks_per_pixel *= (1u << ADAPTIVE_SAMPLING_BLOCK_SIZE_LOG) * (1u << ADAPTIVE_SAMPLING_BLOCK_SIZE_LOG);

  const uint32_t total_tasks_warp = warp_reduce_sum(tasks_per_pixel);

  atomicAdd(&device.ptrs.stage_total_task_counts[args.stage_id], total_tasks_warp);
}

#endif /* CU_LUMINARY_ADAPTIVE_SAMPLING_H */

#ifndef CU_LUMINARY_ADAPTIVE_SAMPLING_H
#define CU_LUMINARY_ADAPTIVE_SAMPLING_H

#include "math.cuh"
#include "memory.cuh"
#include "tonemap.cuh"
#include "utils.cuh"

LUMINARY_FUNCTION float adaptive_sampling_compute_tonemap_compression_factor(const RGBF color, const float exposure) {
  const RGBF exposed_color    = scale_color(color, exposure);
  const RGBF tonemapped_color = tonemap_apply_transform(exposed_color);

  const float exposed_value    = color_luminance(exposed_color);
  const float tonemapped_value = color_luminance(tonemapped_color);

  return (exposed_value > 0.0f) ? tonemapped_value / exposed_value : 1.0f;
}

LUMINARY_FUNCTION uint32_t adaptive_sampling_get_stage_sample_count(const uint32_t stage_sample_counts, const uint32_t stage_id) {
  return ((stage_sample_counts >> (stage_id * 8)) & 0xFF) + 1;
}

LUMINARY_FUNCTION uint32_t
  adaptive_sampling_find_block(const uint32_t task_id, const uint32_t block_start, const uint32_t block_end, uint32_t& offset) {
  uint32_t left  = block_start;
  uint32_t right = block_end;

  while (left < right) {
    const uint32_t mid = (left + right) >> 1;

    const uint32_t block_offset = device.ptrs.adaptive_sampling_block_task_offsets[mid];

    if (task_id < block_offset) {
      right = mid;
    }
    else {
      left = mid + 1;
    }
  }

  const uint32_t block_id = left;

  offset = (block_id > 0) ? device.ptrs.adaptive_sampling_block_task_offsets[block_id - 1] : 0;

  return block_id;
}

LUMINARY_FUNCTION uint32_t adaptive_sampling_get_block_index(const uint32_t x, const uint32_t y) {
  const uint32_t adaptive_sampling_width =
    (device.settings.window_width + (1u << ADAPTIVE_SAMPLING_BLOCK_SIZE_LOG) - 1) >> ADAPTIVE_SAMPLING_BLOCK_SIZE_LOG;
  const uint32_t adaptive_sampling_x = (x - device.settings.window_x) >> ADAPTIVE_SAMPLING_BLOCK_SIZE_LOG;
  const uint32_t adaptive_sampling_y = (y - device.settings.window_y) >> ADAPTIVE_SAMPLING_BLOCK_SIZE_LOG;

  return adaptive_sampling_x + adaptive_sampling_y * adaptive_sampling_width;
}

template <bool FIRST_STAGE_ONLY>
LUMINARY_FUNCTION uint32_t adapative_sampling_get_sample_offset(const uint32_t x, const uint32_t y) {
  uint32_t sample_id = device.state.sample_allocation.stage_sample_offsets[0];

  if constexpr (FIRST_STAGE_ONLY == false) {
    const uint32_t adaptive_sampling_block = adaptive_sampling_get_block_index(x, y);

    const uint32_t adaptive_sampling_counts = device.ptrs.stage_sample_counts[adaptive_sampling_block];

    for (uint32_t stage_id = 0; stage_id < ADAPTIVE_SAMPLER_NUM_STAGES; stage_id++) {
      const uint32_t stage_sample_offset = device.state.sample_allocation.stage_sample_offsets[stage_id + 1];
      const uint32_t stage_sample_count  = adaptive_sampling_get_stage_sample_count(adaptive_sampling_counts, stage_id);

      sample_id += stage_sample_offset * stage_sample_count;
    }
  }

  return sample_id;
}

LUMINARY_FUNCTION uint32_t adaptive_sampling_get_sample_count_from_block_index(const uint32_t adaptive_sampling_block) {
  const uint32_t adaptive_sampling_counts = device.ptrs.stage_sample_counts[adaptive_sampling_block];

  uint32_t count = device.state.adaptive_sampling_accumulated_stages[0];

  for (uint32_t stage_id = 0; stage_id < ADAPTIVE_SAMPLER_NUM_STAGES; stage_id++) {
    const uint32_t stage_sample_offset = device.state.adaptive_sampling_accumulated_stages[stage_id + 1];
    const uint32_t stage_sample_count  = adaptive_sampling_get_stage_sample_count(adaptive_sampling_counts, stage_id);

    count += stage_sample_offset * stage_sample_count;
  }

  const uint32_t this_stage_id = device.state.sample_allocation.stage_id;

  if (this_stage_id > 0) {
    count += ((adaptive_sampling_counts >> ((this_stage_id - 1) * 8)) & 0xFF) + 1;
  }
  else {
    count++;
  }

  return count;
}

LUMINARY_FUNCTION uint32_t adaptive_sampling_get_sample_count(const uint32_t x, const uint32_t y) {
  const uint32_t adaptive_sampling_block = adaptive_sampling_get_block_index(x, y);

  return adaptive_sampling_get_sample_count_from_block_index(adaptive_sampling_block);
}

LUMINARY_FUNCTION uint32_t adaptive_sampling_get_current_tasks_per_pixel(const uint32_t x, const uint32_t y) {
  const uint32_t this_stage_id = device.state.sample_allocation.stage_id;

  if (this_stage_id == 0)
    return 1;

  const uint32_t adaptive_sampling_block  = adaptive_sampling_get_block_index(x, y);
  const uint32_t adaptive_sampling_counts = device.ptrs.stage_sample_counts[adaptive_sampling_block];

  return adaptive_sampling_get_stage_sample_count(adaptive_sampling_counts, this_stage_id - 1);
}

LUMINARY_FUNCTION float adaptive_sampling_get_pixel_variance(const uint32_t x, const uint32_t y, const float inv_n, float& luminance) {
  const bool pixel_in_frame =
    (x - device.settings.window_x < device.settings.window_width) && (y - device.settings.window_y < device.settings.window_height);
  if (pixel_in_frame == false) {
    luminance = 0.0f;
    return 0.0f;
  }

  const uint32_t index = x + y * device.settings.width;

  const float red1   = __ldg(device.ptrs.frame_first_moment[FRAME_CHANNEL_RED] + index) * inv_n;
  const float green1 = __ldg(device.ptrs.frame_first_moment[FRAME_CHANNEL_GREEN] + index) * inv_n;
  const float blue1  = __ldg(device.ptrs.frame_first_moment[FRAME_CHANNEL_BLUE] + index) * inv_n;

  luminance = color_luminance(get_color(red1, green1, blue1));

  const float luminance2   = __ldg(device.ptrs.frame_second_moment_luminance + index) * inv_n;
  const float luminance_sq = color_luminance(get_color(red1 * red1, green1 * green1, blue1 * blue1));

  return fmaxf(luminance2 - luminance_sq, 0.0f);
}

LUMINARY_FUNCTION float adaptive_sampling_get_pixel_variance_and_color(const uint32_t x, const uint32_t y, const float inv_n, RGBF& color) {
  const bool pixel_in_frame =
    (x - device.settings.window_x < device.settings.window_width) && (y - device.settings.window_y < device.settings.window_height);
  if (pixel_in_frame == false) {
    color = splat_color(0.0f);
    return 0.0f;
  }

  const uint32_t index = x + y * device.settings.width;

  const float red1   = __ldg(device.ptrs.frame_first_moment[FRAME_CHANNEL_RED] + index) * inv_n;
  const float green1 = __ldg(device.ptrs.frame_first_moment[FRAME_CHANNEL_GREEN] + index) * inv_n;
  const float blue1  = __ldg(device.ptrs.frame_first_moment[FRAME_CHANNEL_BLUE] + index) * inv_n;

  color = get_color(red1, green1, blue1);

  const float luminance2   = __ldg(device.ptrs.frame_second_moment_luminance + index) * inv_n;
  const float luminance_sq = color_luminance(get_color(red1 * red1, green1 * green1, blue1 * blue1));

  return fmaxf(luminance2 - luminance_sq, 0.0f);
}

LUMINARY_KERNEL void adaptive_sampling_block_reduce_variance(const KernelArgsAdaptiveSamplingBlockReduceVariance args) {
  // Two blocks per warp
  const uint32_t adaptive_sampling_block = THREAD_ID >> (WARP_SIZE_LOG - 1);

  const uint32_t sample_count = adaptive_sampling_get_sample_count_from_block_index(adaptive_sampling_block);
  const float denominator     = 1.0f / sample_count;

  const uint32_t adaptive_sampling_y = adaptive_sampling_block / args.width;
  const uint32_t adaptive_sampling_x = adaptive_sampling_block - adaptive_sampling_y * args.width;

  const uint32_t local_x = THREAD_ID & ADAPTIVE_SAMPLING_BLOCK_SIZE_MASK;
  const uint32_t local_y = (THREAD_ID >> ADAPTIVE_SAMPLING_BLOCK_SIZE_LOG) & ADAPTIVE_SAMPLING_BLOCK_SIZE_MASK;

  const uint32_t x = (adaptive_sampling_x << ADAPTIVE_SAMPLING_BLOCK_SIZE_LOG) + local_x + device.settings.window_x;
  const uint32_t y = (adaptive_sampling_y << ADAPTIVE_SAMPLING_BLOCK_SIZE_LOG) + local_y + device.settings.window_y;

  RGBF color;
  float variance = adaptive_sampling_get_pixel_variance_and_color(x, y, denominator, color);

  if (args.exposure != 0.0f) {
    const float tonemap_compression = adaptive_sampling_compute_tonemap_compression_factor(color, args.exposure);
    variance *= tonemap_compression * tonemap_compression;
  }

  const float block_variance = fabsf(warp_reduce_max<16>(variance));

  if ((THREAD_ID & (WARP_SIZE_MASK >> 1)) == 0) {
    args.dst_block_variance[adaptive_sampling_block] = block_variance;

    atomicAdd(args.dst_sum_variance, block_variance);
  }
}

LUMINARY_KERNEL void adaptive_sampling_compute_stage_sample_counts(const KernelArgsAdaptiveSamplingComputeStageSampleCounts args) {
  const uint32_t adaptive_sampling_block = THREAD_ID;

  const float avg_variance = *args.src_sum_variance / args.num_adaptive_sampling_blocks;

  const float variance              = args.src_block_variance[adaptive_sampling_block];
  uint32_t adaptive_sampling_counts = device.ptrs.stage_sample_counts[adaptive_sampling_block];

  // Zero out all the bits not currently occupied by valid data.
  adaptive_sampling_counts &= (1u << (args.current_stage_id * 8)) - 1;

  uint32_t new_sample_count = (uint32_t) (remap(variance, 0.0f, avg_variance, 0.0f, args.avg_sampling_rate) + 0.5f);

  new_sample_count = max(new_sample_count, 1);
  new_sample_count = min(new_sample_count, args.max_sampling_rate);

  adaptive_sampling_counts |= (new_sample_count - 1) << (args.current_stage_id * 8);

  device.ptrs.stage_sample_counts[adaptive_sampling_block] = adaptive_sampling_counts;
}

LUMINARY_KERNEL void adaptive_sampling_compute_stage_total_task_counts(const KernelArgsAdaptiveSamplingComputeStageTotalTaskCounts args) {
  const uint32_t adaptive_sampling_block = THREAD_ID;

  uint32_t tasks_per_block = 0;
  if (adaptive_sampling_block < args.num_adaptive_sampling_blocks) {
    const uint32_t stage_sample_counts = device.ptrs.stage_sample_counts[adaptive_sampling_block];

    tasks_per_block = adaptive_sampling_get_stage_sample_count(stage_sample_counts, args.stage_id - 1);
    tasks_per_block *= (1u << ADAPTIVE_SAMPLING_BLOCK_SIZE_LOG) * (1u << ADAPTIVE_SAMPLING_BLOCK_SIZE_LOG);
  }

  const uint32_t total_tasks_warp = warp_reduce_sum(tasks_per_block);

  if (THREAD_ID_IN_WARP == 0 && total_tasks_warp > 0) {
    atomicAdd(args.dst, total_tasks_warp);
  }
}

LUMINARY_KERNEL void adaptive_sampling_compute_tasks_per_block(const KernelArgsAdaptiveSamplingComputeTasksPerBlock args) {
  const uint32_t adaptive_sampling_block = THREAD_ID;

  if (adaptive_sampling_block >= args.num_adaptive_sampling_blocks)
    return;

  const uint32_t stage_sample_counts = device.ptrs.stage_sample_counts[adaptive_sampling_block];

  uint32_t tasks_per_block = adaptive_sampling_get_stage_sample_count(stage_sample_counts, args.stage_id - 1);
  tasks_per_block *= (1u << ADAPTIVE_SAMPLING_BLOCK_SIZE_LOG) * (1u << ADAPTIVE_SAMPLING_BLOCK_SIZE_LOG);

  args.dst[adaptive_sampling_block] = tasks_per_block;
}

LUMINARY_KERNEL void adaptive_sampling_compute_block_sum(const KernelArgsAdaptiveSamplingComputeBlockSum args) {
  uint32_t thread_value = 0;
  if (THREAD_ID < args.thread_count)
    thread_value = args.thread_prefix_sum[THREAD_ID];

  const uint32_t warp_value = warp_reduce_sum(thread_value);

  if (THREAD_ID_IN_WARP == 0 && (WARP_ID < args.warp_count))
    args.warp_prefix_sum[WARP_ID] = warp_value;
}

LUMINARY_KERNEL void adaptive_sampling_compute_prefix_sum(const KernelArgsAdaptiveSamplingComputePrefixSum args) {
  uint32_t thread_value = 0;
  if (THREAD_ID < args.thread_count)
    thread_value = args.thread_prefix_sum[THREAD_ID];

  uint32_t thread_prefix_sum = warp_reduce_prefixsum(thread_value);

  uint32_t warp_prefix_sum = 0;
  if (WARP_ID > 0 && WARP_ID < args.warp_count)
    warp_prefix_sum = args.warp_prefix_sum[WARP_ID - 1];

  thread_prefix_sum += warp_prefix_sum;

  if (THREAD_ID < args.thread_count)
    args.thread_prefix_sum[THREAD_ID] = thread_prefix_sum;
}

LUMINARY_KERNEL void adaptive_sampling_compute_tile_block_ranges(const KernelArgsAdaptiveSamplingComputeTileBlockRanges args) {
  const uint32_t tile_id = WARP_ID;

  if (tile_id >= args.tile_count)
    return;

  const uint32_t subtile_id = THREAD_ID_IN_WARP;

  const uint32_t tasks_per_subtile = (args.tasks_per_tile + WARP_SIZE - 1) >> WARP_SIZE_LOG;

  uint32_t last_task_id;
  if (subtile_id < WARP_SIZE - 1) {
    last_task_id = tile_id * args.tasks_per_tile + tasks_per_subtile * (subtile_id + 1);
  }
  else {
    last_task_id = (tile_id + 1) * args.tasks_per_tile;
  }

  uint32_t left  = 0;
  uint32_t right = args.block_count - 1;

  while (left < right) {
    const uint32_t mid = (left + right) >> 1;

    const uint32_t block_offset = args.block_prefix_sum[mid];

    if (last_task_id < block_offset) {
      right = mid;
    }
    else {
      left = mid + 1;
    }
  }

  const uint32_t tile_end_block = left;

  args.dst[THREAD_ID] = tile_end_block;
}

#endif /* CU_LUMINARY_ADAPTIVE_SAMPLING_H */

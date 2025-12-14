#ifndef CU_LUMINARY_ADAPTIVE_SAMPLING_H
#define CU_LUMINARY_ADAPTIVE_SAMPLING_H

#include "math.cuh"
#include "memory.cuh"
#include "utils.cuh"

LUMINARY_FUNCTION float adaptive_sampling_compute_tonemap_compression_factor(const float value, const float exposure) {
  const float exposed_value    = value * exposure;
  const float tonemapped_value = exposed_value / (1.0f + exposed_value);

  return tonemapped_value / exposed_value;
}

LUMINARY_FUNCTION uint32_t adaptive_sampling_get_stage_sample_count(const uint32_t stage_sample_counts, const uint32_t stage_id) {
  return ((stage_sample_counts >> (stage_id * 8)) & 0xFF) + 1;
}

LUMINARY_FUNCTION uint32_t
  adaptive_sampling_find_block(const uint32_t task_id, const uint32_t block_start, const uint32_t block_end, uint32_t& offset) {
  // TODO: Consider using binary search
  offset = (block_start > 0) ? device.ptrs.adaptive_sampling_block_task_offsets[block_start - 1] : 0;

  uint32_t block_id;
  for (block_id = block_start; block_id <= block_end; block_id++) {
    const uint32_t block_offset = device.ptrs.adaptive_sampling_block_task_offsets[block_id];

    if (task_id < block_offset)
      break;

    offset = block_offset;
  }

  return block_id;
}

template <bool FIRST_STAGE_ONLY>
LUMINARY_FUNCTION uint32_t adapative_sampling_get_sample_offset(const uint32_t x, const uint32_t y) {
  uint32_t sample_id = device.state.sample_allocation.stage_sample_offsets[0];

  if constexpr (FIRST_STAGE_ONLY == false) {
    const uint32_t adaptive_sampling_width = device.settings.width >> ADAPTIVE_SAMPLING_BLOCK_SIZE_LOG;
    const uint32_t adaptive_sampling_x     = x >> ADAPTIVE_SAMPLING_BLOCK_SIZE_LOG;
    const uint32_t adaptive_sampling_y     = y >> ADAPTIVE_SAMPLING_BLOCK_SIZE_LOG;

    const uint32_t adaptive_sampling_block = adaptive_sampling_x + adaptive_sampling_y * adaptive_sampling_width;

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
  // TODO: This has to be different because for multi-device the sample count is not computed from the sample allocation.

  const uint32_t adaptive_sampling_counts = device.ptrs.stage_sample_counts[adaptive_sampling_block];

  uint32_t count = device.state.sample_allocation.stage_sample_offsets[0];

  for (uint32_t stage_id = 0; stage_id < ADAPTIVE_SAMPLER_NUM_STAGES; stage_id++) {
    const uint32_t stage_sample_offset = device.state.sample_allocation.stage_sample_offsets[stage_id + 1];
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
  const uint32_t adaptive_sampling_width = device.settings.width >> ADAPTIVE_SAMPLING_BLOCK_SIZE_LOG;
  const uint32_t adaptive_sampling_x     = x >> ADAPTIVE_SAMPLING_BLOCK_SIZE_LOG;
  const uint32_t adaptive_sampling_y     = y >> ADAPTIVE_SAMPLING_BLOCK_SIZE_LOG;

  const uint32_t adaptive_sampling_block = adaptive_sampling_x + adaptive_sampling_y * adaptive_sampling_width;

  return adaptive_sampling_get_sample_count_from_block_index(adaptive_sampling_block);
}

LUMINARY_FUNCTION uint32_t adaptive_sampling_get_current_tasks_per_pixel(const uint32_t x, const uint32_t y) {
  const uint32_t this_stage_id = device.state.sample_allocation.stage_id;

  if (this_stage_id == 0)
    return 1;

  const uint32_t adaptive_sampling_width = device.settings.width >> ADAPTIVE_SAMPLING_BLOCK_SIZE_LOG;
  const uint32_t adaptive_sampling_x     = x >> ADAPTIVE_SAMPLING_BLOCK_SIZE_LOG;
  const uint32_t adaptive_sampling_y     = y >> ADAPTIVE_SAMPLING_BLOCK_SIZE_LOG;

  const uint32_t adaptive_sampling_block = adaptive_sampling_x + adaptive_sampling_y * adaptive_sampling_width;

  const uint32_t adaptive_sampling_counts = device.ptrs.stage_sample_counts[adaptive_sampling_block];

  return adaptive_sampling_get_stage_sample_count(adaptive_sampling_counts, this_stage_id - 1);
}

LUMINARY_FUNCTION float adaptive_sampling_get_pixel_variance(const uint32_t x, const uint32_t y, const float inv_n, float& luminance) {
  const bool pixel_in_frame = (x < device.settings.width) && (y < device.settings.height);
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

LUMINARY_KERNEL void adaptive_sampling_block_reduce_variance(const KernelArgsAdaptiveSamplingBlockReduceVariance args) {
  const uint32_t adaptive_sampling_width =
    (device.settings.width + (1u << ADAPTIVE_SAMPLING_BLOCK_SIZE_LOG) - 1) >> ADAPTIVE_SAMPLING_BLOCK_SIZE_LOG;

  // Two sub_blocks per warp
  const uint32_t sub_block = THREAD_ID >> (WARP_SIZE_LOG - 1);

  const uint32_t adaptive_sampling_block = sub_block >> 2;
  const uint32_t local_id                = sub_block & 0b11;

  const uint32_t sample_count = adaptive_sampling_get_sample_count_from_block_index(adaptive_sampling_block);
  const float denominator     = 1.0f / sample_count;

  const uint32_t adaptive_sampling_y = adaptive_sampling_block / adaptive_sampling_width;
  const uint32_t adaptive_sampling_x = adaptive_sampling_block - adaptive_sampling_y * adaptive_sampling_width;

  const uint32_t sub_block_x =
    (adaptive_sampling_x << ADAPTIVE_SAMPLING_BLOCK_SIZE_LOG) + ((local_id & 0b1) << (ADAPTIVE_SAMPLING_BLOCK_SIZE_LOG - 1));
  const uint32_t sub_block_y =
    (adaptive_sampling_y << ADAPTIVE_SAMPLING_BLOCK_SIZE_LOG) + ((local_id >> 1) << (ADAPTIVE_SAMPLING_BLOCK_SIZE_LOG - 1));

  const uint32_t x = sub_block_x + (THREAD_ID_IN_WARP & ADAPTIVE_SAMPLING_BLOCK_SIZE_MASK);
  const uint32_t y = sub_block_y + (THREAD_ID_IN_WARP >> ADAPTIVE_SAMPLING_BLOCK_SIZE_LOG);

  float luminance;
  float variance = adaptive_sampling_get_pixel_variance(x, y, denominator, luminance);

  if (args.exposure != 0.0f) {
    const float tonemap_compression = adaptive_sampling_compute_tonemap_compression_factor(luminance, args.exposure);
    variance *= tonemap_compression * tonemap_compression;
  }

  const float rel_variance = (luminance > 0.0f) ? variance / luminance : 0.0f;

  const float sub_block_rel_variance = warp_reduce_max<16>(rel_variance);

  if ((THREAD_ID_IN_WARP & (WARP_SIZE_MASK >> 1)) == 0) {
    // TODO: This could be made faster by shuffling both values to THREAD 0 and using a 8 byte store
    args.dst_block_variance[sub_block] = sub_block_rel_variance;

    // Since variance >= 0, we can use the built-in integer atomicMax
    atomicMax((uint32_t*) args.dst_global_variance, __float_as_uint(fabsf(sub_block_rel_variance)));
  }
}

LUMINARY_KERNEL void adaptive_sampling_compute_stage_sample_counts(const KernelArgsAdaptiveSamplingComputeStageSampleCounts args) {
  const uint32_t adaptive_sampling_block = THREAD_ID;

  const uint32_t adaptive_sampling_width =
    (device.settings.width + (1u << ADAPTIVE_SAMPLING_BLOCK_SIZE_LOG) - 1) >> ADAPTIVE_SAMPLING_BLOCK_SIZE_LOG;
  const uint32_t adaptive_sampling_height =
    (device.settings.height + (1u << ADAPTIVE_SAMPLING_BLOCK_SIZE_LOG) - 1) >> ADAPTIVE_SAMPLING_BLOCK_SIZE_LOG;

  const uint32_t adaptive_sampling_y = adaptive_sampling_block / adaptive_sampling_width;
  const uint32_t adaptive_sampling_x = adaptive_sampling_block - adaptive_sampling_y * adaptive_sampling_width;

  float rel_variance = 0.0f;
  if (adaptive_sampling_x > 0) {
    if (adaptive_sampling_y > 0) {
      const float4 neighbor_variances = __ldg(((float4*) args.src_block_variance) + adaptive_sampling_block - 1 - adaptive_sampling_width);

      rel_variance = fmaxf(rel_variance, neighbor_variances.x * (1.0f / 4.0f));
      rel_variance = fmaxf(rel_variance, neighbor_variances.y * (2.0f / 4.0f));
      rel_variance = fmaxf(rel_variance, neighbor_variances.z * (2.0f / 4.0f));
      rel_variance = fmaxf(rel_variance, neighbor_variances.w * (4.0f / 4.0f));
    }

    if (true) {
      const float4 neighbor_variances = __ldg(((float4*) args.src_block_variance) + adaptive_sampling_block - 1);

      rel_variance = fmaxf(rel_variance, neighbor_variances.x * (1.0f / 2.0f));
      rel_variance = fmaxf(rel_variance, neighbor_variances.y * (2.0f / 2.0f));
      rel_variance = fmaxf(rel_variance, neighbor_variances.z * (1.0f / 2.0f));
      rel_variance = fmaxf(rel_variance, neighbor_variances.w * (2.0f / 2.0f));
    }

    if (adaptive_sampling_y < adaptive_sampling_height - 1) {
      const float4 neighbor_variances = __ldg(((float4*) args.src_block_variance) + adaptive_sampling_block - 1 + adaptive_sampling_width);

      rel_variance = fmaxf(rel_variance, neighbor_variances.x * (2.0f / 4.0f));
      rel_variance = fmaxf(rel_variance, neighbor_variances.y * (4.0f / 4.0f));
      rel_variance = fmaxf(rel_variance, neighbor_variances.z * (1.0f / 4.0f));
      rel_variance = fmaxf(rel_variance, neighbor_variances.w * (2.0f / 4.0f));
    }
  }

  if (true) {
    if (adaptive_sampling_y > 0) {
      const float4 neighbor_variances = __ldg(((float4*) args.src_block_variance) + adaptive_sampling_block - adaptive_sampling_width);

      rel_variance = fmaxf(rel_variance, neighbor_variances.x * (2.0f / 4.0f));
      rel_variance = fmaxf(rel_variance, neighbor_variances.y * (2.0f / 4.0f));
      rel_variance = fmaxf(rel_variance, neighbor_variances.z * (4.0f / 4.0f));
      rel_variance = fmaxf(rel_variance, neighbor_variances.w * (4.0f / 4.0f));
    }

    if (true) {
      const float4 neighbor_variances = __ldg(((float4*) args.src_block_variance) + adaptive_sampling_block);

      rel_variance = fmaxf(rel_variance, neighbor_variances.x * (2.0f / 2.0f));
      rel_variance = fmaxf(rel_variance, neighbor_variances.y * (2.0f / 2.0f));
      rel_variance = fmaxf(rel_variance, neighbor_variances.z * (2.0f / 2.0f));
      rel_variance = fmaxf(rel_variance, neighbor_variances.w * (2.0f / 2.0f));
    }

    if (adaptive_sampling_y < adaptive_sampling_height - 1) {
      const float4 neighbor_variances = __ldg(((float4*) args.src_block_variance) + adaptive_sampling_block + adaptive_sampling_width);

      rel_variance = fmaxf(rel_variance, neighbor_variances.x * (4.0f / 4.0f));
      rel_variance = fmaxf(rel_variance, neighbor_variances.y * (4.0f / 4.0f));
      rel_variance = fmaxf(rel_variance, neighbor_variances.z * (2.0f / 4.0f));
      rel_variance = fmaxf(rel_variance, neighbor_variances.w * (2.0f / 4.0f));
    }
  }

  if (adaptive_sampling_x < adaptive_sampling_width - 1) {
    if (adaptive_sampling_y > 0) {
      const float4 neighbor_variances = __ldg(((float4*) args.src_block_variance) + adaptive_sampling_block + 1 - adaptive_sampling_width);

      rel_variance = fmaxf(rel_variance, neighbor_variances.x * (2.0f / 4.0f));
      rel_variance = fmaxf(rel_variance, neighbor_variances.y * (1.0f / 4.0f));
      rel_variance = fmaxf(rel_variance, neighbor_variances.z * (4.0f / 4.0f));
      rel_variance = fmaxf(rel_variance, neighbor_variances.w * (2.0f / 4.0f));
    }

    if (true) {
      const float4 neighbor_variances = __ldg(((float4*) args.src_block_variance) + adaptive_sampling_block + 1);

      rel_variance = fmaxf(rel_variance, neighbor_variances.x * (2.0f / 2.0f));
      rel_variance = fmaxf(rel_variance, neighbor_variances.y * (1.0f / 2.0f));
      rel_variance = fmaxf(rel_variance, neighbor_variances.z * (2.0f / 2.0f));
      rel_variance = fmaxf(rel_variance, neighbor_variances.w * (1.0f / 2.0f));
    }

    if (adaptive_sampling_y < adaptive_sampling_height - 1) {
      const float4 neighbor_variances = __ldg(((float4*) args.src_block_variance) + adaptive_sampling_block + 1 + adaptive_sampling_width);

      rel_variance = fmaxf(rel_variance, neighbor_variances.x * (4.0f / 4.0f));
      rel_variance = fmaxf(rel_variance, neighbor_variances.y * (2.0f / 4.0f));
      rel_variance = fmaxf(rel_variance, neighbor_variances.z * (2.0f / 4.0f));
      rel_variance = fmaxf(rel_variance, neighbor_variances.w * (1.0f / 4.0f));
    }
  }

  uint32_t adaptive_sampling_counts = device.ptrs.stage_sample_counts[adaptive_sampling_block];

  // Zero out all the bits not currently occupied by valid data.
  adaptive_sampling_counts &= (1u << (args.current_stage_id * 8)) - 1;

  const float global_max_variance = __ldg(args.src_global_variance);
  uint32_t new_sample_count       = (uint32_t) (remap(rel_variance, 0.0f, global_max_variance, 0.0f, args.max_sampling_rate) + 0.5f);

  new_sample_count = max(new_sample_count, 1);
  new_sample_count = min(new_sample_count, args.max_sampling_rate);

  adaptive_sampling_counts |= (new_sample_count - 1) << (args.current_stage_id * 8);

  device.ptrs.stage_sample_counts[adaptive_sampling_block] = adaptive_sampling_counts;
}

LUMINARY_KERNEL void adaptive_sampling_compute_stage_total_task_counts(const KernelArgsAdaptiveSamplingComputeStageTotalTaskCounts args) {
  const uint32_t adaptive_sampling_width =
    (device.settings.width + (1u << ADAPTIVE_SAMPLING_BLOCK_SIZE_LOG) - 1) >> ADAPTIVE_SAMPLING_BLOCK_SIZE_LOG;
  const uint32_t adaptive_sampling_height =
    (device.settings.height + (1u << ADAPTIVE_SAMPLING_BLOCK_SIZE_LOG) - 1) >> ADAPTIVE_SAMPLING_BLOCK_SIZE_LOG;

  const uint32_t adaptive_sampling_amount = adaptive_sampling_width * adaptive_sampling_height;

  const uint32_t adaptive_sampling_block = THREAD_ID;

  uint32_t tasks_per_block = 0;
  if (adaptive_sampling_block < adaptive_sampling_amount) {
    const uint32_t stage_sample_counts = device.ptrs.stage_sample_counts[adaptive_sampling_block];

    tasks_per_block = adaptive_sampling_get_stage_sample_count(stage_sample_counts, args.stage_id - 1);
    tasks_per_block *= (1u << ADAPTIVE_SAMPLING_BLOCK_SIZE_LOG) * (1u << ADAPTIVE_SAMPLING_BLOCK_SIZE_LOG);
  }

  const uint32_t total_tasks_warp = warp_reduce_sum(tasks_per_block);

  if (THREAD_ID_IN_WARP == 0 && total_tasks_warp > 0) {
    atomicAdd(&device.ptrs.stage_total_task_counts[args.stage_id], total_tasks_warp);
  }
}

LUMINARY_KERNEL void adaptive_sampling_compute_tasks_per_block(const KernelArgsAdaptiveSamplingComputeTasksPerBlock args) {
  const uint32_t adaptive_sampling_width =
    (device.settings.width + (1u << ADAPTIVE_SAMPLING_BLOCK_SIZE_LOG) - 1) >> ADAPTIVE_SAMPLING_BLOCK_SIZE_LOG;
  const uint32_t adaptive_sampling_height =
    (device.settings.height + (1u << ADAPTIVE_SAMPLING_BLOCK_SIZE_LOG) - 1) >> ADAPTIVE_SAMPLING_BLOCK_SIZE_LOG;

  const uint32_t adaptive_sampling_amount = adaptive_sampling_width * adaptive_sampling_height;

  const uint32_t adaptive_sampling_block = THREAD_ID;

  if (adaptive_sampling_block >= adaptive_sampling_amount)
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

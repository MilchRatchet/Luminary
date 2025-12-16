#ifndef CU_KERNELS_H
#define CU_KERNELS_H

#include "adaptive_sampling.cuh"
#include "bsdf_utils.cuh"
#include "bvh.cuh"
#include "camera.cuh"
#include "cloud.cuh"
#include "medium_stack.cuh"
#include "ocean_utils.cuh"
#include "post_common.cuh"
#include "purkinje.cuh"
#include "sky.cuh"
#include "sky_utils.cuh"
#include "tonemap.cuh"
#include "utils.cuh"
#include "volume.cuh"

//
// Undersampling pattern:
// The lowest undersampling pass is used 4 times
// while the others are done 3 times.
// Once all undersampling passes have been performed,
// each pixel has exactly one sample and we continue
// without undersampling. Example for 3 passes with
// a 8x8 resolution.
//
//  3 | 1 | 2 | 1 | 3 | 1 | 2 | 1
// ---+---+---+---+---+---+---+---
//  1 | 1 | 1 | 1 | 1 | 1 | 1 | 1
// ---+---+---+---+---+---+---+---
//  2 | 1 | 2 | 1 | 2 | 1 | 2 | 1
// ---+---+---+---+---+---+---+---
//  1 | 1 | 1 | 1 | 1 | 1 | 1 | 1
// ---+---+---+---+---+---+---+---
//  3 | 1 | 2 | 1 | 3 | 1 | 2 | 1
// ---+---+---+---+---+---+---+---
//  1 | 1 | 1 | 1 | 1 | 1 | 1 | 1
// ---+---+---+---+---+---+---+---
//  2 | 1 | 2 | 1 | 2 | 1 | 2 | 1
// ---+---+---+---+---+---+---+---
//  1 | 1 | 1 | 1 | 1 | 1 | 1 | 1
//

LUMINARY_KERNEL void tasks_create() {
  HANDLE_DEVICE_ABORT();

  uint32_t task_count   = 0;
  uint32_t result_count = 0;

  const uint32_t undersampling_stage     = (device.state.undersampling & UNDERSAMPLING_STAGE_MASK) >> UNDERSAMPLING_STAGE_SHIFT;
  const uint32_t undersampling_iteration = device.state.undersampling & UNDERSAMPLING_ITERATION_MASK;

  const uint32_t undersampling_scale = 1 << undersampling_stage;

  const uint32_t undersampling_width  = (device.settings.window_width + (1 << undersampling_stage) - 1) >> undersampling_stage;
  const uint32_t undersampling_height = (device.settings.window_height + (1 << undersampling_stage) - 1) >> undersampling_stage;

  const uint32_t amount = undersampling_width * undersampling_height;

  const uint32_t window_offset_x = (device.settings.window_x >> undersampling_stage) << undersampling_stage;
  const uint32_t window_offset_y = (device.settings.window_y >> undersampling_stage) << undersampling_stage;

  const uint32_t num_threads    = NUM_THREADS;
  const uint32_t tasks_per_tile = device.config.num_tasks_per_thread * num_threads;
  const uint32_t start_pixel    = THREAD_ID + device.state.tile_id * tasks_per_tile;
  const uint32_t end_pixel      = min(amount, start_pixel + tasks_per_tile);

  for (uint32_t pixel_id = start_pixel; pixel_id < end_pixel; pixel_id += num_threads) {
    HANDLE_DEVICE_ABORT();

    uint16_t y = (uint16_t) (pixel_id / undersampling_width);
    uint16_t x = (uint16_t) (pixel_id - y * undersampling_width);

    if (undersampling_scale > 1) {
      x *= undersampling_scale;
      y *= undersampling_scale;

      x += (undersampling_iteration & 0b01) ? 0 : undersampling_scale >> 1;
      y += (undersampling_iteration & 0b10) ? 0 : undersampling_scale >> 1;
    }

    x += window_offset_x;
    y += window_offset_y;

    if (x >= device.settings.width || y >= device.settings.height)
      continue;

    if (x < device.settings.window_x || x >= device.settings.window_x + device.settings.window_width)
      continue;

    if (y < device.settings.window_y || y >= device.settings.window_y + device.settings.window_height)
      continue;

    ////////////////////////////////////////////////////////////////////
    // Sample ID
    ////////////////////////////////////////////////////////////////////

    const uint32_t sample_id = adapative_sampling_get_sample_offset<true>(x, y);

    DeviceTask task;
    task.state   = STATE_FLAG_DELTA_PATH | STATE_FLAG_CAMERA_DIRECTION | STATE_FLAG_ALLOW_EMISSION | STATE_FLAG_ALLOW_AMBIENT;
    task.path_id = path_id_get(x, y, sample_id);

    CameraSampleResult camera_result = camera_sample(task.path_id);

    const bool ray_is_valid = color_any(camera_result.weight);

    // Skip rays that haven't managed to leave the camera. During the first sample we need to write out a 0 result though.
    if (sample_id != 0 && ray_is_valid == false)
      continue;

    const uint32_t thread_result_id         = result_count++;
    const uint32_t task_result_base_address = task_get_base_address<DeviceTaskResult>(thread_result_id, TASK_STATE_BUFFER_INDEX_RESULT);

    ////////////////////////////////////////////////////////////////////
    // Task Result
    ////////////////////////////////////////////////////////////////////

    const uint32_t index = path_id_get_pixel_index(task.path_id);

    DeviceTaskResult result;

    result.color = splat_color(0.0f);
    result.index = index;

    task_result_store(task_result_base_address, result);

    // We have written out a 0 result, we can now safely skip this ray if it is not valid.
    if (ray_is_valid == false)
      continue;

    const uint32_t thread_task_id = task_count++;

    ////////////////////////////////////////////////////////////////////
    // Task
    ////////////////////////////////////////////////////////////////////

    task.origin = camera_result.origin;
    task.ray    = camera_result.ray;

    const float ambient_ior = bsdf_refraction_index_ambient(task.origin, task.ray);

    const uint32_t task_base_address = task_get_base_address(thread_task_id, TASK_STATE_BUFFER_INDEX_PRESORT);

    task_store(task_base_address, task);

    ////////////////////////////////////////////////////////////////////
    // Task Trace Result
    ////////////////////////////////////////////////////////////////////

    // Trace result does not need to be initialized

    ////////////////////////////////////////////////////////////////////
    // Task Throughput
    ////////////////////////////////////////////////////////////////////

    DeviceTaskThroughput throughput;

    throughput.record        = record_pack(camera_result.weight);
    throughput.results_index = task_result_base_address;

    task_throughput_store(task_base_address, throughput);

    ////////////////////////////////////////////////////////////////////
    // Task Medium Stack
    ////////////////////////////////////////////////////////////////////

    DeviceTaskMediumStack medium = {};

    medium_stack_ior_modify(medium, ambient_ior, true);

    if (device.fog.active) {
      medium_stack_volume_modify(medium, VOLUME_TYPE_FOG, true);
    }

    if (device.ocean.active) {
      const bool camera_is_underwater = ocean_is_underwater(task.origin);

      if (camera_is_underwater)
        medium_stack_volume_modify(medium, VOLUME_TYPE_OCEAN, true);
    }

    task_medium_store(task_base_address, medium);
  }

  device.ptrs.trace_counts[THREAD_ID]   = task_count;
  device.ptrs.results_counts[THREAD_ID] = result_count;
}

LUMINARY_KERNEL void tasks_create_adaptive_sampling() {
  HANDLE_DEVICE_ABORT();

  uint32_t task_count   = 0;
  uint32_t result_count = 0;

  const uint32_t adaptive_sampling_width =
    (device.settings.width + (1u << ADAPTIVE_SAMPLING_BLOCK_SIZE_LOG) - 1) >> ADAPTIVE_SAMPLING_BLOCK_SIZE_LOG;
  const uint32_t adaptive_sampling_height =
    (device.settings.height + (1u << ADAPTIVE_SAMPLING_BLOCK_SIZE_LOG) - 1) >> ADAPTIVE_SAMPLING_BLOCK_SIZE_LOG;

  const uint32_t num_threads       = NUM_THREADS;
  const uint32_t tasks_per_tile    = device.config.num_tasks_per_thread * num_threads;
  const uint32_t tasks_per_subtile = (tasks_per_tile + WARP_SIZE - 1) >> WARP_SIZE_LOG;

  const uint32_t task_id_offset = tasks_per_tile * device.state.tile_id;
  const uint32_t task_id_start  = task_id_offset + THREAD_ID;
  const uint32_t task_id_end    = min(task_id_offset + tasks_per_tile, device.state.sample_allocation.upper_bound_tasks_per_sample);

  for (uint32_t task_id = task_id_start; task_id < task_id_end; task_id += num_threads) {
    HANDLE_DEVICE_ABORT();

    ////////////////////////////////////////////////////////////////////
    // Adaptive Sampling Block
    ////////////////////////////////////////////////////////////////////
    const uint32_t subtile_id = task_id / tasks_per_subtile;

    const uint32_t adaptive_sampling_block_start = (subtile_id > 0) ? device.ptrs.adaptive_sampling_subtile_block_index[subtile_id - 1] : 0;
    const uint32_t adaptive_sampling_block_end   = device.ptrs.adaptive_sampling_subtile_block_index[subtile_id];

    uint32_t task_base_id;
    const uint32_t adaptive_sampling_block =
      adaptive_sampling_find_block(task_id, adaptive_sampling_block_start, adaptive_sampling_block_end, task_base_id);

    const uint32_t local_task_id = task_id - task_base_id;

    const uint32_t stage_sample_counts = device.ptrs.stage_sample_counts[adaptive_sampling_block];
    const uint32_t tasks_per_pixel =
      adaptive_sampling_get_stage_sample_count(stage_sample_counts, device.state.sample_allocation.stage_id - 1);

    const uint32_t local_pixel_id  = local_task_id / tasks_per_pixel;
    const uint32_t local_sample_id = local_task_id % tasks_per_pixel;

    const uint32_t local_x = local_pixel_id & ADAPTIVE_SAMPLING_BLOCK_SIZE_MASK;
    const uint32_t local_y = local_pixel_id >> ADAPTIVE_SAMPLING_BLOCK_SIZE_LOG;

    const uint32_t block_y = adaptive_sampling_block / adaptive_sampling_width;
    const uint32_t block_x = adaptive_sampling_block - block_y * adaptive_sampling_width;

    uint16_t x = (uint16_t) ((block_x << ADAPTIVE_SAMPLING_BLOCK_SIZE_LOG) + local_x);
    uint16_t y = (uint16_t) ((block_y << ADAPTIVE_SAMPLING_BLOCK_SIZE_LOG) + local_y);

    x += device.settings.window_x;
    y += device.settings.window_y;

    if (x >= device.settings.width || y >= device.settings.height)
      continue;

    if (x < device.settings.window_x || x >= device.settings.window_x + device.settings.window_width)
      continue;

    if (y < device.settings.window_y || y >= device.settings.window_y + device.settings.window_height)
      continue;

    ////////////////////////////////////////////////////////////////////
    // Sample ID
    ////////////////////////////////////////////////////////////////////

    const uint32_t sample_id = adapative_sampling_get_sample_offset<false>(x, y) + local_sample_id;

    DeviceTask task;
    task.state   = STATE_FLAG_DELTA_PATH | STATE_FLAG_CAMERA_DIRECTION | STATE_FLAG_ALLOW_EMISSION | STATE_FLAG_ALLOW_AMBIENT;
    task.path_id = path_id_get(x, y, sample_id);

    CameraSampleResult camera_result = camera_sample(task.path_id);

    const bool ray_is_valid = color_any(camera_result.weight);

    // Skip rays that haven't managed to leave the camera. During the first sample we need to write out a 0 result though.
    if (sample_id != 0 && ray_is_valid == false)
      continue;

    const uint32_t thread_result_id         = result_count++;
    const uint32_t task_result_base_address = task_get_base_address<DeviceTaskResult>(thread_result_id, TASK_STATE_BUFFER_INDEX_RESULT);

    ////////////////////////////////////////////////////////////////////
    // Task Result
    ////////////////////////////////////////////////////////////////////

    const uint32_t index = path_id_get_pixel_index(task.path_id);

    DeviceTaskResult result;

    result.color = splat_color(0.0f);
    result.index = index;

    task_result_store(task_result_base_address, result);

    // We have written out a 0 result, we can now safely skip this ray if it is not valid.
    if (ray_is_valid == false)
      continue;

    const uint32_t thread_task_id = task_count++;

    ////////////////////////////////////////////////////////////////////
    // Task
    ////////////////////////////////////////////////////////////////////

    task.origin = camera_result.origin;
    task.ray    = camera_result.ray;

    const float ambient_ior = bsdf_refraction_index_ambient(task.origin, task.ray);

    const uint32_t task_base_address = task_get_base_address(thread_task_id, TASK_STATE_BUFFER_INDEX_PRESORT);

    task_store(task_base_address, task);

    ////////////////////////////////////////////////////////////////////
    // Task Trace Result
    ////////////////////////////////////////////////////////////////////

    // Trace result does not need to be initialized

    ////////////////////////////////////////////////////////////////////
    // Task Throughput
    ////////////////////////////////////////////////////////////////////

    DeviceTaskThroughput throughput;

    throughput.record        = record_pack(camera_result.weight);
    throughput.results_index = task_result_base_address;

    task_throughput_store(task_base_address, throughput);

    ////////////////////////////////////////////////////////////////////
    // Task Medium Stack
    ////////////////////////////////////////////////////////////////////

    DeviceTaskMediumStack medium = {};

    medium_stack_ior_modify(medium, ambient_ior, true);

    if (device.fog.active) {
      medium_stack_volume_modify(medium, VOLUME_TYPE_FOG, true);
    }

    if (device.ocean.active) {
      const bool camera_is_underwater = ocean_is_underwater(task.origin);

      if (camera_is_underwater)
        medium_stack_volume_modify(medium, VOLUME_TYPE_OCEAN, true);
    }

    task_medium_store(task_base_address, medium);
  }

  device.ptrs.trace_counts[THREAD_ID]   = task_count;
  device.ptrs.results_counts[THREAD_ID] = result_count;
}

LUMINARY_KERNEL void sky_process_inscattering_events() {
  HANDLE_DEVICE_ABORT();

  const int task_count = device.ptrs.trace_counts[THREAD_ID];

  LUMINARY_ASSUME(task_count <= MAXIMUM_TASKS_PER_THREAD);

  for (int i = 0; i < task_count; i++) {
    HANDLE_DEVICE_ABORT();

    const uint32_t task_base_address = task_get_base_address(i, TASK_STATE_BUFFER_INDEX_PRESORT);
    DeviceTask task                  = task_load(task_base_address);
    DeviceTaskTrace trace            = task_trace_load(task_base_address);

    if (trace.handle.instance_id == HIT_TYPE_SKY)
      continue;

    const vec3 sky_origin          = world_to_sky_transform(task.origin);
    const float inscattering_limit = world_to_sky_scale(trace.depth);

    DeviceTaskThroughput throughput = task_throughput_load(task_base_address);

    RGBF record = record_unpack(throughput.record);

    const RGBF inscattering = sky_trace_inscattering(sky_origin, task.ray, inscattering_limit, record, task.path_id);

    write_beauty_buffer(inscattering, throughput.results_index);

    throughput.record = record_pack(record);

    task_throughput_store(task_base_address, throughput);
  }
}

#define SORT_TYPE_NUM_BITS 12
#define SORT_TYPE_MASK ((1 << SORT_TYPE_NUM_BITS) - 1)

LUMINARY_KERNEL void tasks_sort() {
  HANDLE_DEVICE_ABORT();

  uint32_t original_thread_task_counts[SHADING_TASK_INDEX_TOTAL];

  for (uint32_t task_index = 0; task_index < SHADING_TASK_INDEX_TOTAL; task_index++) {
    original_thread_task_counts[task_index] = __ldcs(device.ptrs.task_counts + TASK_ADDRESS_OFFSET_IMPL(task_index));
  }

  uint32_t warp_counts[SHADING_TASK_INDEX_TOTAL];

  for (uint32_t task_index = 0; task_index < SHADING_TASK_INDEX_TOTAL; task_index++) {
    warp_counts[task_index] = warp_reduce_sum(original_thread_task_counts[task_index]);
  }

  uint32_t warp_offsets[SHADING_TASK_INDEX_TOTAL];

  for (uint32_t task_index = 0; task_index < SHADING_TASK_INDEX_TOTAL; task_index++) {
    warp_offsets[task_index] = (task_index > 0) ? (warp_offsets[task_index - 1] + warp_counts[task_index - 1]) : 0;
  }

  const uint32_t thread_id_in_warp = THREAD_ID & WARP_SIZE_MASK;
  uint32_t thread_shift            = WARP_SIZE - 1 - thread_id_in_warp;

  uint32_t read_thread_counts[SHADING_TASK_INDEX_TOTAL];

  for (uint32_t task_index = 0; task_index < SHADING_TASK_INDEX_TOTAL; task_index++) {
    read_thread_counts[task_index]                                = (warp_counts[task_index] + thread_shift) >> WARP_SIZE_LOG;
    device.ptrs.task_counts[TASK_ADDRESS_OFFSET_IMPL(task_index)] = read_thread_counts[task_index];

    thread_shift = (thread_shift + (warp_counts[task_index] & WARP_SIZE_MASK)) & WARP_SIZE_MASK;
  }

  uint32_t read_thread_offsets[SHADING_TASK_INDEX_TOTAL];

  for (uint32_t task_index = 0; task_index < SHADING_TASK_INDEX_TOTAL; task_index++) {
    read_thread_offsets[task_index] = (task_index > 0) ? (read_thread_offsets[task_index - 1] + read_thread_counts[task_index - 1]) : 0;
    device.ptrs.task_offsets[TASK_ADDRESS_OFFSET_IMPL(task_index)] = read_thread_offsets[task_index];
  }

  const uint32_t task_count          = __ldcs(device.ptrs.trace_counts + THREAD_ID);
  const uint32_t max_warp_task_count = warp_reduce_max(task_count);

  LUMINARY_ASSUME(max_warp_task_count <= MAXIMUM_TASKS_PER_THREAD);

  uint64_t offset_mask = 0;

  for (uint32_t task_id = 0; task_id < max_warp_task_count; task_id++) {
    // Due to the prefix sum, all threads must always keep participating, the actual load/stores must hence be predicated off.
    const bool thread_predicate = task_id < task_count;

    DeviceTask task;
    DeviceTaskTrace trace;
    DeviceTaskThroughput throughput;
    DeviceTaskMediumStack medium;
    ShadingTaskIndex index = SHADING_TASK_INDEX_INVALID;

    if (thread_predicate) {
      const uint32_t src_task_base_address = task_get_base_address(task_id, TASK_STATE_BUFFER_INDEX_PRESORT);

      task       = task_load(src_task_base_address);
      trace      = task_trace_load(src_task_base_address);
      throughput = task_throughput_load(src_task_base_address);
      medium     = task_medium_load(src_task_base_address);

      index = shading_task_index_from_instance_id(trace.handle.instance_id);
    }

    const uint64_t index_entry     = (index != SHADING_TASK_INDEX_INVALID) ? ((uint64_t) 1) << (index * SORT_TYPE_NUM_BITS) : 0;
    const uint64_t offset_result   = offset_mask + warp_reduce_prefixsum(index_entry) - index_entry;
    const uint64_t warp_sum_offset = warp_reduce_sum(index_entry);
    offset_mask += warp_sum_offset;

    // It is important that threads participate in the reduction even when their task is invalid so their mask stays synced.
    if ((thread_predicate == false) || (index == SHADING_TASK_INDEX_INVALID))
      continue;

    const uint32_t dst_offset = warp_offsets[index] + (((uint32_t) (offset_result >> (index * SORT_TYPE_NUM_BITS))) & SORT_TYPE_MASK);

    const uint32_t dst_task_base_address = task_arbitrary_warp_address(dst_offset, TASK_STATE_BUFFER_INDEX_POSTSORT);

    task_store(dst_task_base_address, task);
    task_trace_store(dst_task_base_address, trace);
    task_throughput_store(dst_task_base_address, throughput);
    task_medium_store(dst_task_base_address, medium);
  }

  device.ptrs.trace_counts[THREAD_ID] = 0;
}

LUMINARY_FUNCTION RGBF final_image_get_undersampling_sample(
  const uint16_t source_x, const uint16_t source_y, const uint32_t output_scale, const uint32_t undersampling_iteration) {
  const uint32_t offset_x = (undersampling_iteration & 0b01) ? 0 : output_scale >> 1;
  const uint32_t offset_y = (undersampling_iteration & 0b10) ? 0 : output_scale >> 1;

  const uint32_t pixel_x = min(source_x + offset_x, device.settings.width - 1);
  const uint32_t pixel_y = min(source_y + offset_y, device.settings.height - 1);

  const uint32_t index = pixel_x + pixel_y * device.settings.width;

  const float red   = __ldg(device.ptrs.frame_result[FRAME_CHANNEL_RED] + index);
  const float green = __ldg(device.ptrs.frame_result[FRAME_CHANNEL_GREEN] + index);
  const float blue  = __ldg(device.ptrs.frame_result[FRAME_CHANNEL_BLUE] + index);

  return get_color(red, green, blue);
}

LUMINARY_KERNEL void generate_final_image(const KernelArgsGenerateFinalImage args) {
  HANDLE_DEVICE_ABORT();

  const uint32_t undersampling_stage = (args.undersampling & UNDERSAMPLING_STAGE_MASK) >> UNDERSAMPLING_STAGE_SHIFT;

  const uint32_t undersampling_output = max(undersampling_stage, device.settings.supersampling);
  const uint32_t undersampling_input  = undersampling_stage;

  const uint32_t output_scale = 1u << (undersampling_output - undersampling_input);

  const uint32_t output_width  = device.settings.width >> undersampling_output;
  const uint32_t output_height = device.settings.height >> undersampling_output;

  const uint32_t input_width  = device.settings.width >> undersampling_input;
  const uint32_t input_height = device.settings.height >> undersampling_input;

  const uint32_t output_amount = output_width * output_height;

  for (uint32_t output_pixel = THREAD_ID; output_pixel < output_amount; output_pixel += blockDim.x * gridDim.x) {
    const uint16_t y = (uint16_t) (output_pixel / output_width);
    const uint16_t x = (uint16_t) (output_pixel - y * output_width);

    RGBF color = get_color(0.0f, 0.0f, 0.0f);

    const uint16_t source_x = x * output_scale;
    const uint16_t source_y = y * output_scale;

    for (uint32_t yi = 0; yi < output_scale; yi++) {
      for (uint32_t xi = 0; xi < output_scale; xi++) {
        const uint32_t pixel_x = min(source_x + xi, input_width - 1);
        const uint32_t pixel_y = min(source_y + yi, input_height - 1);

        const uint32_t index = pixel_x + pixel_y * input_width;

        const float red   = __ldg(device.ptrs.frame_result[FRAME_CHANNEL_RED] + index);
        const float green = __ldg(device.ptrs.frame_result[FRAME_CHANNEL_GREEN] + index);
        const float blue  = __ldg(device.ptrs.frame_result[FRAME_CHANNEL_BLUE] + index);

        RGBF pixel = get_color(red, green, blue);
        pixel      = tonemap_apply(pixel, pixel_x, pixel_y, args.color_correction, args.agx_params);

        color = add_color(color, pixel);
      }
    }

    color = scale_color(color, 1.0f / (output_scale * output_scale));

    const uint32_t dst_index = x + y * output_width;

    __stcs(device.ptrs.frame_output[FRAME_CHANNEL_RED] + dst_index, color.r);
    __stcs(device.ptrs.frame_output[FRAME_CHANNEL_GREEN] + dst_index, color.g);
    __stcs(device.ptrs.frame_output[FRAME_CHANNEL_BLUE] + dst_index, color.b);
  }
}

LUMINARY_KERNEL void convert_RGBF_to_ARGB8(const KernelArgsConvertRGBFToARGB8 args) {
  HANDLE_DEVICE_ABORT();

  uint32_t id = THREAD_ID;

  const uint32_t amount = args.width * args.height;
  const float scale_x   = 1.0f / (args.width - 1);
  const float scale_y   = 1.0f / (args.height - 1);

  const uint32_t undersampling_stage = (args.undersampling & UNDERSAMPLING_STAGE_MASK) >> UNDERSAMPLING_STAGE_SHIFT;

  const uint32_t undersampling_output = max(undersampling_stage, device.settings.supersampling);

  const uint32_t dst_width  = device.settings.width >> device.settings.supersampling;
  const uint32_t dst_height = device.settings.height >> device.settings.supersampling;

  const uint32_t src_width = device.settings.width >> undersampling_output;

  const uint32_t undersampling_mem = undersampling_output - device.settings.supersampling;

  const float mem_scale    = 1.0f / (1 << undersampling_mem);
  const bool scaled_output = (args.width != dst_width) || (args.height != dst_height);

  while (id < amount) {
    const uint32_t y = id / args.width;
    const uint32_t x = id - y * args.width;

    RGBF pixel;
    if (scaled_output) {
      const float sx = x * scale_x;
      const float sy = y * scale_y;

      const float red   = post_sample_buffer_clamp(device.ptrs.frame_output[FRAME_CHANNEL_RED], sx, sy, dst_width, dst_height, mem_scale);
      const float green = post_sample_buffer_clamp(device.ptrs.frame_output[FRAME_CHANNEL_GREEN], sx, sy, dst_width, dst_height, mem_scale);
      const float blue  = post_sample_buffer_clamp(device.ptrs.frame_output[FRAME_CHANNEL_BLUE], sx, sy, dst_width, dst_height, mem_scale);

      pixel = get_color(red, green, blue);
    }
    else {
      const uint32_t src_x = x >> undersampling_mem;
      const uint32_t src_y = y >> undersampling_mem;

      const uint32_t src_index = src_x + src_y * src_width;

      const float red   = __ldg(device.ptrs.frame_output[FRAME_CHANNEL_RED] + src_index);
      const float green = __ldg(device.ptrs.frame_output[FRAME_CHANNEL_GREEN] + src_index);
      const float blue  = __ldg(device.ptrs.frame_output[FRAME_CHANNEL_BLUE] + src_index);

      pixel = get_color(red, green, blue);
    }

    switch (args.filter) {
      case LUMINARY_FILTER_NONE:
        break;
      case LUMINARY_FILTER_GRAY:
        pixel = filter_gray(pixel);
        break;
      case LUMINARY_FILTER_SEPIA:
        pixel = filter_sepia(pixel);
        break;
      case LUMINARY_FILTER_GAMEBOY:
        pixel = filter_gameboy(pixel, x, y);
        break;
      case LUMINARY_FILTER_2BITGRAY:
        pixel = filter_2bitgray(pixel, x, y);
        break;
      case LUMINARY_FILTER_CRT:
        pixel = filter_crt(pixel, x, y);
        break;
      case LUMINARY_FILTER_BLACKWHITE:
        pixel = filter_blackwhite(pixel, x, y);
        break;
    }

    const float dither = (device.camera.dithering) ? random_dither_mask(x, y) : 0.5f;

    pixel.r = fmaxf(0.0f, fminf(255.9999f, dither + 255.0f * linearRGB_to_SRGB(pixel.r)));
    pixel.g = fmaxf(0.0f, fminf(255.9999f, dither + 255.0f * linearRGB_to_SRGB(pixel.g)));
    pixel.b = fmaxf(0.0f, fminf(255.9999f, dither + 255.0f * linearRGB_to_SRGB(pixel.b)));

    ARGB8 converted_pixel;
    converted_pixel.a = 0xFF;
    converted_pixel.r = (uint8_t) pixel.r;
    converted_pixel.g = (uint8_t) pixel.g;
    converted_pixel.b = (uint8_t) pixel.b;

    args.dst[x + y * args.width] = converted_pixel;

    id += blockDim.x * gridDim.x;
  }
}

LUMINARY_KERNEL void buffer_add(const KernelArgsBufferAdd args) {
  static_assert(THREADS_PER_BLOCK == 128, "This assumes this threads per blocks value.");
  uint32_t offset = args.base_offset + THREAD_ID * 4;

  if (offset >= args.num_elements)
    return;

  if (offset + 4 >= args.num_elements) {
    for (; offset < args.num_elements; offset++) {
      const float src_data = __ldcs(args.src + offset);
      float dst_data       = __ldcs(args.dst + offset);

      dst_data += src_data;

      __stwt(args.dst + offset, dst_data);
    }

    return;
  }

  const float4 src_data = __ldcs((float4*) (args.src + offset));
  float4 dst_data       = __ldcs((float4*) (args.dst + offset));

  dst_data.x += src_data.x;
  dst_data.y += src_data.y;
  dst_data.z += src_data.z;
  dst_data.w += src_data.w;

  __stwt((float4*) (args.dst + offset), dst_data);
}

#endif /* CU_KERNELS_H */

#include "device_adaptive_sampler.h"

#include "device.h"
#include "internal_error.h"
#include "kernel.h"
#include "kernel_args.h"

LuminaryResult adaptive_sampler_create(AdaptiveSampler** sampler) {
  __CHECK_NULL_ARGUMENT(sampler);

  __FAILURE_HANDLE(host_malloc(sampler, sizeof(AdaptiveSampler)));
  memset(*sampler, 0, sizeof(AdaptiveSampler));

  return LUMINARY_SUCCESS;
}

LuminaryResult adaptive_sampler_get_buffer_sizes(AdaptiveSampler* sampler, DeviceAdaptiveSamplerBufferSizes* sizes) {
  __CHECK_NULL_ARGUMENT(sampler);
  __CHECK_NULL_ARGUMENT(sizes);

  sizes->stage_sample_counts_size     = sizeof(uint32_t) * sampler->width * sampler->height;
  sizes->stage_total_task_counts_size = sizeof(uint32_t) * (ADAPTIVE_SAMPLER_NUM_STAGES + 1);

  return LUMINARY_SUCCESS;
}

LuminaryResult adaptive_sampler_start_sampling(AdaptiveSampler* sampler, uint32_t width, uint32_t height) {
  __CHECK_NULL_ARGUMENT(sampler);

  sampler->width  = (width + ((1 << ADAPTIVE_SAMPLING_BLOCK_SIZE_LOG) - 1)) >> ADAPTIVE_SAMPLING_BLOCK_SIZE_LOG;
  sampler->height = (height + ((1 << ADAPTIVE_SAMPLING_BLOCK_SIZE_LOG) - 1)) >> ADAPTIVE_SAMPLING_BLOCK_SIZE_LOG;

  memset(&sampler->allocator, 0, sizeof(DeviceSampleAllocation));

  sampler->allocator.upper_bound_tasks_per_sample = (sampler->width * sampler->height) << (2 * ADAPTIVE_SAMPLING_BLOCK_SIZE_LOG);
  sampler->queued_stage_build                     = ADAPTIVE_SAMPLING_STAGE_INVALID;

  return LUMINARY_SUCCESS;
}

LuminaryResult adaptive_sampler_allocate_sample(AdaptiveSampler* sampler, DeviceSampleAllocation* allocation, uint32_t num_samples) {
  __CHECK_NULL_ARGUMENT(sampler);
  __CHECK_NULL_ARGUMENT(allocation);

  __DEBUG_ASSERT(num_samples < 256);
  __DEBUG_ASSERT(sampler->width > 0 && sampler->height > 0);

  memcpy(allocation, &sampler->allocator, sizeof(DeviceSampleAllocation));

  allocation->num_samples = num_samples;

  sampler->allocator.stage_sample_offsets[sampler->allocator.stage_id] += num_samples;
  sampler->allocator.global_sample_id += num_samples;

  return LUMINARY_SUCCESS;
}

LuminaryResult adaptive_sampler_get_task_count_upper_bound(AdaptiveSampler* sampler, uint32_t* task_count, uint8_t stage_id) {
  __CHECK_NULL_ARGUMENT(sampler);
  __CHECK_NULL_ARGUMENT(task_count);

  __DEBUG_ASSERT(stage_id <= ADAPTIVE_SAMPLER_NUM_STAGES + 1);

  *task_count = sampler->stage_total_task_counts[stage_id];

  return LUMINARY_SUCCESS;
}

static LuminaryResult _adaptive_sampler_finalize_build(AdaptiveSampler* sampler, Device* device) {
  __CHECK_NULL_ARGUMENT(sampler);
  __CHECK_NULL_ARGUMENT(device);

  CUresult result = cuEventQuery(sampler->stage_build_event);

  if (result == CUDA_ERROR_NOT_READY)
    return LUMINARY_SUCCESS;

  CUDA_FAILURE_HANDLE(result);

  sampler->allocator.stage_id++;
  sampler->allocator.upper_bound_tasks_per_sample = sampler->stage_total_task_counts[sampler->allocator.stage_id];

  sampler->queued_stage_build = ADAPTIVE_SAMPLING_STAGE_INVALID;

  return LUMINARY_SUCCESS;
}

LuminaryResult adaptive_sampler_compute_next_stage(AdaptiveSampler* sampler, Device* device, uint8_t stage_id) {
  __CHECK_NULL_ARGUMENT(sampler);
  __CHECK_NULL_ARGUMENT(device);

  __DEBUG_ASSERT(device->is_main_device);

  if (sampler->allocator.stage_id == stage_id)
    return LUMINARY_SUCCESS;

  if (sampler->queued_stage_build == stage_id) {
    __FAILURE_HANDLE(_adaptive_sampler_finalize_build(sampler, device));
    return LUMINARY_SUCCESS;
  }

  __DEBUG_ASSERT(sampler->queued_stage_build == ADAPTIVE_SAMPLING_STAGE_INVALID);

  sampler->queued_stage_build = stage_id;

  DeviceAdaptiveSamplerBufferSizes buffer_sizes;
  __FAILURE_HANDLE(adaptive_sampler_get_buffer_sizes(sampler, &buffer_sizes));

  if (buffer_sizes.stage_sample_counts_size != sampler->allocated_stage_sample_counts_size) {
    sampler->allocated_stage_sample_counts_size = buffer_sizes.stage_sample_counts_size;

    if (sampler->stage_sample_counts)
      __FAILURE_HANDLE(device_free_staging(&sampler->stage_sample_counts));

    __FAILURE_HANDLE(device_malloc_staging(
      &sampler->stage_sample_counts, buffer_sizes.stage_sample_counts_size,
      DEVICE_MEMORY_STAGING_FLAG_PCIE_TRANSFER_ONLY | DEVICE_MEMORY_STAGING_FLAG_SHARED));
  }

  if (sampler->stage_total_task_counts == (uint32_t*) 0) {
    __FAILURE_HANDLE(
      device_malloc_staging(&sampler->stage_total_task_counts, buffer_sizes.stage_sample_counts_size, DEVICE_MEMORY_STAGING_FLAG_SHARED));

    sampler->stage_total_task_counts[0] = (sampler->width * sampler->height) << (2 * ADAPTIVE_SAMPLING_BLOCK_SIZE_LOG);
  }

  if (sampler->stage_build_event == (CUevent) 0) {
    CUDA_FAILURE_HANDLE(cuEventCreate(&sampler->stage_build_event, CU_EVENT_DISABLE_TIMING));
  }

  const uint32_t num_adaptive_sampling_blocks = sampler->width * sampler->height;
  const uint32_t warps_per_block              = THREADS_PER_BLOCK >> WARP_SIZE_LOG;

  {
    KernelArgsAdaptiveSamplingComputeStageSampleCounts args;
    args.current_stage_id = sampler->allocator.stage_id;

    const uint32_t num_blocks = (num_adaptive_sampling_blocks + warps_per_block - 1) / warps_per_block;

    __FAILURE_HANDLE(kernel_execute_custom(
      device->cuda_kernels[CUDA_KERNEL_TYPE_ADAPTIVE_SAMPLING_COMPUTE_STAGE_SAMPLE_COUNTS], THREADS_PER_BLOCK, 1, 1, num_blocks, 1, 1,
      &args, device->stream_main));
  }

  {
    KernelArgsAdaptiveSamplingComputeStageTotalTaskCounts args;
    args.stage_id = sampler->allocator.stage_id + 1;

    const uint32_t num_blocks = (((num_adaptive_sampling_blocks + WARP_SIZE - 1) >> WARP_SIZE_LOG) + warps_per_block - 1) / warps_per_block;

    __FAILURE_HANDLE(kernel_execute_custom(
      device->cuda_kernels[CUDA_KERNEL_TYPE_ADAPTIVE_SAMPLING_COMPUTE_STAGE_TOTAL_TASK_COUNTS], THREADS_PER_BLOCK, 1, 1, num_blocks, 1, 1,
      &args, device->stream_main));
  }

  __FAILURE_HANDLE(device_download(
    sampler->stage_sample_counts, device->adaptive_sampler->stage_sample_counts, 0, buffer_sizes.stage_sample_counts_size,
    device->stream_main));
  __FAILURE_HANDLE(device_download(
    sampler->stage_total_task_counts, device->adaptive_sampler->stage_total_task_counts, 0, buffer_sizes.stage_total_task_counts_size,
    device->stream_main));

  CUDA_FAILURE_HANDLE(cuEventRecord(sampler->stage_build_event, device->stream_main));

  return LUMINARY_SUCCESS;
}

LuminaryResult adaptive_sampler_unload(AdaptiveSampler* sampler) {
  __CHECK_NULL_ARGUMENT(sampler);

  if (sampler->stage_sample_counts)
    __FAILURE_HANDLE(device_free_staging(&sampler->stage_sample_counts));

  if (sampler->stage_total_task_counts)
    __FAILURE_HANDLE(device_free_staging(&sampler->stage_total_task_counts));

  if (sampler->stage_build_event != (CUevent) 0) {
    CUDA_FAILURE_HANDLE(cuEventDestroy(sampler->stage_build_event));
    sampler->stage_build_event = (CUevent) 0;
  }

  sampler->allocated_stage_sample_counts_size = 0;
  sampler->width                              = 0;
  sampler->height                             = 0;
  sampler->queued_stage_build                 = ADAPTIVE_SAMPLING_STAGE_INVALID;

  return LUMINARY_SUCCESS;
}

LuminaryResult adaptive_sampler_destroy(AdaptiveSampler** sampler) {
  __CHECK_NULL_ARGUMENT(sampler);

  __DEBUG_ASSERT((*sampler)->stage_sample_counts == (uint32_t*) 0);
  __DEBUG_ASSERT((*sampler)->stage_total_task_counts == (uint32_t*) 0);

  __FAILURE_HANDLE(host_free(sampler));

  return LUMINARY_SUCCESS;
}

static uint32_t _device_adaptive_sampler_compute_prefix_mip_count(const uint32_t width, const uint32_t height) {
  uint32_t count = width * height;

  if (count == 0)
    return 1;

  uint32_t i = 0;

  while (count > 0) {
    i++;
    count = count >> WARP_SIZE_LOG;
  }

  return i + 1;
}

static LuminaryResult _device_adaptive_sampler_reset(DeviceAdaptiveSampler* sampler) {
  __CHECK_NULL_ARGUMENT(sampler);

  if (sampler->stage_sample_counts) {
    __FAILURE_HANDLE(device_free(&sampler->stage_sample_counts));
  }

  if (sampler->stage_total_task_counts) {
    __FAILURE_HANDLE(device_free(&sampler->stage_total_task_counts));
  }

  for (uint32_t mip_id = 0; mip_id < sampler->num_prefix_mips; mip_id++) {
    __FAILURE_HANDLE(device_free(&sampler->stage_prefix_mips[mip_id]));
  }

  if (sampler->stage_prefix_mips) {
    __FAILURE_HANDLE(host_free(&sampler->stage_prefix_mips));
  }

  if (sampler->subtile_last_blocks) {
    __FAILURE_HANDLE(device_free(&sampler->subtile_last_blocks));
  }

  sampler->width                  = 0;
  sampler->height                 = 0;
  sampler->num_prefix_mips        = 0;
  sampler->num_allocated_subtiles = 0;

  return LUMINARY_SUCCESS;
}

LuminaryResult device_adaptive_sampler_create(DeviceAdaptiveSampler** sampler) {
  __CHECK_NULL_ARGUMENT(sampler);

  __FAILURE_HANDLE(host_malloc(sampler, sizeof(DeviceAdaptiveSampler)));
  memset(*sampler, 0, sizeof(DeviceAdaptiveSampler));

  return LUMINARY_SUCCESS;
}

LuminaryResult device_adaptive_sampler_reset(DeviceAdaptiveSampler* sampler) {
  __CHECK_NULL_ARGUMENT(sampler);

  sampler->allocated_stage_id = ADAPTIVE_SAMPLING_STAGE_INVALID;

  return LUMINARY_SUCCESS;
}

LuminaryResult device_adaptive_sampler_update(
  DeviceAdaptiveSampler* sampler, Device* device, AdaptiveSampler* shared_sampler, bool* buffers_have_changed) {
  __CHECK_NULL_ARGUMENT(sampler);
  __CHECK_NULL_ARGUMENT(device);
  __CHECK_NULL_ARGUMENT(shared_sampler);

  *buffers_have_changed = false;

  uint32_t tile_count;
  __FAILURE_HANDLE(device_renderer_get_tile_count(device->renderer, device, 0, &tile_count));

  const uint32_t subtile_count = tile_count * WARP_SIZE;

  if (sampler->width != shared_sampler->width || sampler->height != shared_sampler->height) {
    __FAILURE_HANDLE(_device_adaptive_sampler_reset(sampler));

    sampler->width  = shared_sampler->width;
    sampler->height = shared_sampler->height;

    DeviceAdaptiveSamplerBufferSizes buffer_sizes;
    __FAILURE_HANDLE(adaptive_sampler_get_buffer_sizes(shared_sampler, &buffer_sizes));

    __FAILURE_HANDLE(device_malloc(&sampler->stage_sample_counts, buffer_sizes.stage_sample_counts_size));
    __FAILURE_HANDLE(device_malloc(&sampler->stage_total_task_counts, buffer_sizes.stage_total_task_counts_size));

    sampler->num_prefix_mips = _device_adaptive_sampler_compute_prefix_mip_count(sampler->width, sampler->height);

    __FAILURE_HANDLE(host_malloc(&sampler->stage_prefix_mips, sizeof(uint32_t*) * sampler->num_prefix_mips));
    memset(sampler->stage_prefix_mips, 0, sizeof(uint32_t*) * sampler->num_prefix_mips);

    uint32_t num_entries_mip = sampler->width * sampler->height;

    for (uint32_t mip_id = 0; mip_id < sampler->num_prefix_mips; mip_id++) {
      __FAILURE_HANDLE(device_malloc(&sampler->stage_prefix_mips[mip_id], sizeof(uint32_t) * num_entries_mip));

      num_entries_mip = (num_entries_mip + WARP_SIZE - 1) >> WARP_SIZE_LOG;
    }

    __FAILURE_HANDLE(device_malloc(&sampler->subtile_last_blocks, sizeof(uint32_t) * subtile_count));

    sampler->num_allocated_subtiles = subtile_count;
    *buffers_have_changed           = true;
  }

  return LUMINARY_SUCCESS;
}

LuminaryResult device_adaptive_sampler_get_device_buffer_ptrs(DeviceAdaptiveSampler* sampler, DeviceAdaptiveSamplerDeviceBufferPtrs* ptrs) {
  __CHECK_NULL_ARGUMENT(sampler);
  __CHECK_NULL_ARGUMENT(ptrs);

  ptrs->stage_sample_counts                   = DEVICE_CUPTR(sampler->stage_sample_counts);
  ptrs->stage_total_task_counts               = DEVICE_CUPTR(sampler->stage_total_task_counts);
  ptrs->adaptive_sampling_block_task_offsets  = DEVICE_CUPTR(sampler->stage_prefix_mips[0]);
  ptrs->adaptive_sampling_subtile_block_index = DEVICE_CUPTR(sampler->subtile_last_blocks);

  return LUMINARY_SUCCESS;
}

LuminaryResult device_adaptive_sampler_ensure_stage(DeviceAdaptiveSampler* sampler, Device* device, bool* buffers_have_changed) {
  __CHECK_NULL_ARGUMENT(sampler);
  __CHECK_NULL_ARGUMENT(device);

  *buffers_have_changed = false;

  const uint8_t stage_id = device->renderer->sample_allocation.stage_id;

  if (sampler->allocated_stage_id != ADAPTIVE_SAMPLING_STAGE_INVALID && sampler->allocated_stage_id >= stage_id)
    return LUMINARY_SUCCESS;

  uint32_t tile_count;
  __FAILURE_HANDLE(device_renderer_get_tile_count(device->renderer, device, 0, &tile_count));

  const uint32_t subtile_count = tile_count * WARP_SIZE;

  if (subtile_count >= sampler->num_allocated_subtiles) {
    if (sampler->subtile_last_blocks != (uint32_t*) 0) {
      __FAILURE_HANDLE(device_free(&sampler->subtile_last_blocks));
    }

    __FAILURE_HANDLE(device_malloc(&sampler->subtile_last_blocks, sizeof(uint32_t) * subtile_count));

    sampler->num_allocated_subtiles = subtile_count;
    *buffers_have_changed           = true;
  }

  const uint32_t num_adaptive_sampling_blocks = sampler->width * sampler->height;

  {
    KernelArgsAdaptiveSamplingComputeTasksPerBlock args;
    args.stage_id = stage_id;
    args.dst      = DEVICE_PTR(sampler->stage_prefix_mips[0]);

    const uint32_t num_blocks = (num_adaptive_sampling_blocks + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    __FAILURE_HANDLE(kernel_execute_custom(
      device->cuda_kernels[CUDA_KERNEL_TYPE_ADAPTIVE_SAMPLING_COMPUTE_TASKS_PER_BLOCK], THREADS_PER_BLOCK, 1, 1, num_blocks, 1, 1, &args,
      device->stream_main));
  }

  for (uint32_t mip_id = 0; mip_id < sampler->num_prefix_mips - 1; mip_id++) {
    KernelArgsAdaptiveSamplingComputeBlockSum args;
    args.thread_prefix_sum = DEVICE_PTR(sampler->stage_prefix_mips[mip_id]);
    args.warp_prefix_sum   = DEVICE_PTR(sampler->stage_prefix_mips[mip_id + 1]);
    args.thread_count      = (num_adaptive_sampling_blocks + (1u << WARP_SIZE_LOG * mip_id) - 1) >> (WARP_SIZE_LOG * mip_id);
    args.warp_count        = (num_adaptive_sampling_blocks + (1u << WARP_SIZE_LOG * (mip_id + 1)) - 1) >> (WARP_SIZE_LOG * (mip_id + 1));

    const uint32_t num_blocks = (args.thread_count + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    __FAILURE_HANDLE(kernel_execute_custom(
      device->cuda_kernels[CUDA_KERNEL_TYPE_ADAPTIVE_SAMPLING_COMPUTE_BLOCK_SUM], THREADS_PER_BLOCK, 1, 1, num_blocks, 1, 1, &args,
      device->stream_main));
  }

  for (uint32_t mip_id = sampler->num_prefix_mips - 1; mip_id > 0; mip_id--) {
    KernelArgsAdaptiveSamplingComputePrefixSum args;
    args.thread_prefix_sum = DEVICE_PTR(sampler->stage_prefix_mips[mip_id - 1]);
    args.warp_prefix_sum   = DEVICE_PTR(sampler->stage_prefix_mips[mip_id]);
    args.thread_count      = (num_adaptive_sampling_blocks + (1u << WARP_SIZE_LOG * (mip_id - 1)) - 1) >> (WARP_SIZE_LOG * (mip_id - 1));
    args.warp_count        = (num_adaptive_sampling_blocks + (1u << WARP_SIZE_LOG * mip_id) - 1) >> (WARP_SIZE_LOG * mip_id);

    const uint32_t num_blocks = (args.thread_count + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    __FAILURE_HANDLE(kernel_execute_custom(
      device->cuda_kernels[CUDA_KERNEL_TYPE_ADAPTIVE_SAMPLING_COMPUTE_PREFIX_SUM], THREADS_PER_BLOCK, 1, 1, num_blocks, 1, 1, &args,
      device->stream_main));
  }

  {
    uint32_t tasks_per_tile;
    __FAILURE_HANDLE(device_get_allocated_task_count(device, &tasks_per_tile));

    KernelArgsAdaptiveSamplingComputeTileBlockRanges args;
    args.block_count      = num_adaptive_sampling_blocks;
    args.block_prefix_sum = DEVICE_PTR(sampler->stage_prefix_mips[0]);
    args.dst              = DEVICE_PTR(sampler->subtile_last_blocks);
    args.tasks_per_tile   = tasks_per_tile;
    args.tile_count       = tile_count;

    const uint32_t num_blocks = (subtile_count + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    __FAILURE_HANDLE(kernel_execute_custom(
      device->cuda_kernels[CUDA_KERNEL_TYPE_ADAPTIVE_SAMPLING_COMPUTE_TILE_BLOCK_RANGES], THREADS_PER_BLOCK, 1, 1, num_blocks, 1, 1, &args,
      device->stream_main));
  }

  sampler->allocated_stage_id = stage_id;

  return LUMINARY_SUCCESS;
}

LuminaryResult device_adaptive_sampler_destroy(DeviceAdaptiveSampler** sampler) {
  __CHECK_NULL_ARGUMENT(sampler);

  __FAILURE_HANDLE(_device_adaptive_sampler_reset(*sampler));

  __FAILURE_HANDLE(host_free(sampler));

  return LUMINARY_SUCCESS;
}

LuminaryResult device_sample_allocation_step_next(DeviceSampleAllocation* allocation) {
  __CHECK_NULL_ARGUMENT(allocation);

  if (allocation->num_samples) {
    allocation->global_sample_id++;
    allocation->stage_sample_offsets[allocation->stage_id]++;

    allocation->num_samples--;
  }

  return LUMINARY_SUCCESS;
}

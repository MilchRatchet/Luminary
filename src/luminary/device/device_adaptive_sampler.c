#include "device_adaptive_sampler.h"

#include "device.h"
#include "internal_error.h"
#include "kernel.h"
#include "kernel_args.h"

LuminaryResult device_adaptive_sampler_create(DeviceAdaptiveSampler** sampler) {
  __CHECK_NULL_ARGUMENT(sampler);

  __FAILURE_HANDLE(host_malloc(sampler, sizeof(DeviceAdaptiveSampler)));
  memset(*sampler, 0, sizeof(DeviceAdaptiveSampler));

  return LUMINARY_SUCCESS;
}

LuminaryResult device_adaptive_sampler_get_buffer_sizes(DeviceAdaptiveSampler* sampler, DeviceAdaptiveSamplerBufferSizes* sizes) {
  __CHECK_NULL_ARGUMENT(sampler);
  __CHECK_NULL_ARGUMENT(sizes);

  sizes->stage_sample_counts_size     = sizeof(uint32_t) * sampler->width * sampler->height;
  sizes->stage_total_task_counts_size = sizeof(uint32_t) * (ADAPTIVE_SAMPLER_NUM_STAGES + 1);

  return LUMINARY_SUCCESS;
}

LuminaryResult device_adaptive_sampler_start_sampling(DeviceAdaptiveSampler* sampler, uint32_t width, uint32_t height) {
  __CHECK_NULL_ARGUMENT(sampler);

  sampler->width  = (width + ((1 << ADAPTIVE_SAMPLING_BLOCK_SIZE_LOG) - 1)) >> ADAPTIVE_SAMPLING_BLOCK_SIZE_LOG;
  sampler->height = (height + ((1 << ADAPTIVE_SAMPLING_BLOCK_SIZE_LOG) - 1)) >> ADAPTIVE_SAMPLING_BLOCK_SIZE_LOG;

  memset(&sampler->allocator, 0, sizeof(DeviceSampleAllocation));

  sampler->allocator.upper_bound_tasks_per_sample = (sampler->width * sampler->height) << (2 * ADAPTIVE_SAMPLING_BLOCK_SIZE_LOG);
  sampler->queued_stage_build                     = ADAPTIVE_SAMPLING_STAGE_INVALID;

  return LUMINARY_SUCCESS;
}

LuminaryResult device_adaptive_sampler_allocate_sample(
  DeviceAdaptiveSampler* sampler, DeviceSampleAllocation* allocation, uint32_t num_samples) {
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

LuminaryResult device_adaptive_sampler_get_task_count_upper_bound(DeviceAdaptiveSampler* sampler, uint32_t* task_count, uint8_t stage_id) {
  __CHECK_NULL_ARGUMENT(sampler);
  __CHECK_NULL_ARGUMENT(task_count);

  __DEBUG_ASSERT(stage_id <= ADAPTIVE_SAMPLER_NUM_STAGES + 1);

  *task_count = sampler->stage_total_task_counts[stage_id];

  return LUMINARY_SUCCESS;
}

static LuminaryResult _device_adaptive_sampler_finalize_build(DeviceAdaptiveSampler* sampler, Device* device) {
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

LuminaryResult device_adaptive_sampler_compute_next_stage(DeviceAdaptiveSampler* sampler, Device* device, uint8_t stage_id) {
  __CHECK_NULL_ARGUMENT(sampler);
  __CHECK_NULL_ARGUMENT(device);

  __DEBUG_ASSERT(device->is_main_device);

  if (sampler->allocator.stage_id == stage_id)
    return LUMINARY_SUCCESS;

  if (sampler->queued_stage_build == stage_id) {
    __FAILURE_HANDLE(_device_adaptive_sampler_finalize_build(sampler, device));
    return LUMINARY_SUCCESS;
  }

  __DEBUG_ASSERT(sampler->queued_stage_build == ADAPTIVE_SAMPLING_STAGE_INVALID);

  sampler->queued_stage_build = stage_id;

  DeviceAdaptiveSamplerBufferSizes buffer_sizes;
  __FAILURE_HANDLE(device_adaptive_sampler_get_buffer_sizes(sampler, &buffer_sizes));

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
    sampler->stage_sample_counts, device->buffers.stage_sample_counts, 0, buffer_sizes.stage_sample_counts_size, device->stream_main));
  __FAILURE_HANDLE(device_download(
    sampler->stage_total_task_counts, device->buffers.stage_total_task_counts, 0, buffer_sizes.stage_total_task_counts_size,
    device->stream_main));

  CUDA_FAILURE_HANDLE(cuEventRecord(sampler->stage_build_event, device->stream_main));

  return LUMINARY_SUCCESS;
}

LuminaryResult device_adaptive_sampler_unload(DeviceAdaptiveSampler* sampler) {
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

LuminaryResult device_adaptive_sampler_destroy(DeviceAdaptiveSampler** sampler) {
  __CHECK_NULL_ARGUMENT(sampler);

  __DEBUG_ASSERT((*sampler)->stage_sample_counts == (uint32_t*) 0);
  __DEBUG_ASSERT((*sampler)->stage_total_task_counts == (uint32_t*) 0);

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

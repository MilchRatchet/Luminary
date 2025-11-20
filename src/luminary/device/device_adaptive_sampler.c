#include "device_adaptive_sampler.h"

#include "internal_error.h"

LuminaryResult device_adaptive_sampler_create(DeviceAdaptiveSampler** sampler) {
  __CHECK_NULL_ARGUMENT(sampler);

  __FAILURE_HANDLE(host_malloc(sampler, sizeof(DeviceAdaptiveSampler)));
  memset(*sampler, 0, sizeof(DeviceAdaptiveSampler));

  return LUMINARY_SUCCESS;
}

LuminaryResult device_adaptive_sampler_get_buffer_sizes(DeviceAdaptiveSampler* sampler, DeviceAdaptiveSamplerBufferSizes* sizes) {
  __CHECK_NULL_ARGUMENT(sampler);
  __CHECK_NULL_ARGUMENT(sizes);

  sizes->stage_sample_counts_size = sizeof(uint32_t) * sampler->width * sampler->height;

  return LUMINARY_SUCCESS;
}

LuminaryResult device_adaptive_sampler_start_sampling(DeviceAdaptiveSampler* sampler, uint32_t width, uint32_t height) {
  __CHECK_NULL_ARGUMENT(sampler);

  const uint32_t new_width  = (width + ((1 << ADAPTIVE_SAMPLING_BLOCK_SIZE_LOG) - 1)) >> ADAPTIVE_SAMPLING_BLOCK_SIZE_LOG;
  const uint32_t new_height = (height + ((1 << ADAPTIVE_SAMPLING_BLOCK_SIZE_LOG) - 1)) >> ADAPTIVE_SAMPLING_BLOCK_SIZE_LOG;

  if (new_width != sampler->width || new_height != sampler->height) {
    sampler->width  = new_width;
    sampler->height = new_height;

    DeviceAdaptiveSamplerBufferSizes buffer_sizes;
    __FAILURE_HANDLE(device_adaptive_sampler_get_buffer_sizes(sampler, &buffer_sizes));

    if (sampler->stage_sample_counts)
      __FAILURE_HANDLE(host_free(&sampler->stage_sample_counts));

    // We don't need these to be zeroed because they get overwritten whenever any stage becomes available.
    __FAILURE_HANDLE(host_malloc(&sampler->stage_sample_counts, buffer_sizes.stage_sample_counts_size));
  }

  memset(&sampler->allocator, 0, sizeof(DeviceSampleAllocation));
  memset(sampler->stage_total_task_counts, 0, sizeof(uint32_t) * (ADAPTIVE_SAMPLER_NUM_STAGES + 1));

  sampler->allocator.upper_bound_paths_per_sample = (sampler->width * sampler->height) << (2 * ADAPTIVE_SAMPLING_BLOCK_SIZE_LOG);

  return LUMINARY_SUCCESS;
}

LuminaryResult device_adaptive_sampler_allocate_sample(
  DeviceAdaptiveSampler* sampler, DeviceSampleAllocation* allocation, uint32_t num_samples) {
  __CHECK_NULL_ARGUMENT(sampler);
  __CHECK_NULL_ARGUMENT(allocation);

  __DEBUG_ASSERT(num_samples < 256);

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

LuminaryResult device_adaptive_sampler_destroy(DeviceAdaptiveSampler** sampler) {
  __CHECK_NULL_ARGUMENT(sampler);

  if ((*sampler)->stage_sample_counts)
    __FAILURE_HANDLE(host_free(&(*sampler)->stage_sample_counts));

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

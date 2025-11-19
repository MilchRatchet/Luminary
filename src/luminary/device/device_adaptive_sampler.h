#ifndef LUMINARY_DEVICE_ADAPTIVE_SAMPLER_H
#define LUMINARY_DEVICE_ADAPTIVE_SAMPLER_H

#include "device_utils.h"

struct DeviceAdaptiveSamplerBufferSizes {
  size_t stage_sample_counts_size;
} typedef DeviceAdaptiveSamplerBufferSizes;

struct DeviceAdaptiveSampler {
  uint32_t width;
  uint32_t height;
  DeviceSampleAllocation allocator;
  uint32_t* stage_sample_counts;
  uint32_t stage_total_task_counts[1 + ADAPTIVE_SAMPLER_NUM_STAGES];
} typedef DeviceAdaptiveSampler;

LuminaryResult device_adaptive_sampler_create(DeviceAdaptiveSampler** sampler);
LuminaryResult device_adaptive_sampler_get_buffer_sizes(DeviceAdaptiveSampler* sampler, DeviceAdaptiveSamplerBufferSizes* sizes);
LuminaryResult device_adaptive_sampler_start_sampling(DeviceAdaptiveSampler* sampler, uint32_t width, uint32_t height);
LuminaryResult device_adaptive_sampler_allocate_sample(
  DeviceAdaptiveSampler* sampler, DeviceSampleAllocation* allocation, uint32_t num_samples);
LuminaryResult device_adaptive_sampler_get_task_count_upper_bound(DeviceAdaptiveSampler* sampler, uint32_t* task_count, uint8_t stage_id);
LuminaryResult device_adaptive_sampler_destroy(DeviceAdaptiveSampler** sampler);

LuminaryResult device_sample_allocation_step_next(DeviceSampleAllocation* allocation);

#endif /* LUMINARY_DEVICE_ADAPTIVE_SAMPLER_H */

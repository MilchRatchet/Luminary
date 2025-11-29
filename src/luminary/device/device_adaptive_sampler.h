#ifndef LUMINARY_DEVICE_ADAPTIVE_SAMPLER_H
#define LUMINARY_DEVICE_ADAPTIVE_SAMPLER_H

#include "device_memory.h"
#include "device_utils.h"

struct Device typedef Device;

struct DeviceAdaptiveSamplerBufferSizes {
  size_t stage_sample_counts_size;
  size_t stage_total_task_counts_size;
} typedef DeviceAdaptiveSamplerBufferSizes;

struct DeviceAdaptiveSampler {
  uint32_t width;
  uint32_t height;
  size_t allocated_stage_sample_counts_size;
  uint8_t queued_stage_build;
  DeviceSampleAllocation allocator;
  STAGING uint32_t* stage_sample_counts;
  STAGING uint32_t* stage_total_task_counts;
  CUevent stage_build_event;
} typedef DeviceAdaptiveSampler;

LuminaryResult device_adaptive_sampler_create(DeviceAdaptiveSampler** sampler);
LuminaryResult device_adaptive_sampler_get_buffer_sizes(DeviceAdaptiveSampler* sampler, DeviceAdaptiveSamplerBufferSizes* sizes);
LuminaryResult device_adaptive_sampler_start_sampling(DeviceAdaptiveSampler* sampler, uint32_t width, uint32_t height);
LuminaryResult device_adaptive_sampler_allocate_sample(
  DeviceAdaptiveSampler* sampler, DeviceSampleAllocation* allocation, uint32_t num_samples);
LuminaryResult device_adaptive_sampler_get_task_count_upper_bound(DeviceAdaptiveSampler* sampler, uint32_t* task_count, uint8_t stage_id);
DEVICE_CTX_FUNC LuminaryResult device_adaptive_sampler_compute_next_stage(DeviceAdaptiveSampler* sampler, Device* device, uint8_t stage_id);
DEVICE_CTX_FUNC LuminaryResult device_adaptive_sampler_unload(DeviceAdaptiveSampler* sampler);
LuminaryResult device_adaptive_sampler_destroy(DeviceAdaptiveSampler** sampler);

LuminaryResult device_sample_allocation_step_next(DeviceSampleAllocation* allocation);

#endif /* LUMINARY_DEVICE_ADAPTIVE_SAMPLER_H */

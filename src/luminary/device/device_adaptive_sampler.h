#ifndef LUMINARY_DEVICE_ADAPTIVE_SAMPLER_H
#define LUMINARY_DEVICE_ADAPTIVE_SAMPLER_H

#include "device_memory.h"
#include "device_utils.h"

struct Device typedef Device;

struct DeviceAdaptiveSamplerBufferSizes {
  size_t stage_sample_counts_size;
  size_t stage_total_task_counts_size;
} typedef DeviceAdaptiveSamplerBufferSizes;

struct AdaptiveSampler {
  uint32_t width;
  uint32_t height;
  size_t allocated_stage_sample_counts_size;
  uint8_t queued_stage_build;
  DeviceSampleAllocation allocator;
  STAGING uint32_t* stage_sample_counts;
  STAGING uint32_t* stage_total_task_counts;
  CUevent stage_build_event;
} typedef AdaptiveSampler;

LuminaryResult adaptive_sampler_create(AdaptiveSampler** sampler);
LuminaryResult adaptive_sampler_get_buffer_sizes(AdaptiveSampler* sampler, DeviceAdaptiveSamplerBufferSizes* sizes);
LuminaryResult adaptive_sampler_start_sampling(AdaptiveSampler* sampler, uint32_t width, uint32_t height);
LuminaryResult adaptive_sampler_allocate_sample(AdaptiveSampler* sampler, DeviceSampleAllocation* allocation, uint32_t num_samples);
LuminaryResult adaptive_sampler_get_task_count_upper_bound(AdaptiveSampler* sampler, uint32_t* task_count, uint8_t stage_id);
DEVICE_CTX_FUNC LuminaryResult adaptive_sampler_compute_next_stage(AdaptiveSampler* sampler, Device* device, uint8_t stage_id);
DEVICE_CTX_FUNC LuminaryResult adaptive_sampler_unload(AdaptiveSampler* sampler);
LuminaryResult adaptive_sampler_destroy(AdaptiveSampler** sampler);

struct DeviceAdaptiveSamplerDeviceBufferPtrs {
  // TODO: Turn these into CUdeviceptr. I don't want to pass the buffers. To enable this we will need typeof() support which comes with C23.
  // MSVC supports it but clang-cl only added std:clatest recently and they don't seem to support typeof() yet.
  DEVICE uint32_t* stage_sample_counts;
  DEVICE uint32_t* stage_total_task_counts;
  DEVICE uint32_t* adaptive_sampling_block_task_offsets;
  DEVICE uint32_t* tile_last_adaptive_sampling_block_index;
} typedef DeviceAdaptiveSamplerDeviceBufferPtrs;

struct DeviceAdaptiveSampler {
  uint32_t width;
  uint32_t height;
  uint8_t allocated_stage_id;
  DEVICE uint32_t* stage_sample_counts;
  DEVICE uint32_t* stage_total_task_counts;
  uint32_t num_prefix_mips;
  DEVICE uint32_t** stage_prefix_mips;
  uint32_t num_allocated_subtiles;
  DEVICE uint32_t* subtile_last_blocks;
} typedef DeviceAdaptiveSampler;

LuminaryResult device_adaptive_sampler_create(DeviceAdaptiveSampler** sampler);
LuminaryResult device_adaptive_sampler_reset(DeviceAdaptiveSampler* sampler);
DEVICE_CTX_FUNC LuminaryResult device_adaptive_sampler_update(
  DeviceAdaptiveSampler* sampler, Device* device, AdaptiveSampler* shared_sampler, bool* buffers_have_changed);
DEVICE_CTX_FUNC LuminaryResult
  device_adaptive_sampler_get_device_buffer_ptrs(DeviceAdaptiveSampler* sampler, DeviceAdaptiveSamplerDeviceBufferPtrs* ptrs);
DEVICE_CTX_FUNC LuminaryResult
  device_adaptive_sampler_ensure_stage(DeviceAdaptiveSampler* sampler, Device* device, bool* buffers_have_changed);
DEVICE_CTX_FUNC LuminaryResult device_adaptive_sampler_destroy(DeviceAdaptiveSampler** sampler);

LuminaryResult device_sample_allocation_step_next(DeviceSampleAllocation* allocation);

#endif /* LUMINARY_DEVICE_ADAPTIVE_SAMPLER_H */

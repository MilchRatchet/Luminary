#ifndef LUMINARY_DEVICE_ADAPTIVE_SAMPLER_H
#define LUMINARY_DEVICE_ADAPTIVE_SAMPLER_H

#include "device_memory.h"
#include "device_utils.h"

struct Device typedef Device;
struct DeviceRenderer typedef DeviceRenderer;

struct DeviceAdaptiveSamplerBufferSizes {
  size_t stage_sample_counts_size;
  size_t stage_total_task_counts_size;
  size_t variance_buffer_size;
} typedef DeviceAdaptiveSamplerBufferSizes;

struct AdaptiveSamplerSetupInfo {
  bool enabled;
  uint32_t width;
  uint32_t height;
  uint32_t max_sampling_rate;
  bool exposure_aware;
  float exposure;
} typedef AdaptiveSamplerSetupInfo;

struct AdaptiveSampler {
  uint32_t width;
  uint32_t height;
  size_t allocated_stage_sample_counts_size;
  size_t allocated_variance_buffer_size;
  uint8_t queued_stage_build;
  DeviceSampleAllocation allocator;
  STAGING uint32_t* stage_sample_counts;
  STAGING uint32_t* stage_total_task_counts;
  DEVICE uint32_t* stage_total_task_counts_buffer;
  DEVICE float* variance_buffer;
  DEVICE float* global_variance_buffer;
  uint32_t max_sampling_rate;
  float exposure;
  CUevent stage_build_event;
} typedef AdaptiveSampler;

LuminaryResult adaptive_sampler_create(AdaptiveSampler** sampler);
LuminaryResult adaptive_sampler_get_buffer_sizes(AdaptiveSampler* sampler, DeviceAdaptiveSamplerBufferSizes* sizes);
LuminaryResult adaptive_sampler_setup(AdaptiveSampler* sampler, const AdaptiveSamplerSetupInfo* info);
LuminaryResult adaptive_sampler_allocate_sample(AdaptiveSampler* sampler, DeviceSampleAllocation* allocation, uint32_t num_samples);
LuminaryResult adaptive_sampler_get_task_count_upper_bound(AdaptiveSampler* sampler, uint32_t* task_count, uint8_t stage_id);
DEVICE_CTX_FUNC LuminaryResult adaptive_sampler_compute_next_stage(AdaptiveSampler* sampler, Device* device, uint8_t stage_id);
DEVICE_CTX_FUNC LuminaryResult adaptive_sampler_unload(AdaptiveSampler* sampler);
LuminaryResult adaptive_sampler_destroy(AdaptiveSampler** sampler);

struct DeviceAdaptiveSamplerDeviceBufferPtrs {
  CUdeviceptr stage_sample_counts;
  CUdeviceptr adaptive_sampling_block_task_offsets;
  CUdeviceptr adaptive_sampling_subtile_block_index;
} typedef DeviceAdaptiveSamplerDeviceBufferPtrs;

struct DeviceAdaptiveSampler {
  uint32_t width;
  uint32_t height;
  uint8_t allocated_stage_id;
  DEVICE uint32_t* stage_sample_counts;
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
DEVICE_CTX_FUNC LuminaryResult device_adaptive_sampler_ensure_stage(
  DeviceAdaptiveSampler* sampler, Device* device, DeviceRenderer* renderer, AdaptiveSampler* shared_sampler, bool* buffers_have_changed);
DEVICE_CTX_FUNC LuminaryResult device_adaptive_sampler_destroy(DeviceAdaptiveSampler** sampler);

LuminaryResult device_sample_allocation_step_next(DeviceSampleAllocation* allocation);

#endif /* LUMINARY_DEVICE_ADAPTIVE_SAMPLER_H */

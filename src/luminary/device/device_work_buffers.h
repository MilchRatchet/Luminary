#ifndef LUMINARY_DEVICE_WORK_BUFFERS_H
#define LUMINARY_DEVICE_WORK_BUFFERS_H

#include "device_utils.h"

struct Device typedef Device;

struct DeviceWorkBuffersAllocationProperties {
  uint32_t external_pixel_count;
  uint32_t internal_pixel_count;
  uint32_t gbuffer_pixel_count;
  uint32_t thread_count;
  uint32_t task_count;
} typedef DeviceWorkBuffersAllocationProperties;

struct DeviceWorkBuffersPtrs {
  CUdeviceptr task_states;
  CUdeviceptr task_direct_light;
  CUdeviceptr task_results;
  CUdeviceptr results_counts;
  CUdeviceptr trace_counts;
  CUdeviceptr task_counts;
  CUdeviceptr task_offsets;
  CUdeviceptr frame_first_moment[FRAME_CHANNEL_COUNT];
  CUdeviceptr frame_second_moment_luminance;
  CUdeviceptr frame_result[FRAME_CHANNEL_COUNT];
  CUdeviceptr frame_output[FRAME_CHANNEL_COUNT];
  CUdeviceptr frame_swap;
  CUdeviceptr gbuffer_meta;
} typedef DeviceWorkBuffersPtrs;

struct DeviceWorkBuffers {
  DEVICE DeviceTaskState* task_states;
  DEVICE DeviceTaskDirectLight* task_direct_light;
  DEVICE DeviceTaskResult* task_results;
  DEVICE uint16_t* results_counts;
  DEVICE uint16_t* trace_counts;
  DEVICE uint16_t* task_counts;
  DEVICE uint16_t* task_offsets;
  DEVICE float* frame_first_moment[FRAME_CHANNEL_COUNT];
  DEVICE float* frame_second_moment_luminance;
  DEVICE float* frame_result[FRAME_CHANNEL_COUNT];
  DEVICE float* frame_output[FRAME_CHANNEL_COUNT];
  DEVICE float* frame_swap;
  DEVICE GBufferMetaData* gbuffer_meta;

  uint32_t allocated_external_pixel_count;
  uint32_t allocated_internal_pixel_count;
  uint32_t allocated_gbuffer_pixel_count;
  uint32_t allocated_thread_count;
  uint32_t allocated_task_count;
} typedef DeviceWorkBuffers;

LuminaryResult device_work_buffers_create(DeviceWorkBuffers** buffers);
DEVICE_CTX_FUNC LuminaryResult device_work_buffers_update(
  DeviceWorkBuffers* buffers, const DeviceWorkBuffersAllocationProperties* properties, bool* buffers_have_changed);
DEVICE_CTX_FUNC LuminaryResult device_work_buffers_get_ptrs(DeviceWorkBuffers* buffers, DeviceWorkBuffersPtrs* ptrs);
DEVICE_CTX_FUNC LuminaryResult device_work_buffers_destroy(DeviceWorkBuffers** buffers);

#endif /* LUMINARY_DEVICE_WORK_BUFFERS_H */

#include "device_work_buffers.h"

#include "device.h"
#include "internal_error.h"

static LuminaryResult _device_work_buffers_alloc(DEVICE void** buffer, const size_t requested_size) {
  __CHECK_NULL_ARGUMENT(buffer);

  if (*buffer) {
    __FAILURE_HANDLE(device_free(buffer));
  }

  __FAILURE_HANDLE(device_malloc(buffer, requested_size));

  return LUMINARY_SUCCESS;
}

static LuminaryResult _device_work_buffers_get_ptr(DEVICE void* buffer, CUdeviceptr* ptr) {
  __CHECK_NULL_ARGUMENT(ptr);

  *ptr = DEVICE_CUPTR(buffer);

  return LUMINARY_SUCCESS;
}

static LuminaryResult _device_work_buffers_free(DEVICE void** buffer) {
  __CHECK_NULL_ARGUMENT(buffer);

  if (*buffer) {
    __FAILURE_HANDLE(device_free(buffer));
  }

  return LUMINARY_SUCCESS;
}

#define _BUFFER_ALLOC(__internal_macro_buffer, __internal_macro_size) \
  _device_work_buffers_alloc((void**) &(__internal_macro_buffer), (__internal_macro_size))

#define _BUFFER_GET_PTR(__internal_macro_src, __internal_macro_dst, __internal_macro_member) \
  _device_work_buffers_get_ptr(                                                              \
    (void*) (__internal_macro_src)->__internal_macro_member, (CUdeviceptr*) &(__internal_macro_dst)->__internal_macro_member)

#define _BUFFER_FREE(__internal_macro_buffer) _device_work_buffers_free((void**) &(__internal_macro_buffer))

LuminaryResult device_work_buffers_create(DeviceWorkBuffers** buffers) {
  __CHECK_NULL_ARGUMENT(buffers);

  __FAILURE_HANDLE(host_malloc(buffers, sizeof(DeviceWorkBuffers)));
  memset(*buffers, 0, sizeof(DeviceWorkBuffers));

  return LUMINARY_SUCCESS;
}

LuminaryResult device_work_buffers_update(
  DeviceWorkBuffers* buffers, const DeviceWorkBuffersAllocationProperties* properties, bool* buffers_have_changed) {
  __CHECK_NULL_ARGUMENT(buffers);
  __CHECK_NULL_ARGUMENT(properties);
  __CHECK_NULL_ARGUMENT(buffers_have_changed);

  *buffers_have_changed = false;

  if (properties->external_pixel_count != buffers->allocated_external_pixel_count) {
    for (uint32_t channel_id = 0; channel_id < FRAME_CHANNEL_COUNT; channel_id++) {
      __FAILURE_HANDLE(_BUFFER_ALLOC(buffers->frame_output[channel_id], sizeof(float) * properties->external_pixel_count));
    }

    buffers->allocated_external_pixel_count = properties->external_pixel_count;
    *buffers_have_changed                   = true;
  }

  if (properties->internal_pixel_count != buffers->allocated_internal_pixel_count) {
    for (uint32_t channel_id = 0; channel_id < FRAME_CHANNEL_COUNT; channel_id++) {
      __FAILURE_HANDLE(_BUFFER_ALLOC(buffers->frame_first_moment[channel_id], sizeof(float) * properties->internal_pixel_count));
      __FAILURE_HANDLE(_BUFFER_ALLOC(buffers->frame_second_moment[channel_id], sizeof(float) * properties->internal_pixel_count));
      __FAILURE_HANDLE(_BUFFER_ALLOC(buffers->frame_result[channel_id], sizeof(float) * properties->internal_pixel_count));
    }

    buffers->allocated_internal_pixel_count = properties->internal_pixel_count;
    *buffers_have_changed                   = true;
  }

  if (properties->gbuffer_pixel_count != buffers->allocated_gbuffer_pixel_count) {
    __FAILURE_HANDLE(_BUFFER_ALLOC(buffers->gbuffer_meta, sizeof(GBufferMetaData) * properties->gbuffer_pixel_count));

    buffers->allocated_gbuffer_pixel_count = properties->gbuffer_pixel_count;
    *buffers_have_changed                  = true;
  }

  if (properties->thread_count != buffers->allocated_thread_count) {
    __FAILURE_HANDLE(_BUFFER_ALLOC(buffers->results_counts, sizeof(uint16_t) * properties->thread_count));
    __FAILURE_HANDLE(_BUFFER_ALLOC(buffers->trace_counts, sizeof(uint16_t) * properties->thread_count));
    __FAILURE_HANDLE(_BUFFER_ALLOC(buffers->task_counts, sizeof(uint16_t) * 5 * properties->thread_count));
    __FAILURE_HANDLE(_BUFFER_ALLOC(buffers->task_offsets, sizeof(uint16_t) * 5 * properties->thread_count));

    buffers->allocated_thread_count = properties->thread_count;
    *buffers_have_changed           = true;
  }

  if (properties->task_count != buffers->allocated_task_count) {
    __FAILURE_HANDLE(_BUFFER_ALLOC(buffers->task_states, sizeof(DeviceTaskState) * properties->task_count * TASK_STATE_BUFFER_INDEX_COUNT));
    __FAILURE_HANDLE(_BUFFER_ALLOC(
      buffers->task_direct_light, sizeof(DeviceTaskDirectLight) * properties->task_count * TASK_STATE_BUFFER_INDEX_DIRECT_LIGHT_COUNT));
    __FAILURE_HANDLE(_BUFFER_ALLOC(buffers->task_results, sizeof(DeviceTaskResult) * properties->task_count));

    buffers->allocated_task_count = properties->task_count;
    *buffers_have_changed         = true;
  }

  return LUMINARY_SUCCESS;
}

LuminaryResult device_work_buffers_get_ptrs(DeviceWorkBuffers* buffers, DeviceWorkBuffersPtrs* ptrs) {
  __CHECK_NULL_ARGUMENT(buffers);
  __CHECK_NULL_ARGUMENT(ptrs);

  __FAILURE_HANDLE(_BUFFER_GET_PTR(buffers, ptrs, task_states));
  __FAILURE_HANDLE(_BUFFER_GET_PTR(buffers, ptrs, task_direct_light));
  __FAILURE_HANDLE(_BUFFER_GET_PTR(buffers, ptrs, task_results));
  __FAILURE_HANDLE(_BUFFER_GET_PTR(buffers, ptrs, results_counts));
  __FAILURE_HANDLE(_BUFFER_GET_PTR(buffers, ptrs, trace_counts));
  __FAILURE_HANDLE(_BUFFER_GET_PTR(buffers, ptrs, task_counts));
  __FAILURE_HANDLE(_BUFFER_GET_PTR(buffers, ptrs, task_offsets));
  __FAILURE_HANDLE(_BUFFER_GET_PTR(buffers, ptrs, frame_swap));
  __FAILURE_HANDLE(_BUFFER_GET_PTR(buffers, ptrs, gbuffer_meta));

  for (uint32_t channel_id = 0; channel_id < FRAME_CHANNEL_COUNT; channel_id++) {
    __FAILURE_HANDLE(_BUFFER_GET_PTR(buffers, ptrs, frame_first_moment[channel_id]));
    __FAILURE_HANDLE(_BUFFER_GET_PTR(buffers, ptrs, frame_second_moment[channel_id]));
    __FAILURE_HANDLE(_BUFFER_GET_PTR(buffers, ptrs, frame_result[channel_id]));
    __FAILURE_HANDLE(_BUFFER_GET_PTR(buffers, ptrs, frame_output[channel_id]));
  }

  return LUMINARY_SUCCESS;
}

LuminaryResult device_work_buffers_destroy(DeviceWorkBuffers** buffers) {
  __CHECK_NULL_ARGUMENT(buffers);
  __CHECK_NULL_ARGUMENT(*buffers);

  __FAILURE_HANDLE(_BUFFER_FREE((*buffers)->task_states));
  __FAILURE_HANDLE(_BUFFER_FREE((*buffers)->task_direct_light));
  __FAILURE_HANDLE(_BUFFER_FREE((*buffers)->task_results));
  __FAILURE_HANDLE(_BUFFER_FREE((*buffers)->results_counts));
  __FAILURE_HANDLE(_BUFFER_FREE((*buffers)->trace_counts));
  __FAILURE_HANDLE(_BUFFER_FREE((*buffers)->task_counts));
  __FAILURE_HANDLE(_BUFFER_FREE((*buffers)->task_offsets));
  __FAILURE_HANDLE(_BUFFER_FREE((*buffers)->frame_swap));
  __FAILURE_HANDLE(_BUFFER_FREE((*buffers)->gbuffer_meta));

  for (uint32_t channel_id = 0; channel_id < FRAME_CHANNEL_COUNT; channel_id++) {
    __FAILURE_HANDLE(_BUFFER_FREE((*buffers)->frame_first_moment[channel_id]));
    __FAILURE_HANDLE(_BUFFER_FREE((*buffers)->frame_second_moment[channel_id]));
    __FAILURE_HANDLE(_BUFFER_FREE((*buffers)->frame_result[channel_id]));
    __FAILURE_HANDLE(_BUFFER_FREE((*buffers)->frame_output[channel_id]));
  }

  __FAILURE_HANDLE(host_free(buffers));

  return LUMINARY_SUCCESS;
}

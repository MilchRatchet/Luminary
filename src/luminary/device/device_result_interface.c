#include "device_result_interface.h"

#include "device.h"
#include "internal_error.h"
#include "kernel_args.h"

#define RESULT_INTERFACE_CONSUMER_EVENT_ID_INVALID (0xFFFFFFFF)

LuminaryResult device_result_interface_create(DeviceResultInterface** interface) {
  __CHECK_NULL_ARGUMENT(interface);

  __FAILURE_HANDLE(host_malloc(interface, sizeof(DeviceResultInterface)));
  memset(*interface, 0, sizeof(DeviceResultInterface));

  __FAILURE_HANDLE(mutex_create(&(*interface)->mutex));

  __FAILURE_HANDLE(array_create(&(*interface)->queued_results, sizeof(DeviceResultMap), 16));

  for (uint32_t device_id = 0; device_id < LUMINARY_MAX_NUM_DEVICES; device_id++) {
    __FAILURE_HANDLE(array_create(&(*interface)->allocated_results[device_id], sizeof(DeviceResultEntry), 16));
  }

  __FAILURE_HANDLE(array_create(&(*interface)->allocated_events, sizeof(DeviceResultEvent), 16));

  return LUMINARY_SUCCESS;
}

static LuminaryResult _device_result_interface_entry_allocate(DeviceResultInterface* interface, DeviceResultEntry* entry) {
  __CHECK_NULL_ARGUMENT(interface);
  __CHECK_NULL_ARGUMENT(entry);

  const uint32_t pixel_count = interface->pixel_count;

  for (uint32_t channel_id = 0; channel_id < FRAME_CHANNEL_COUNT; channel_id++) {
    __FAILURE_HANDLE(device_malloc_staging(
      &entry->frame_first_moment[channel_id], pixel_count * sizeof(float),
      DEVICE_MEMORY_STAGING_FLAG_PCIE_TRANSFER_ONLY | DEVICE_MEMORY_STAGING_FLAG_SHARED));
  }

  __FAILURE_HANDLE(device_malloc_staging(
    &entry->frame_second_moment_luminance, pixel_count * sizeof(float),
    DEVICE_MEMORY_STAGING_FLAG_PCIE_TRANSFER_ONLY | DEVICE_MEMORY_STAGING_FLAG_SHARED));

  CUDA_FAILURE_HANDLE(cuEventCreate(&entry->available_event, CU_EVENT_DISABLE_TIMING));

  entry->consumer_event_id = RESULT_INTERFACE_CONSUMER_EVENT_ID_INVALID;
  entry->queued            = false;

  memset(entry->stage_sample_counts, 0, sizeof(entry->stage_sample_counts));

  return LUMINARY_SUCCESS;
}

static LuminaryResult _device_result_interface_entry_free(DeviceResultInterface* interface, DeviceResultEntry* entry) {
  __CHECK_NULL_ARGUMENT(interface)
  __CHECK_NULL_ARGUMENT(entry);

  if (entry->consumer_event_id != RESULT_INTERFACE_CONSUMER_EVENT_ID_INVALID) {
    interface->allocated_events[entry->consumer_event_id].assigned = false;
  }

  for (uint32_t channel_id = 0; channel_id < FRAME_CHANNEL_COUNT; channel_id++) {
    __FAILURE_HANDLE(device_free_staging(&entry->frame_first_moment[channel_id]));
  }

  __FAILURE_HANDLE(device_free_staging(&entry->frame_second_moment_luminance));

  CUDA_FAILURE_HANDLE(cuEventDestroy(entry->available_event));

  return LUMINARY_SUCCESS;
}

LuminaryResult device_result_interface_set_pixel_count(
  DeviceResultInterface* interface, uint32_t width, uint32_t height, bool* entries_must_be_freed) {
  __CHECK_NULL_ARGUMENT(interface);
  __CHECK_NULL_ARGUMENT(entries_must_be_freed);

  *entries_must_be_freed = false;

  const uint32_t new_pixel_count = width * height;

  if (new_pixel_count != interface->pixel_count) {
    __FAILURE_HANDLE(array_clear(interface->queued_results));

    interface->pixel_count = new_pixel_count;
    *entries_must_be_freed = true;
  }

  return LUMINARY_SUCCESS;
}

LuminaryResult device_result_interface_free_entries(DeviceResultInterface* interface, uint32_t device_id) {
  __CHECK_NULL_ARGUMENT(interface);

  uint32_t num_allocated_results;
  __FAILURE_HANDLE(array_get_num_elements(interface->allocated_results[device_id], &num_allocated_results));

  for (uint32_t result_id = 0; result_id < num_allocated_results; result_id++) {
    __FAILURE_HANDLE(_device_result_interface_entry_free(interface, interface->allocated_results[device_id] + result_id));
  }

  __FAILURE_HANDLE(array_clear(interface->allocated_results[device_id]));

  return LUMINARY_SUCCESS;
}

LuminaryResult device_result_interface_queue_result(DeviceResultInterface* interface, Device* device) {
  __CHECK_NULL_ARGUMENT(interface);
  __CHECK_NULL_ARGUMENT(device);

  __DEBUG_ASSERT(device->is_main_device == false);

  uint32_t recommended_sample_count;
  __FAILURE_HANDLE(device_get_recommended_sample_queue_counts(device, &recommended_sample_count));

  uint32_t aggregate_sample_count;
  __FAILURE_HANDLE(device_renderer_get_total_executed_samples(device->renderer, &aggregate_sample_count));

  if (aggregate_sample_count < recommended_sample_count)
    return LUMINARY_SUCCESS;

  uint32_t currently_allocated_results;
  __FAILURE_HANDLE(array_get_num_elements(interface->allocated_results[device->index], &currently_allocated_results));

  uint32_t selected_result = 0xFFFFFFFF;

  for (uint32_t result_id = 0; result_id < currently_allocated_results; result_id++) {
    if (interface->allocated_results[result_id]->queued == false) {
      selected_result = result_id;
      break;
    }
  }

  if (selected_result == 0xFFFFFFFF) {
    DeviceResultEntry entry;
    __FAILURE_HANDLE(_device_result_interface_entry_allocate(interface, &entry));

    __FAILURE_HANDLE(array_push(&interface->allocated_results[device->index], &entry));

    selected_result = currently_allocated_results;
  }

  DeviceResultEntry* entry = interface->allocated_results[device->index] + selected_result;

  if (entry->consumer_event_id != RESULT_INTERFACE_CONSUMER_EVENT_ID_INVALID) {
    __DEBUG_ASSERT(interface->allocated_events[entry->consumer_event_id].assigned);

    CUDA_FAILURE_HANDLE(
      cuStreamWaitEvent(device->stream_main, interface->allocated_events[entry->consumer_event_id].event, CU_EVENT_WAIT_DEFAULT));
  }

  const uint32_t pixel_count = interface->pixel_count;

  for (uint32_t channel_id = 0; channel_id < FRAME_CHANNEL_COUNT; channel_id++) {
    __FAILURE_HANDLE(device_download(
      entry->frame_first_moment[channel_id], device->work_buffers->frame_first_moment[channel_id], 0, pixel_count * sizeof(float),
      device->stream_main));
  }

  __FAILURE_HANDLE(device_download(
    entry->frame_second_moment_luminance, device->work_buffers->frame_second_moment_luminance, 0, pixel_count * sizeof(float),
    device->stream_main));

  CUDA_FAILURE_HANDLE(cuEventRecord(entry->available_event, device->stream_main));

  __FAILURE_HANDLE(device_renderer_externalize_samples(device->renderer, entry->stage_sample_counts));

  DeviceResultMap map;
  map.allocation_id = selected_result;
  map.device_id     = device->index;

  __FAILURE_HANDLE(array_push(&interface->queued_results, &map));

  return LUMINARY_SUCCESS;
}

static LuminaryResult _device_result_interface_update_buffer(
  DeviceResultInterface* interface, Device* device, DEVICE void* dst, STAGING void* src, uint32_t num_elements) {
  __CHECK_NULL_ARGUMENT(interface);
  __CHECK_NULL_ARGUMENT(device);
  __CHECK_NULL_ARGUMENT(dst);
  __CHECK_NULL_ARGUMENT(src);

  size_t allocated_swap_size;
  __FAILURE_HANDLE(device_memory_get_size(device->work_buffers->frame_swap, &allocated_swap_size));

  const uint32_t max_elements_per_iteration = min(device->properties.max_block_count * 128 * 4, allocated_swap_size / sizeof(float));

  uint32_t element_offset         = 0;
  uint32_t num_remaining_elements = num_elements;

  while (num_remaining_elements > 0) {
    const uint32_t num_elements_this_iteration = min(max_elements_per_iteration, num_remaining_elements);

    __FAILURE_HANDLE(device_upload(
      device->work_buffers->frame_swap, ((float*) src) + element_offset, 0, num_elements_this_iteration * sizeof(float),
      device->stream_main));

    const uint32_t num_blocks_this_iteration = (num_elements_this_iteration + (4 * THREADS_PER_BLOCK - 1)) / (4 * THREADS_PER_BLOCK);

    KernelArgsBufferAdd args;
    args.dst          = DEVICE_PTR(dst);
    args.src          = DEVICE_PTR(device->work_buffers->frame_swap);
    args.base_offset  = element_offset;
    args.num_elements = num_elements_this_iteration;

    __FAILURE_HANDLE(kernel_execute_custom(
      device->cuda_kernels[CUDA_KERNEL_TYPE_BUFFER_ADD], THREADS_PER_BLOCK, 1, 1, num_blocks_this_iteration, 1, 1, &args,
      device->stream_main));

    num_remaining_elements -= num_elements_this_iteration;
    element_offset += num_elements_this_iteration;
  }

  return LUMINARY_SUCCESS;
}

static LuminaryResult _device_result_interface_get_consumer_event(DeviceResultInterface* interface, uint32_t* consumer_event_id) {
  __CHECK_NULL_ARGUMENT(interface);
  __CHECK_NULL_ARGUMENT(consumer_event_id);

  uint32_t num_allocated_events;
  __FAILURE_HANDLE(array_get_num_elements(interface->allocated_events, &num_allocated_events));

  uint32_t selected_id = RESULT_INTERFACE_CONSUMER_EVENT_ID_INVALID;

  for (uint32_t event_id = 0; event_id < num_allocated_events; event_id++) {
    if (interface->allocated_events[event_id].assigned == false) {
      selected_id = event_id;
      break;
    }
  }

  if (selected_id == RESULT_INTERFACE_CONSUMER_EVENT_ID_INVALID) {
    DeviceResultEvent event;

    CUDA_FAILURE_HANDLE(cuEventCreate(&event.event, CU_EVENT_DISABLE_TIMING));
    event.assigned = false;

    __FAILURE_HANDLE(array_push(&interface->allocated_events, &event));

    selected_id = num_allocated_events;
  }

  interface->allocated_events[selected_id].assigned = RESULT_INTERFACE_CONSUMER_EVENT_ID_INVALID;

  *consumer_event_id = selected_id;

  return LUMINARY_SUCCESS;
}

LuminaryResult device_result_interface_gather_results(DeviceResultInterface* interface, Device* device) {
  __CHECK_NULL_ARGUMENT(interface);
  __CHECK_NULL_ARGUMENT(device);

  __DEBUG_ASSERT(device->is_main_device);

  __FAILURE_HANDLE_LOCK_CRITICAL();
  __FAILURE_HANDLE_CRITICAL(mutex_lock(interface->mutex));

  uint32_t num_queued_results;
  __FAILURE_HANDLE_CRITICAL(array_get_num_elements(interface->queued_results, &num_queued_results));

  const uint32_t pixel_count = interface->pixel_count;

  for (uint32_t result_id = 0; result_id < num_queued_results; result_id++) {
    const DeviceResultMap map = interface->queued_results[result_id];

    DeviceResultEntry* entry = interface->allocated_results[map.device_id] + map.allocation_id;

    CUDA_FAILURE_HANDLE(cuStreamWaitEvent(device->stream_main, entry->available_event, CU_EVENT_WAIT_DEFAULT));

    for (uint32_t channel_id = 0; channel_id < FRAME_CHANNEL_COUNT; channel_id++) {
      __FAILURE_HANDLE(_device_result_interface_update_buffer(
        interface, device, device->work_buffers->frame_first_moment[channel_id], entry->frame_first_moment[channel_id], pixel_count));
    }

    __FAILURE_HANDLE(_device_result_interface_update_buffer(
      interface, device, device->work_buffers->frame_second_moment_luminance, entry->frame_second_moment_luminance, pixel_count));

    if (entry->consumer_event_id == RESULT_INTERFACE_CONSUMER_EVENT_ID_INVALID) {
      __FAILURE_HANDLE(_device_result_interface_get_consumer_event(interface, &entry->consumer_event_id));
    }

    CUDA_FAILURE_HANDLE(cuEventRecord(interface->allocated_events[entry->consumer_event_id].event, device->stream_main));

    __FAILURE_HANDLE(device_renderer_register_external_samples(device->renderer, entry->stage_sample_counts));

    entry->queued = false;
  }

  __FAILURE_HANDLE_CRITICAL(array_clear(interface->queued_results));

  __FAILURE_HANDLE_UNLOCK_CRITICAL();
  __FAILURE_HANDLE(mutex_unlock(interface->mutex));

  __FAILURE_HANDLE_CHECK_CRITICAL();

  return LUMINARY_SUCCESS;
}

LuminaryResult device_result_interface_destroy(DeviceResultInterface** interface) {
  __CHECK_NULL_ARGUMENT(interface);
  __CHECK_NULL_ARGUMENT(*interface);

  __FAILURE_HANDLE(mutex_destroy(&(*interface)->mutex));

  __FAILURE_HANDLE(array_destroy(&(*interface)->queued_results));

  for (uint32_t device_id = 0; device_id < LUMINARY_MAX_NUM_DEVICES; device_id++) {
    __FAILURE_HANDLE(array_destroy(&(*interface)->allocated_results[device_id]));
  }

  __FAILURE_HANDLE(array_destroy(&(*interface)->allocated_events));

  __FAILURE_HANDLE(host_free(interface));

  return LUMINARY_SUCCESS;
}

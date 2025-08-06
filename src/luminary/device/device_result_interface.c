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

  __FAILURE_HANDLE(device_malloc_staging(&entry->direct_lighting_results, pixel_count * sizeof(RGBF), true));

  for (uint32_t bucket_id = 0; bucket_id < MAX_NUM_INDIRECT_BUCKETS; bucket_id++) {
    __FAILURE_HANDLE(device_malloc_staging(&entry->indirect_lighting_results_red[bucket_id], pixel_count * sizeof(float), true));
    __FAILURE_HANDLE(device_malloc_staging(&entry->indirect_lighting_results_green[bucket_id], pixel_count * sizeof(float), true));
    __FAILURE_HANDLE(device_malloc_staging(&entry->indirect_lighting_results_blue[bucket_id], pixel_count * sizeof(float), true));
  }

  CUDA_FAILURE_HANDLE(cuEventCreate(&entry->available_event, CU_EVENT_DISABLE_TIMING));

  entry->consumer_event_id = RESULT_INTERFACE_CONSUMER_EVENT_ID_INVALID;
  entry->queued            = false;
  entry->sample_count      = 0;

  return LUMINARY_SUCCESS;
}

static LuminaryResult _device_result_interface_entry_free(DeviceResultInterface* interface, DeviceResultEntry* entry) {
  __CHECK_NULL_ARGUMENT(interface)
  __CHECK_NULL_ARGUMENT(entry);

  if (entry->consumer_event_id != RESULT_INTERFACE_CONSUMER_EVENT_ID_INVALID) {
    interface->allocated_events[entry->consumer_event_id].assigned = false;
  }

  __FAILURE_HANDLE(device_free_staging(&entry->direct_lighting_results));

  for (uint32_t bucket_id = 0; bucket_id < MAX_NUM_INDIRECT_BUCKETS; bucket_id++) {
    __FAILURE_HANDLE(device_free_staging(&entry->indirect_lighting_results_red[bucket_id]));
    __FAILURE_HANDLE(device_free_staging(&entry->indirect_lighting_results_green[bucket_id]));
    __FAILURE_HANDLE(device_free_staging(&entry->indirect_lighting_results_blue[bucket_id]));
  }

  CUDA_FAILURE_HANDLE(cuEventDestroy(entry->available_event));

  return LUMINARY_SUCCESS;
}

LuminaryResult device_result_interface_set_pixel_count(DeviceResultInterface* interface, uint32_t width, uint32_t height) {
  __CHECK_NULL_ARGUMENT(interface);

  interface->pixel_count = width * height;

  __FAILURE_HANDLE(array_clear(interface->queued_results));

  for (uint32_t device_id = 0; device_id < LUMINARY_MAX_NUM_DEVICES; device_id++) {
    uint32_t num_allocated_results;
    __FAILURE_HANDLE(array_get_num_elements(interface->allocated_results[device_id], &num_allocated_results));

    for (uint32_t result_id = 0; result_id < num_allocated_results; result_id++) {
      __FAILURE_HANDLE(_device_result_interface_entry_free(interface, interface->allocated_results[device_id] + result_id));
    }

    __FAILURE_HANDLE(array_clear(interface->allocated_results[device_id]));
  }

  return LUMINARY_SUCCESS;
}

LuminaryResult device_result_interface_queue_result(DeviceResultInterface* interface, Device* device) {
  __CHECK_NULL_ARGUMENT(interface);
  __CHECK_NULL_ARGUMENT(device);

  __DEBUG_ASSERT(device->is_main_device == false);

  uint32_t recommended_sample_count;
  __FAILURE_HANDLE(device_get_recommended_sample_queue_counts(device, &recommended_sample_count));

  if (device->aggregate_sample_count < recommended_sample_count)
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

  __FAILURE_HANDLE(device_download(
    entry->direct_lighting_results, device->buffers.frame_direct_accumulate, 0, pixel_count * sizeof(RGBF), device->stream_main));

  for (uint32_t bucket_id = 0; bucket_id < MAX_NUM_INDIRECT_BUCKETS; bucket_id++) {
    __FAILURE_HANDLE(device_download(
      entry->indirect_lighting_results_red[bucket_id], device->buffers.frame_indirect_accumulate_red[bucket_id], 0,
      pixel_count * sizeof(float), device->stream_main));
    __FAILURE_HANDLE(device_download(
      entry->indirect_lighting_results_green[bucket_id], device->buffers.frame_indirect_accumulate_green[bucket_id], 0,
      pixel_count * sizeof(float), device->stream_main));
    __FAILURE_HANDLE(device_download(
      entry->indirect_lighting_results_blue[bucket_id], device->buffers.frame_indirect_accumulate_blue[bucket_id], 0,
      pixel_count * sizeof(float), device->stream_main));
  }

  CUDA_FAILURE_HANDLE(cuEventRecord(entry->available_event, device->stream_main));

  entry->sample_count = device->aggregate_sample_count;

  DeviceResultMap map;
  map.allocation_id = selected_result;
  map.device_id     = device->index;

  __FAILURE_HANDLE(array_push(&interface->queued_results, &map));

  device->aggregate_sample_count = 0;

  return LUMINARY_SUCCESS;
}

static LuminaryResult _device_result_interface_update_buffer(
  DeviceResultInterface* interface, Device* device, DEVICE void* dst, STAGING void* src, uint32_t num_elements) {
  __CHECK_NULL_ARGUMENT(interface);
  __CHECK_NULL_ARGUMENT(device);
  __CHECK_NULL_ARGUMENT(dst);
  __CHECK_NULL_ARGUMENT(src);

  __FAILURE_HANDLE(device_upload(device->buffers.frame_direct_buffer, src, 0, num_elements * sizeof(float), device->stream_main));

  uint32_t num_blocks  = (num_elements + (4 * THREADS_PER_BLOCK - 1)) / (4 * THREADS_PER_BLOCK);
  uint32_t base_offset = 0;

  while (num_blocks > 0) {
    uint32_t num_blocks_this_iteration = min(num_blocks, device->properties.max_block_count);

    KernelArgsBufferAdd args;
    args.dst          = DEVICE_PTR(dst);
    args.src          = DEVICE_PTR(device->buffers.frame_direct_buffer);
    args.base_offset  = base_offset;
    args.num_elements = num_elements;

    __FAILURE_HANDLE(kernel_execute_custom(
      device->cuda_kernels[CUDA_KERNEL_TYPE_BUFFER_ADD], THREADS_PER_BLOCK, 1, 1, num_blocks_this_iteration, 1, 1, &args,
      device->stream_main));

    num_blocks -= num_blocks_this_iteration;
    base_offset += num_blocks_this_iteration * 4 * THREADS_PER_BLOCK;
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

    __FAILURE_HANDLE(_device_result_interface_update_buffer(
      interface, device, device->buffers.frame_direct_accumulate, entry->direct_lighting_results, pixel_count * 3));

    for (uint32_t bucket_id = 0; bucket_id < MAX_NUM_INDIRECT_BUCKETS; bucket_id++) {
      __FAILURE_HANDLE(_device_result_interface_update_buffer(
        interface, device, device->buffers.frame_indirect_accumulate_red[bucket_id], entry->indirect_lighting_results_red[bucket_id],
        pixel_count));
      __FAILURE_HANDLE(_device_result_interface_update_buffer(
        interface, device, device->buffers.frame_indirect_accumulate_green[bucket_id], entry->indirect_lighting_results_green[bucket_id],
        pixel_count));
      __FAILURE_HANDLE(_device_result_interface_update_buffer(
        interface, device, device->buffers.frame_indirect_accumulate_blue[bucket_id], entry->indirect_lighting_results_blue[bucket_id],
        pixel_count));
    }

    if (entry->consumer_event_id == RESULT_INTERFACE_CONSUMER_EVENT_ID_INVALID) {
      __FAILURE_HANDLE(_device_result_interface_get_consumer_event(interface, &entry->consumer_event_id));
    }

    CUDA_FAILURE_HANDLE(cuEventRecord(interface->allocated_events[entry->consumer_event_id].event, device->stream_main));

    device->aggregate_sample_count += entry->sample_count;

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

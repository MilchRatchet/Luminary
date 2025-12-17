#include "device_output.h"

#include "device.h"
#include "device_renderer.h"
#include "internal_error.h"
#include "kernel_args.h"

LuminaryResult device_output_create(DeviceOutput** output) {
  __CHECK_NULL_ARGUMENT(output);

  __FAILURE_HANDLE(host_malloc(output, sizeof(DeviceOutput)));
  memset(*output, 0, sizeof(DeviceOutput));

  __FAILURE_HANDLE(array_create(&(*output)->output_requests, sizeof(DeviceOutputRequest), 16));

  CUDA_FAILURE_HANDLE(cuEventCreate(&(*output)->event_output_ready, CU_EVENT_DISABLE_TIMING));
  CUDA_FAILURE_HANDLE(cuEventCreate(&(*output)->event_output_finished, CU_EVENT_DISABLE_TIMING));

  for (uint32_t buffer_id = 0; buffer_id < DEVICE_OUTPUT_BUFFER_COUNT; buffer_id++) {
    __FAILURE_HANDLE(vault_object_create(&(*output)->buffer_objects[buffer_id]));
  }

  for (uint32_t callback_id = 0; callback_id < DEVICE_OUTPUT_CALLBACK_COUNT; callback_id++) {
    CUDA_FAILURE_HANDLE(cuEventCreate(&(*output)->event_output_callback[callback_id], CU_EVENT_DISABLE_TIMING));
  }

  // Default properties
  LuminaryOutputProperties properties;
  memset(&properties, 0, sizeof(LuminaryOutputProperties));

  properties.enabled = false;
  properties.width   = 1920;
  properties.height  = 1080;

  device_output_set_properties(*output, properties);

  (*output)->color_correction = (RGBF) {.r = 1.0f, .g = 1.0f, .b = 1.0f};
  (*output)->agx_params       = (AGXCustomParams) {.power = 1.0f, .saturation = 1.0f, .slope = 1.0f};
  (*output)->filter           = LUMINARY_FILTER_NONE;

  return LUMINARY_SUCCESS;
}

LuminaryResult device_output_get_recurring_enabled(DeviceOutput* output, bool* recurring_enabled) {
  __CHECK_NULL_ARGUMENT(output);

  *recurring_enabled = output->recurring_outputs_enabled;

  return LUMINARY_SUCCESS;
}

LuminaryResult device_output_set_output_dirty(DeviceOutput* output) {
  __CHECK_NULL_ARGUMENT(output);

  output->recurring_output_is_dirty = true;

  return LUMINARY_SUCCESS;
}

static LuminaryResult _device_output_allocate_device_buffer(DeviceOutput* output, uint32_t width, uint32_t height) {
  __CHECK_NULL_ARGUMENT(output);

  bool allocate_device_buffer = true;
  size_t required_output_size = width * height * sizeof(ARGB8);

  if (output->device_buffer) {
    size_t allocated_size;
    device_memory_get_size(output->device_buffer, &allocated_size);

    allocate_device_buffer = required_output_size > allocated_size;
  }

  if (allocate_device_buffer) {
    if (output->device_buffer) {
      __FAILURE_HANDLE(device_free(&output->device_buffer));
    }

    __FAILURE_HANDLE(device_malloc(&output->device_buffer, required_output_size));
  }

  return LUMINARY_SUCCESS;
}

LuminaryResult device_output_set_properties(DeviceOutput* output, LuminaryOutputProperties properties) {
  __CHECK_NULL_ARGUMENT(output);

  output->recurring_outputs_enabled = properties.enabled;

  uint32_t width  = (properties.enabled) ? properties.width : 0;
  uint32_t height = (properties.enabled) ? properties.height : 0;

  if ((output->width != width) || (output->height != height)) {
    output->width  = width;
    output->height = height;

    for (uint32_t buffer_id = 0; buffer_id < DEVICE_OUTPUT_BUFFER_COUNT; buffer_id++) {
      __FAILURE_HANDLE_LOCK_CRITICAL();
      __FAILURE_HANDLE(vault_object_lock(output->buffer_objects[buffer_id]));

      if (output->buffers[buffer_id]) {
        __FAILURE_HANDLE_CRITICAL(device_free_staging(&output->buffers[buffer_id]));
      }

      __FAILURE_HANDLE_CRITICAL(
        device_malloc_staging(&output->buffers[buffer_id], width * height * sizeof(ARGB8), DEVICE_MEMORY_STAGING_FLAG_NONE));

      __FAILURE_HANDLE_CRITICAL(
        vault_object_set(output->buffer_objects[buffer_id], output->buffer_allocation_count, output->buffers[buffer_id]));

      __FAILURE_HANDLE_UNLOCK_CRITICAL();
      __FAILURE_HANDLE(vault_object_unlock(output->buffer_objects[buffer_id]));

      __FAILURE_HANDLE_CHECK_CRITICAL();
    }

    output->buffer_allocation_count++;

    __FAILURE_HANDLE(_device_output_allocate_device_buffer(output, width, height));
  }

  return LUMINARY_SUCCESS;
}

LuminaryResult device_output_set_camera_params(DeviceOutput* output, const Camera* camera) {
  __CHECK_NULL_ARGUMENT(output);
  __CHECK_NULL_ARGUMENT(camera);

  output->color_correction = camera->color_correction;
  output->agx_params =
    (AGXCustomParams) {.power = camera->agx_custom_power, .saturation = camera->agx_custom_saturation, .slope = camera->agx_custom_slope};
  output->filter = camera->filter;

  return LUMINARY_SUCCESS;
}

LuminaryResult device_output_add_request(DeviceOutput* output, OutputRequestProperties props) {
  __CHECK_NULL_ARGUMENT(output);

  DeviceOutputRequest output_request;

  output_request.queued = false;
  __FAILURE_HANDLE(
    device_malloc_staging(&output_request.buffer, sizeof(ARGB8) * props.width * props.height, DEVICE_MEMORY_STAGING_FLAG_NONE));
  __FAILURE_HANDLE(vault_object_create(&output_request.buffer_object));
  output_request.props = props;

  __FAILURE_HANDLE_LOCK_CRITICAL();
  __FAILURE_HANDLE(vault_object_lock(output_request.buffer_object));

  __FAILURE_HANDLE_CRITICAL(vault_object_set(output_request.buffer_object, 0, output_request.buffer));

  __FAILURE_HANDLE_UNLOCK_CRITICAL();
  __FAILURE_HANDLE(vault_object_unlock(output_request.buffer_object));

  __FAILURE_HANDLE_CHECK_CRITICAL();

  __FAILURE_HANDLE(array_push(&output->output_requests, &output_request));

  __FAILURE_HANDLE(_device_output_allocate_device_buffer(output, props.width, props.height));

  return LUMINARY_SUCCESS;
}

LuminaryResult device_output_register_callback(DeviceOutput* output, CUhostFn callback_func, DeviceCommonCallbackData data) {
  __CHECK_NULL_ARGUMENT(output);
  __CHECK_NULL_ARGUMENT(callback_func);

  output->registered_callback_func = callback_func;

  for (uint32_t callback_index = 0; callback_index < DEVICE_OUTPUT_CALLBACK_COUNT; callback_index++) {
    output->callback_data[callback_index].common.device_manager = data.device_manager;
    output->callback_data[callback_index].common.device_index   = data.device_index;
  }

  return LUMINARY_SUCCESS;
}

static LuminaryResult _device_output_generate_output(
  DeviceOutput* output, Device* device, DeviceOutputCallbackData* callback_data, CUevent callback_event) {
  __CHECK_NULL_ARGUMENT(output);
  __CHECK_NULL_ARGUMENT(device);

  const uint32_t width  = callback_data->descriptor.meta_data.width;
  const uint32_t height = callback_data->descriptor.meta_data.height;

  // We access the buffer without locking because the handle will only be available to other threads
  // once the callback has happened which is when the download has finished.
  STAGING void* dst_buffer;
  __FAILURE_HANDLE(vault_handle_get(callback_data->descriptor.data_handle, &dst_buffer));

  KernelArgsConvertRGBFToARGB8 args;
  args.dst           = DEVICE_PTR(output->device_buffer);
  args.width         = width;
  args.height        = height;
  args.filter        = output->filter;
  args.undersampling = device->undersampling_state;

  __FAILURE_HANDLE(
    kernel_execute_with_args(device->cuda_kernels[CUDA_KERNEL_TYPE_CONVERT_RGBF_TO_ARGB8], (void*) &args, device->stream_output));

  __FAILURE_HANDLE(device_download(dst_buffer, output->device_buffer, 0, width * height * sizeof(ARGB8), device->stream_output));

  CUDA_FAILURE_HANDLE(cuEventRecord(callback_event, device->stream_output));
  CUDA_FAILURE_HANDLE(cuStreamWaitEvent(device->stream_callbacks, callback_event, CU_EVENT_WAIT_DEFAULT));
  CUDA_FAILURE_HANDLE(cuLaunchHostFunc(device->stream_callbacks, output->registered_callback_func, (void*) callback_data));

  output->callback_index = (output->callback_index + 1) % DEVICE_OUTPUT_CALLBACK_COUNT;

  return LUMINARY_SUCCESS;
}

static bool _device_output_recurring_needs_queueing(DeviceOutput* output, const uint32_t current_sample_count) {
  __CHECK_NULL_ARGUMENT(output);

  if (output->recurring_outputs_enabled == false)
    return false;

  if (output->recurring_output_is_dirty)
    return true;

  // TODO: Do we still need any advanced logic here with adaptive sampling?
  return true;
}

static bool _device_output_request_needs_queueing(DeviceOutputRequest* request, const uint32_t current_sample_count) {
  if (request->queued)
    return false;

  if ((request->props.sample_count > 0) && request->props.sample_count != current_sample_count)
    return false;

  return true;
}

LuminaryResult device_output_will_output(DeviceOutput* output, DeviceRenderer* renderer, bool* does_output) {
  __CHECK_NULL_ARGUMENT(output);
  __CHECK_NULL_ARGUMENT(renderer);
  __CHECK_NULL_ARGUMENT(does_output);

  uint32_t aggregate_sample_count;
  __FAILURE_HANDLE(device_renderer_get_total_executed_samples(renderer, &aggregate_sample_count));

  bool generate_output = false;

  generate_output |= _device_output_recurring_needs_queueing(output, aggregate_sample_count);

  uint32_t num_output_requests;
  __FAILURE_HANDLE(array_get_num_elements(output->output_requests, &num_output_requests));

  for (uint32_t output_request_id = 0; output_request_id < num_output_requests; output_request_id++) {
    DeviceOutputRequest* output_request = output->output_requests + output_request_id;

    generate_output |= _device_output_request_needs_queueing(output_request, aggregate_sample_count);
  }

  *does_output = generate_output;

  return LUMINARY_SUCCESS;
}

LuminaryResult device_output_generate_output(DeviceOutput* output, Device* device, uint32_t render_event_id) {
  __CHECK_NULL_ARGUMENT(output);
  __CHECK_NULL_ARGUMENT(device);

  __FAILURE_HANDLE(device_output_wait_for_completion(output, device->stream_main));

  // The output settings could have changed since the the last rendered sample, make sure we use the current settings.
  __FAILURE_HANDLE(device_sync_constant_memory(device));

  uint32_t aggregate_sample_count;
  __FAILURE_HANDLE(device_renderer_get_total_executed_samples(device->renderer, &aggregate_sample_count));

  KernelArgsGenerateFinalImage generate_final_image_args;

  generate_final_image_args.color_correction = output->color_correction;
  generate_final_image_args.agx_params       = output->agx_params;
  generate_final_image_args.undersampling    = device->undersampling_state;

  __FAILURE_HANDLE(kernel_execute_with_args(
    device->cuda_kernels[CUDA_KERNEL_TYPE_GENERATE_FINAL_IMAGE], (void*) &generate_final_image_args, device->stream_main));

  CUDA_FAILURE_HANDLE(cuEventRecord(output->event_output_ready, device->stream_main));
  CUDA_FAILURE_HANDLE(cuStreamWaitEvent(device->stream_output, output->event_output_ready, CU_EVENT_WAIT_DEFAULT));

  if (_device_output_recurring_needs_queueing(output, aggregate_sample_count) == true) {
    __DEBUG_ASSERT(output->width > 0 && output->height > 0);

    output->recurring_output_is_dirty = false;

    DeviceOutputCallbackData* data = output->callback_data + output->callback_index;

    data->render_event_id                      = render_event_id;
    data->descriptor.is_recurring_output       = true;
    data->descriptor.meta_data.width           = output->width;
    data->descriptor.meta_data.height          = output->height;
    data->descriptor.meta_data.sample_count    = aggregate_sample_count;
    data->descriptor.meta_data.is_first_output = (device->undersampling_state & UNDERSAMPLING_FIRST_SAMPLE_MASK) != 0;

    __FAILURE_HANDLE(vault_handle_create(&data->descriptor.data_handle, output->buffer_objects[output->buffer_index]));

    CUevent event_output_callback = output->event_output_callback[output->callback_index];

    __FAILURE_HANDLE(_device_output_generate_output(output, device, data, event_output_callback));

    output->buffer_index   = (output->buffer_index + 1) % DEVICE_OUTPUT_BUFFER_COUNT;
    output->callback_index = (output->callback_index + 1) % DEVICE_OUTPUT_CALLBACK_COUNT;
  }

  uint32_t num_output_requests;
  __FAILURE_HANDLE(array_get_num_elements(output->output_requests, &num_output_requests));

  for (uint32_t output_request_id = 0; output_request_id < num_output_requests; output_request_id++) {
    DeviceOutputRequest* output_request = output->output_requests + output_request_id;

    // TODO: If the output request has already been processed, mark it for deletion, else we leak precious staging memory.

    if (_device_output_request_needs_queueing(output_request, aggregate_sample_count) == false)
      continue;

    DeviceOutputCallbackData* data = output->callback_data + output->callback_index;

    data->render_event_id                      = render_event_id;
    data->descriptor.is_recurring_output       = false;
    data->descriptor.meta_data.width           = output_request->props.width;
    data->descriptor.meta_data.height          = output_request->props.height;
    data->descriptor.meta_data.sample_count    = aggregate_sample_count;
    data->descriptor.meta_data.is_first_output = (device->undersampling_state & UNDERSAMPLING_FIRST_SAMPLE_MASK) != 0;

    __FAILURE_HANDLE(vault_handle_create(&data->descriptor.data_handle, output_request->buffer_object));

    CUevent event_output_callback = output->event_output_callback[output->callback_index];

    __FAILURE_HANDLE(_device_output_generate_output(output, device, data, event_output_callback));

    output->callback_index = (output->callback_index + 1) % DEVICE_OUTPUT_CALLBACK_COUNT;

    output_request->queued = true;
  }

  CUDA_FAILURE_HANDLE(cuEventRecord(output->event_output_finished, device->stream_output));

  return LUMINARY_SUCCESS;
}

LuminaryResult device_output_wait_for_completion(DeviceOutput* output, CUstream stream) {
  __CHECK_NULL_ARGUMENT(output);

  CUDA_FAILURE_HANDLE(cuStreamWaitEvent(stream, output->event_output_finished, CU_EVENT_WAIT_DEFAULT));

  return LUMINARY_SUCCESS;
}

LuminaryResult device_output_destroy(DeviceOutput** output) {
  __CHECK_NULL_ARGUMENT(output);

  for (uint32_t buffer_id = 0; buffer_id < DEVICE_OUTPUT_BUFFER_COUNT; buffer_id++) {
    __FAILURE_HANDLE(vault_object_reset((*output)->buffer_objects[buffer_id]));

    if ((*output)->buffers[buffer_id]) {
      __FAILURE_HANDLE(device_free_staging(&(*output)->buffers[buffer_id]));
    }

    __FAILURE_HANDLE(vault_object_destroy(&(*output)->buffer_objects[buffer_id]));
  }

  if ((*output)->device_buffer) {
    __FAILURE_HANDLE(device_free(&(*output)->device_buffer));
  }

  uint32_t num_output_requests;
  __FAILURE_HANDLE(array_get_num_elements((*output)->output_requests, &num_output_requests));

  for (uint32_t output_request_id = 0; output_request_id < num_output_requests; output_request_id++) {
    __FAILURE_HANDLE(vault_object_reset((*output)->output_requests[output_request_id].buffer_object));

    if ((*output)->output_requests[output_request_id].buffer) {
      __FAILURE_HANDLE(device_free_staging(&(*output)->output_requests[output_request_id].buffer));
    }

    __FAILURE_HANDLE(vault_object_destroy(&(*output)->output_requests[output_request_id].buffer_object));
  }

  for (uint32_t callback_id = 0; callback_id < DEVICE_OUTPUT_CALLBACK_COUNT; callback_id++) {
    CUDA_FAILURE_HANDLE(cuEventDestroy((*output)->event_output_callback[callback_id]));
  }

  CUDA_FAILURE_HANDLE(cuEventDestroy((*output)->event_output_ready));
  CUDA_FAILURE_HANDLE(cuEventDestroy((*output)->event_output_finished));

  __FAILURE_HANDLE(array_destroy(&(*output)->output_requests));

  __FAILURE_HANDLE(host_free(output));

  return LUMINARY_SUCCESS;
}

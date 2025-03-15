#include "device_output.h"

#include "device.h"
#include "internal_error.h"
#include "kernel_args.h"

LuminaryResult device_output_create(DeviceOutput** output) {
  __CHECK_NULL_ARGUMENT(output);

  __FAILURE_HANDLE(host_malloc(output, sizeof(DeviceOutput)));
  memset(*output, 0, sizeof(DeviceOutput));

  __FAILURE_HANDLE(array_create(&(*output)->output_requests, sizeof(DeviceOutputRequest), 16));

  CUDA_FAILURE_HANDLE(cuEventCreate(&(*output)->event_output_ready, CU_EVENT_DISABLE_TIMING));
  CUDA_FAILURE_HANDLE(cuEventCreate(&(*output)->event_output_finished, CU_EVENT_DISABLE_TIMING));

  // Default size
  device_output_set_size(*output, 1920, 1080);

  (*output)->color_correction = (RGBF) {.r = 1.0f, .g = 1.0f, .b = 1.0f};
  (*output)->agx_params       = (AGXCustomParams) {.power = 1.0f, .saturation = 1.0f, .slope = 1.0f};
  (*output)->filter           = LUMINARY_FILTER_NONE;

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

LuminaryResult device_output_set_size(DeviceOutput* output, uint32_t width, uint32_t height) {
  __CHECK_NULL_ARGUMENT(output);

  if ((output->width != width) || (output->height != height)) {
    output->width  = width;
    output->height = height;

    for (uint32_t buffer_id = 0; buffer_id < DEVICE_OUTPUT_BUFFER_COUNT; buffer_id++) {
      if (output->buffers[buffer_id]) {
        __FAILURE_HANDLE(device_free_staging(&output->buffers[buffer_id]));
      }

      __FAILURE_HANDLE(device_malloc_staging(&output->buffers[buffer_id], width * height * sizeof(ARGB8), false));
    }

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
  __FAILURE_HANDLE(device_malloc_staging(&output_request.buffer, sizeof(ARGB8) * props.width * props.height, false));
  output_request.props = props;

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

static LuminaryResult _device_output_generate_output(DeviceOutput* output, Device* device, DeviceOutputCallbackData* callback_data) {
  __CHECK_NULL_ARGUMENT(output);
  __CHECK_NULL_ARGUMENT(device);

  const uint32_t width  = callback_data->descriptor.meta_data.width;
  const uint32_t height = callback_data->descriptor.meta_data.height;

  STAGING void* dst_buffer = callback_data->descriptor.data;

  KernelArgsConvertRGBFToARGB8 args;
  args.dst    = DEVICE_PTR(output->device_buffer);
  args.width  = width;
  args.height = height;
  args.filter = output->filter;

  __FAILURE_HANDLE(
    kernel_execute_with_args(device->cuda_kernels[CUDA_KERNEL_TYPE_CONVERT_RGBF_TO_ARGB8], (void*) &args, device->stream_output));

  __FAILURE_HANDLE(device_download(dst_buffer, output->device_buffer, 0, width * height * sizeof(ARGB8), device->stream_output));

  CUDA_FAILURE_HANDLE(cuEventRecord(device->event_queue_output, device->stream_output));
  CUDA_FAILURE_HANDLE(cuStreamWaitEvent(device->stream_callbacks, device->event_queue_output, CU_EVENT_WAIT_DEFAULT));
  CUDA_FAILURE_HANDLE(cuLaunchHostFunc(device->stream_callbacks, output->registered_callback_func, (void*) callback_data));

  output->callback_index = (output->callback_index + 1) % DEVICE_OUTPUT_CALLBACK_COUNT;

  return LUMINARY_SUCCESS;
}

LuminaryResult device_output_generate_output(DeviceOutput* output, Device* device, uint32_t render_event_id) {
  __CHECK_NULL_ARGUMENT(output);
  __CHECK_NULL_ARGUMENT(device);

  if (output->width == 0 || output->height == 0) {
    __RETURN_ERROR(LUMINARY_ERROR_API_EXCEPTION, "Device output is set to a size of 0.");
  }

  CUDA_FAILURE_HANDLE(cuStreamWaitEvent(device->stream_main, output->event_output_finished, CU_EVENT_WAIT_DEFAULT));

  KernelArgsGenerateFinalImage generate_final_image_args;

  generate_final_image_args.src              = DEVICE_PTR(device->buffers.frame_current_result);
  generate_final_image_args.color_correction = output->color_correction;
  generate_final_image_args.agx_params       = output->agx_params;

  __FAILURE_HANDLE(kernel_execute_with_args(
    device->cuda_kernels[CUDA_KERNEL_TYPE_GENERATE_FINAL_IMAGE], (void*) &generate_final_image_args, device->stream_main));

  CUDA_FAILURE_HANDLE(cuEventRecord(output->event_output_ready, device->stream_main));
  CUDA_FAILURE_HANDLE(cuStreamWaitEvent(device->stream_output, output->event_output_ready, CU_EVENT_WAIT_DEFAULT));

  DeviceOutputCallbackData* data = output->callback_data + output->callback_index;

  data->render_event_id                      = render_event_id;
  data->descriptor.is_recurring_output       = true;
  data->descriptor.meta_data.width           = output->width;
  data->descriptor.meta_data.height          = output->height;
  data->descriptor.meta_data.sample_count    = device->sample_count.current_sample_count;
  data->descriptor.meta_data.is_first_output = (device->undersampling_state & UNDERSAMPLING_FIRST_SAMPLE_MASK) != 0;
  data->descriptor.data                      = output->buffers[output->buffer_index];

  __FAILURE_HANDLE(_device_output_generate_output(output, device, data));

  output->buffer_index   = (output->buffer_index + 1) % DEVICE_OUTPUT_BUFFER_COUNT;
  output->callback_index = (output->callback_index + 1) % DEVICE_OUTPUT_CALLBACK_COUNT;

  uint32_t num_output_requests;
  __FAILURE_HANDLE(array_get_num_elements(output->output_requests, &num_output_requests));

  for (uint32_t output_request_id = 0; output_request_id < num_output_requests; output_request_id++) {
    DeviceOutputRequest* output_request = output->output_requests + output_request_id;

    if (output_request->queued)
      continue;

    if ((output_request->props.sample_count > 0) && output_request->props.sample_count != device->sample_count.current_sample_count)
      continue;

    data = output->callback_data + output->callback_index;

    data->render_event_id                      = render_event_id;
    data->descriptor.is_recurring_output       = false;
    data->descriptor.meta_data.width           = output_request->props.width;
    data->descriptor.meta_data.height          = output_request->props.height;
    data->descriptor.meta_data.sample_count    = device->sample_count.current_sample_count;
    data->descriptor.meta_data.is_first_output = (device->undersampling_state & UNDERSAMPLING_FIRST_SAMPLE_MASK) != 0;
    data->descriptor.data                      = output_request->buffer;

    __FAILURE_HANDLE(_device_output_generate_output(output, device, data));

    output->callback_index = (output->callback_index + 1) % DEVICE_OUTPUT_CALLBACK_COUNT;

    output_request->queued = true;
  }

  CUDA_FAILURE_HANDLE(cuEventRecord(output->event_output_finished, device->stream_output));

  return LUMINARY_SUCCESS;
}

LuminaryResult device_output_destroy(DeviceOutput** output) {
  __CHECK_NULL_ARGUMENT(output);

  for (uint32_t buffer_id = 0; buffer_id < DEVICE_OUTPUT_BUFFER_COUNT; buffer_id++) {
    if ((*output)->buffers[buffer_id]) {
      __FAILURE_HANDLE(device_free_staging(&(*output)->buffers[buffer_id]));
    }
  }

  if ((*output)->device_buffer) {
    __FAILURE_HANDLE(device_free(&(*output)->device_buffer));
  }

  uint32_t num_output_requests;
  __FAILURE_HANDLE(array_get_num_elements((*output)->output_requests, &num_output_requests));

  for (uint32_t output_request_id = 0; output_request_id < num_output_requests; output_request_id++) {
    __FAILURE_HANDLE(device_free_staging(&(*output)->output_requests[output_request_id].buffer));
  }

  CUDA_FAILURE_HANDLE(cuEventDestroy((*output)->event_output_ready));
  CUDA_FAILURE_HANDLE(cuEventDestroy((*output)->event_output_finished));

  __FAILURE_HANDLE(array_destroy(&(*output)->output_requests));

  __FAILURE_HANDLE(host_free(output));

  return LUMINARY_SUCCESS;
}

#include "device_output.h"

#include "device.h"
#include "internal_error.h"
#include "kernel_args.h"

LuminaryResult device_output_create(DeviceOutput** output) {
  __CHECK_NULL_ARGUMENT(output);

  __FAILURE_HANDLE(host_malloc(output, sizeof(DeviceOutput)));
  memset(*output, 0, sizeof(DeviceOutput));

  // Default size
  device_output_set_size(*output, 1920, 1080);

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

    if (output->device_buffer) {
      __FAILURE_HANDLE(device_free(&output->device_buffer));
    }

    __FAILURE_HANDLE(device_malloc(&output->device_buffer, width * height * sizeof(ARGB8)));
  }

  return LUMINARY_SUCCESS;
}

LuminaryResult device_output_register_callback(DeviceOutput* output, CUhostFn callback_func, DeviceCommonCallbackData data) {
  __CHECK_NULL_ARGUMENT(output);
  __CHECK_NULL_ARGUMENT(callback_func);

  output->registered_callback_func = callback_func;

  for (uint32_t buffer_index = 0; buffer_index < DEVICE_OUTPUT_BUFFER_COUNT; buffer_index++) {
    output->callback_data[buffer_index].common.device_manager = data.device_manager;
    output->callback_data[buffer_index].common.device_index   = data.device_index;
  }

  return LUMINARY_SUCCESS;
}

LuminaryResult device_output_generate_output(DeviceOutput* output, Device* device) {
  __CHECK_NULL_ARGUMENT(output);
  __CHECK_NULL_ARGUMENT(device);

  if (output->width == 0 || output->height == 0) {
    __RETURN_ERROR(LUMINARY_ERROR_API_EXCEPTION, "Device output is set to a size of 0.");
  }

  KernelArgsGenerateFinalImage generate_final_image_args;

  // TODO: Use the correct buffer based on post
  generate_final_image_args.src              = DEVICE_PTR(device->buffers.frame_accumulate);
  generate_final_image_args.color_correction = (RGBF){.r = 1.0f, .g = 1.0f, .b = 1.0f};
  generate_final_image_args.agx_params       = (AGXCustomParams){.power = 1.0f, .saturation = 1.0f, .slope = 1.0f};

  __FAILURE_HANDLE(kernel_execute_with_args(
    device->cuda_kernels[CUDA_KERNEL_TYPE_GENERATE_FINAL_IMAGE], (void*) &generate_final_image_args, device->stream_main));

  KernelArgsConvertRGBFToARGB8 args;
  args.dst    = DEVICE_PTR(output->device_buffer);
  args.width  = output->width;
  args.height = output->height;
  args.filter = LUMINARY_FILTER_NONE;  // TODO: Pass the actual filter to here.

  __FAILURE_HANDLE(
    kernel_execute_with_args(device->cuda_kernels[CUDA_KERNEL_TYPE_CONVERT_RGBF_TO_ARGB8], (void*) &args, device->stream_main));

  __FAILURE_HANDLE(device_download(
    output->buffers[output->buffer_index], output->device_buffer, 0, output->width * output->height * sizeof(ARGB8), device->stream_main));

  CUDA_FAILURE_HANDLE(cuEventRecord(device->event_queue_output, device->stream_main));

  DeviceOutputCallbackData* data = output->callback_data + output->buffer_index;

  data->width           = output->width;
  data->height          = output->height;
  data->data            = output->buffers[output->buffer_index];
  data->is_first_output = (device->undersampling_state & UNDERSAMPLING_FIRST_SAMPLE_MASK);

  CUDA_FAILURE_HANDLE(cuStreamWaitEvent(device->stream_callbacks, device->event_queue_output, CU_EVENT_WAIT_DEFAULT));
  CUDA_FAILURE_HANDLE(cuLaunchHostFunc(device->stream_callbacks, output->registered_callback_func, (void*) data));

  output->buffer_index = (output->buffer_index + 1) % DEVICE_OUTPUT_BUFFER_COUNT;

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

  __FAILURE_HANDLE(host_free(output));

  return LUMINARY_SUCCESS;
}

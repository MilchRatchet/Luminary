#include "device_output.h"

#include "device.h"
#include "internal_error.h"
#include "kernel_args.h"

LuminaryResult device_output_create(DeviceOutput** output) {
  __CHECK_NULL_ARGUMENT(output);

  __FAILURE_HANDLE(host_malloc(output, sizeof(DeviceOutput)));
  memset(*output, 0, sizeof(DeviceOutput));

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

      __FAILURE_HANDLE(device_malloc_staging(&output->buffers[buffer_id], width * height * sizeof(XRGB8), false));
    }

    if (output->device_buffer) {
      __FAILURE_HANDLE(device_free(&output->device_buffer));
    }

    __FAILURE_HANDLE(device_malloc(&output->device_buffer, width * height * sizeof(XRGB8)));
  }

  return LUMINARY_SUCCESS;
}

LuminaryResult device_output_generate_output(DeviceOutput* output, Device* device) {
  __CHECK_NULL_ARGUMENT(output);
  __CHECK_NULL_ARGUMENT(device);

  if (output->width == 0 || output->height == 0) {
    __RETURN_ERROR(LUMINARY_ERROR_API_EXCEPTION, "Device output is set to a size of 0.");
  }

  __FAILURE_HANDLE(kernel_execute(device->cuda_kernels[CUDA_KERNEL_TYPE_GENERATE_FINAL_IMAGE], device->stream_main));

  KernelArgsConvertRGBFToXRGB8 args;
  args.dst    = DEVICE_PTR(output->device_buffer);
  args.width  = output->width;
  args.height = output->height;
  args.filter = LUMINARY_FILTER_NONE;  // TODO: Pass the actual filter to here.

  __FAILURE_HANDLE(
    kernel_execute_with_args(device->cuda_kernels[CUDA_KERNEL_TYPE_CONVERT_RGBF_TO_XRGB8], (void*) &args, device->stream_main));

  __FAILURE_HANDLE(device_download(
    output->buffers[output->buffer_index], output->device_buffer, 0, output->width * output->height * sizeof(XRGB8), device->stream_main));

  // TODO: Use a CUhostFn call to queue the copy of the output to the host thread (or just sync here)

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

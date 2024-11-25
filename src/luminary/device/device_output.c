#include "device_output.h"

#include "internal_error.h"

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
  }

  return LUMINARY_SUCCESS;
}

LuminaryResult device_output_destroy(DeviceOutput** output) {
  __CHECK_NULL_ARGUMENT(output);

  for (uint32_t buffer_id = 0; buffer_id < DEVICE_OUTPUT_BUFFER_COUNT; buffer_id++) {
    if ((*output)->buffers[buffer_id]) {
      __FAILURE_HANDLE(device_free_staging(&(*output)->buffers[buffer_id]));
    }
  }

  __FAILURE_HANDLE(host_free(output));

  return LUMINARY_SUCCESS;
}

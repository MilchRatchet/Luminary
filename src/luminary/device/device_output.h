#ifndef LUMINARY_DEVICE_OUTPUT_H
#define LUMINARY_DEVICE_OUTPUT_H

#include "device_callback.h"
#include "device_memory.h"
#include "device_utils.h"

struct Device typedef Device;

#define DEVICE_OUTPUT_BUFFER_COUNT 8

struct DeviceOutput {
  uint32_t width;
  uint32_t height;
  STAGING void* buffers[DEVICE_OUTPUT_BUFFER_COUNT];
  uint32_t buffer_index;
  DEVICE XRGB8* device_buffer;
  CUhostFn registered_callback_func;
  DeviceOutputCallbackData callback_data[DEVICE_OUTPUT_BUFFER_COUNT];
} typedef DeviceOutput;

DEVICE_CTX_FUNC LuminaryResult device_output_create(DeviceOutput** output);
DEVICE_CTX_FUNC LuminaryResult device_output_set_size(DeviceOutput* output, uint32_t width, uint32_t height);
DEVICE_CTX_FUNC LuminaryResult device_output_register_callback(DeviceOutput* output, CUhostFn callback_func, DeviceCommonCallbackData data);
DEVICE_CTX_FUNC LuminaryResult device_output_generate_output(DeviceOutput* output, Device* device);
DEVICE_CTX_FUNC LuminaryResult device_output_destroy(DeviceOutput** output);

#endif /* LUMINARY_DEVICE_OUTPUT_H */

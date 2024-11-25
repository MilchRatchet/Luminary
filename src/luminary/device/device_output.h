#ifndef LUMINARY_DEVICE_OUTPUT_H
#define LUMINARY_DEVICE_OUTPUT_H

#include "device_memory.h"
#include "device_utils.h"

struct Device typedef Device;

#define DEVICE_OUTPUT_BUFFER_COUNT 4

struct DeviceOutput {
  uint32_t width;
  uint32_t height;
  STAGING void* buffers[DEVICE_OUTPUT_BUFFER_COUNT];
} typedef DeviceOutput;

LuminaryResult device_output_create(DeviceOutput** output);
LuminaryResult device_output_set_size(DeviceOutput* output, uint32_t width, uint32_t height);
LuminaryResult device_output_destroy(DeviceOutput** output);

#endif /* LUMINARY_DEVICE_OUTPUT_H */

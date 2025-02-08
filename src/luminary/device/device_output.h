#ifndef LUMINARY_DEVICE_OUTPUT_H
#define LUMINARY_DEVICE_OUTPUT_H

#include "device_callback.h"
#include "device_memory.h"
#include "device_utils.h"

struct Device typedef Device;

#define DEVICE_OUTPUT_BUFFER_COUNT 128
#define DEVICE_OUTPUT_CALLBACK_COUNT 128

struct DeviceOutputRequest {
  bool queued;
  STAGING void* buffer;
  OutputRequestProperties props;
} typedef DeviceOutputRequest;

struct DeviceOutput {
  uint32_t width;
  uint32_t height;
  STAGING void* buffers[DEVICE_OUTPUT_BUFFER_COUNT];
  uint32_t buffer_index;
  DEVICE ARGB8* device_buffer;
  CUhostFn registered_callback_func;
  DeviceOutputCallbackData callback_data[DEVICE_OUTPUT_CALLBACK_COUNT];
  uint32_t callback_index;
  ARRAY DeviceOutputRequest* output_requests;
  RGBF color_correction;
  AGXCustomParams agx_params;
  LuminaryFilter filter;
} typedef DeviceOutput;

DEVICE_CTX_FUNC LuminaryResult device_output_create(DeviceOutput** output);
DEVICE_CTX_FUNC LuminaryResult device_output_set_size(DeviceOutput* output, uint32_t width, uint32_t height);
DEVICE_CTX_FUNC LuminaryResult device_output_set_camera_params(DeviceOutput* output, const Camera* camera);
DEVICE_CTX_FUNC LuminaryResult device_output_add_request(DeviceOutput* output, OutputRequestProperties props);
DEVICE_CTX_FUNC LuminaryResult device_output_register_callback(DeviceOutput* output, CUhostFn callback_func, DeviceCommonCallbackData data);
DEVICE_CTX_FUNC LuminaryResult device_output_generate_output(DeviceOutput* output, Device* device, uint32_t render_event_id);
DEVICE_CTX_FUNC LuminaryResult device_output_destroy(DeviceOutput** output);

#endif /* LUMINARY_DEVICE_OUTPUT_H */

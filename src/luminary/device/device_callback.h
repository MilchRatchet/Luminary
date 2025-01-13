#ifndef LUMINARY_DEVICE_CALLBACK_H
#define LUMINARY_DEVICE_CALLBACK_H

#include "device_utils.h"

struct DeviceManager typedef DeviceManager;

struct DeviceCommonCallbackData {
  DeviceManager* device_manager;
  uint32_t device_index;
} typedef DeviceCommonCallbackData;

struct DeviceOutputCallbackData {
  DeviceCommonCallbackData common;
  OutputDescriptor descriptor;
  uint32_t render_event_id;
} typedef DeviceOutputCallbackData;

struct DeviceRenderCallbackData {
  DeviceCommonCallbackData common;
  uint64_t render_id;
  uint32_t render_event_id;
} typedef DeviceRenderCallbackData;

#endif /* LUMINARY_DEVICE_CALLBACK_H */

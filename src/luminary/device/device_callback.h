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
  uint32_t width;
  uint32_t height;
  void* data;
} typedef DeviceOutputCallbackData;

struct DeviceRenderCallbackData {
  DeviceCommonCallbackData common;
} typedef DeviceRenderCallbackData;

#endif /* LUMINARY_DEVICE_CALLBACK_H */

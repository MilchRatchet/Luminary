#ifndef LUMINARY_HOST_H
#define LUMINARY_HOST_H

#include "array.h"
#include "device.h"
#include "error.h"
#include "utils.h"

LUMINARY_API struct LuminaryHost {
  ARRAY Device* devices;
  Camera camera;
} typedef LuminaryHost;

LUMINARY_API enum LuminaryDeviceSelectorStrategy {
  /* Select the device with highest estimated compute performance. */
  LUMINARY_DEVICE_SELECTOR_STRATEGY_PERFORMANCE = 0,
  /* Select the device with highest amount of memory. */
  LUMINARY_DEVICE_SELECTOR_STRATEGY_MEMORY = 1,
  /* Select the device that matches a specified name. */
  LUMINARY_DEVICE_SELECTOR_STRATEGY_NAME = 2
} typedef LuminaryDeviceSelectorStrategy;

LUMINARY_API struct LuminaryDeviceSelector {
  LuminaryDeviceSelectorStrategy strategy;
  const char* name;
} typedef LuminaryDeviceSelector;

LUMINARY_API LuminaryResult luminary_host_create(LuminaryHost** host);
LUMINARY_API LuminaryResult luminary_host_destroy(LuminaryHost** host);

LUMINARY_API LuminaryResult luminary_host_add_device(LuminaryHost* host, LuminaryDeviceSelector luminary_device_selector);
LUMINARY_API LuminaryResult luminary_host_remove_device(LuminaryHost* host, LuminaryDeviceSelector luminary_device_selector);

LUMINARY_API LuminaryResult luminary_host_load_lum_file(LuminaryHost* host);
LUMINARY_API LuminaryResult luminary_host_load_obj_file(LuminaryHost* host);

LUMINARY_API LuminaryResult luminary_host_start_render(LuminaryHost* host);
LUMINARY_API LuminaryResult luminary_host_skip_render(LuminaryHost* host);
LUMINARY_API LuminaryResult luminary_host_stop_render(LuminaryHost* host);

LUMINARY_API LuminaryResult luminary_host_get_last_render(LuminaryHost* host);

#endif /* LUMINARY_HOST_H */

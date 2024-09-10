#ifndef LUMINARY_HOST_H
#define LUMINARY_HOST_H

#include <luminary/api_utils.h>
#include <luminary/error.h>

struct LuminaryHost;
typedef struct LuminaryHost LuminaryHost;

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

LUMINARY_API LuminaryResult luminary_host_set_enable_output(LuminaryHost* host, bool enable_output);
LUMINARY_API LuminaryResult luminary_host_get_last_render(LuminaryHost* host);

LUMINARY_API LuminaryResult luminary_host_get_camera(LuminaryHost* host, LuminaryCamera& camera);
LUMINARY_API LuminaryResult luminary_host_set_camera(LuminaryHost* host, LuminaryCamera camera);

#endif /* LUMINARY_HOST_H */

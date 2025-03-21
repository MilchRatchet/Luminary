#include "device/device.h"
#include "internal_host_memory.h"
#include "internal_log.h"
#include "utils.h"

void luminary_init(void) {
  // Order of initialization is very important.
  _log_init();
  _host_memory_init();
  _device_init();
}

void luminary_shutdown(void) {
  _device_shutdown();
  _host_memory_shutdown();
  _log_shutdown();
}

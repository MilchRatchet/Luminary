#include "config.h"
#include "device/device.h"
#include "error_registry.h"
#include "internal_host_memory.h"
#include "internal_log.h"
#include "internal_stack_trace.h"
#include "utils.h"

void luminary_init(void) {
  // Order of initialization is very important.
  _log_init();

  info_message("Luminary %s", LUMINARY_VERSION);
  info_message("Build: %s (%s) - %s", LUMINARY_BRANCH_NAME, LUMINARY_VERSION_HASH, LUMINARY_VERSION_DATE);

  _host_memory_init();
  _error_registry_init();
  _stack_trace_init();

  info_message("Luminary finished initialization.");
}

void luminary_shutdown(void) {
  _stack_trace_shutdown();
  _error_registry_uninit();
  _host_memory_shutdown();
  _log_shutdown();
}

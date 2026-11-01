#include "debug_window.h"

#include "display.h"
#include "elements/separator.h"

static const char* const _window_debug_type_names[WINDOW_DEBUG_TYPE_COUNT] = {
  [WINDOW_DEBUG_TYPE_HOST_MEMORY]   = "Host Memory",
  [WINDOW_DEBUG_TYPE_DEVICE_MEMORY] = "Device Memory",
};

static void _window_debug_action(
  Window* window, Display* display, LuminaryHost* host, const MouseState* mouse_state, WindowDebugType type) {
  MD_CHECK_NULL_ARGUMENT(window);
  MD_CHECK_NULL_ARGUMENT(display);
  MD_CHECK_NULL_ARGUMENT(host);

  element_separator(
    window, mouse_state,
    (ElementSeparatorArgs) {.text = window_debug_type_get_name(type), .size = (ElementSize) {.rel_width = 1.0f, .height = 32}});
}

static void _window_debug_host_memory_action(Window* window, Display* display, LuminaryHost* host, const MouseState* mouse_state) {
  _window_debug_action(window, display, host, mouse_state, WINDOW_DEBUG_TYPE_HOST_MEMORY);
}

static void _window_debug_device_memory_action(Window* window, Display* display, LuminaryHost* host, const MouseState* mouse_state) {
  _window_debug_action(window, display, host, mouse_state, WINDOW_DEBUG_TYPE_DEVICE_MEMORY);
}

static void (*const _window_debug_action_funcs[WINDOW_DEBUG_TYPE_COUNT])(
  Window* window, Display* display, LuminaryHost* host, const MouseState* mouse_state) = {
  [WINDOW_DEBUG_TYPE_HOST_MEMORY]   = _window_debug_host_memory_action,
  [WINDOW_DEBUG_TYPE_DEVICE_MEMORY] = _window_debug_device_memory_action,
};

const char* window_debug_type_get_name(WindowDebugType type) {
  if (type >= WINDOW_DEBUG_TYPE_COUNT) {
    crash_message("Invalid debug window type was passed.");
  }

  return _window_debug_type_names[type];
}

void window_debug_create(Window** window, WindowDebugType type) {
  MD_CHECK_NULL_ARGUMENT(window);

  if (type >= WINDOW_DEBUG_TYPE_COUNT) {
    crash_message("Invalid debug window type was passed.");
  }

  window_create(window);

  (*window)->type            = WINDOW_TYPE_DEBUG;
  (*window)->visibility_mask = WINDOW_VISIBILITY_UTILITIES;
  (*window)->x               = 192 + (((uint32_t) type) * 64);
  (*window)->y               = 192 + (((uint32_t) type) * 64);
  (*window)->width           = 512;
  (*window)->height          = 128;
  (*window)->padding         = 8;
  (*window)->is_horizontal   = false;
  (*window)->is_visible      = false;
  (*window)->is_movable      = true;
  (*window)->background      = true;
  (*window)->auto_size       = true;
  (*window)->action_func     = _window_debug_action_funcs[type];
  (*window)->fixed_depth     = false;

  window_create_subwindow(*window);

  window_allocate_memory(*window);
}

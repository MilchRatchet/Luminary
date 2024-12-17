#include "entity_properties.h"

static bool _window_entity_properties_action(Window* window, Display* display, LuminaryHost* host) {
  MD_CHECK_NULL_ARGUMENT(window);
  MD_CHECK_NULL_ARGUMENT(display);
  MD_CHECK_NULL_ARGUMENT(host);

  return false;
}

void window_entity_properties_create(Window** window) {
  MD_CHECK_NULL_ARGUMENT(window);

  window_create(window);

  (*window)->x             = 128;
  (*window)->y             = 128;
  (*window)->width         = 256;
  (*window)->height        = 512;
  (*window)->padding       = 16;
  (*window)->is_horizontal = false;
  (*window)->is_visible    = true;
  (*window)->background    = true;
  (*window)->action_func   = _window_entity_properties_action;

  window_allocate_memory(*window);
}

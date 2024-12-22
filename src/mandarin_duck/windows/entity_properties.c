#include "entity_properties.h"

#include "elements/text.h"

static bool _window_entity_properties_action(Window* window, Display* display, LuminaryHost* host) {
  MD_CHECK_NULL_ARGUMENT(window);
  MD_CHECK_NULL_ARGUMENT(display);
  MD_CHECK_NULL_ARGUMENT(host);

  window_push_section(window, 24, 0);
  {
    element_text(
      window, display,
      (ElementTextArgs){
        .color = 0xFFFFFFFF, .size = (ElementSize){.is_relative = true, .rel_width = 0.5f, .rel_height = 1.0f}, .text = "Camera"});

    element_text(
      window, display,
      (ElementTextArgs){
        .color = 0xFFFFFFFF, .size = (ElementSize){.is_relative = true, .rel_width = 0.5f, .rel_height = 1.0f}, .text = "Example"});
  }
  window_pop_section(window);

  element_text(
    window, display,
    (ElementTextArgs){.color = 0xFFFFFFFF, .size = (ElementSize){.is_relative = false, .width = 128, .height = 24}, .text = "Position X"});

  return false;
}

void window_entity_properties_create(Window** window) {
  MD_CHECK_NULL_ARGUMENT(window);

  window_create(window);

  (*window)->x             = 128;
  (*window)->y             = 128;
  (*window)->width         = 256;
  (*window)->height        = 512;
  (*window)->padding       = 8;
  (*window)->is_horizontal = false;
  (*window)->is_visible    = true;
  (*window)->background    = true;
  (*window)->action_func   = _window_entity_properties_action;

  window_allocate_memory(*window);
}

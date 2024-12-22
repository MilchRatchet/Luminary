#include "entity_properties.h"

#include <float.h>

#include "elements/slider.h"
#include "elements/text.h"

static bool _window_entity_properties_action(Window* window, Display* display, LuminaryHost* host) {
  MD_CHECK_NULL_ARGUMENT(window);
  MD_CHECK_NULL_ARGUMENT(display);
  MD_CHECK_NULL_ARGUMENT(host);

  window_push_section(window, 24, 0);
  {
    window_margin_relative(window, 0.25f);
    element_text(
      window, display,
      (ElementTextArgs){
        .color    = 0xFFFFFFFF,
        .size     = (ElementSize){.is_relative = true, .rel_width = 0.5f, .rel_height = 1.0f},
        .text     = "Camera",
        .center_x = true,
        .center_y = true});
  }
  window_pop_section(window);

  window_push_section(window, 24, 0);
  {
    element_text(
      window, display,
      (ElementTextArgs){
        .color    = 0xFFFFFFFF,
        .size     = (ElementSize){.is_relative = true, .rel_width = 0.5f, .rel_height = 1.0f},
        .text     = "Position X",
        .center_x = false,
        .center_y = true});

    float test = 1337.0f;

    element_slider(
      window, display,
      (ElementSliderArgs){
        .color        = 0xFFFFFFFF,
        .size         = (ElementSize){.is_relative = true, .rel_width = 0.5f, .rel_height = 1.0f},
        .data_binding = &test,
        .is_integer   = false,
        .min          = 0.0f,
        .max          = FLT_MAX,
        .center_x     = false,
        .center_y     = true});
  }

  window_pop_section(window);

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

#include "entity_properties.h"

#include <float.h>

#include "elements/slider.h"
#include "elements/text.h"

static bool _window_entity_properties_action(Window* window, Display* display, LuminaryHost* host) {
  MD_CHECK_NULL_ARGUMENT(window);
  MD_CHECK_NULL_ARGUMENT(display);
  MD_CHECK_NULL_ARGUMENT(host);

  LuminaryCamera camera;
  LUM_FAILURE_HANDLE(luminary_host_get_camera(host, &camera));

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

  window_push_section(window, 32, 0);
  {
    element_text(
      window, display,
      (ElementTextArgs){
        .color    = 0xFFFFFFFF,
        .size     = (ElementSize){.is_relative = true, .rel_width = 0.5f, .rel_height = 0.75f},
        .text     = "Position",
        .center_x = false,
        .center_y = true});

    element_slider(
      window, display,
      (ElementSliderArgs){
        .type              = ELEMENT_SLIDER_DATA_TYPE_VECTOR,
        .color             = 0xFFFFFFFF,
        .size              = (ElementSize){.is_relative = true, .rel_width = 0.5f, .rel_height = 0.75f},
        .data_binding      = &camera.pos,
        .min               = -FLT_MAX,
        .max               = FLT_MAX,
        .component_padding = 4,
        .center_x          = false,
        .center_y          = true});
  }
  window_pop_section(window);

  window_push_section(window, 32, 0);
  {
    element_text(
      window, display,
      (ElementTextArgs){
        .color    = 0xFFFFFFFF,
        .size     = (ElementSize){.is_relative = true, .rel_width = 0.5f, .rel_height = 0.75f},
        .text     = "Rotation",
        .center_x = false,
        .center_y = true});

    element_slider(
      window, display,
      (ElementSliderArgs){
        .type              = ELEMENT_SLIDER_DATA_TYPE_VECTOR,
        .color             = 0xFFFFFFFF,
        .size              = (ElementSize){.is_relative = true, .rel_width = 0.5f, .rel_height = 0.75f},
        .data_binding      = &camera.rotation,
        .min               = -FLT_MAX,
        .max               = FLT_MAX,
        .component_padding = 4,
        .center_x          = false,
        .center_y          = true});
  }
  window_pop_section(window);

  window_push_section(window, 32, 0);
  {
    element_text(
      window, display,
      (ElementTextArgs){
        .color    = 0xFFFFFFFF,
        .size     = (ElementSize){.is_relative = true, .rel_width = 0.5f, .rel_height = 0.75f},
        .text     = "Field of View",
        .center_x = false,
        .center_y = true});

    element_slider(
      window, display,
      (ElementSliderArgs){
        .type              = ELEMENT_SLIDER_DATA_TYPE_FLOAT,
        .color             = 0xFFFFFFFF,
        .size              = (ElementSize){.is_relative = true, .rel_width = 0.5f, .rel_height = 0.75f},
        .data_binding      = &camera.fov,
        .min               = 0.0f,
        .max               = FLT_MAX,
        .component_padding = 4,
        .center_x          = true,
        .center_y          = true});
  }
  window_pop_section(window);

  window_push_section(window, 32, 0);
  {
    element_text(
      window, display,
      (ElementTextArgs){
        .color    = 0xFFFFFFFF,
        .size     = (ElementSize){.is_relative = true, .rel_width = 0.5f, .rel_height = 0.75f},
        .text     = "Focal Length",
        .center_x = false,
        .center_y = true});

    element_slider(
      window, display,
      (ElementSliderArgs){
        .type              = ELEMENT_SLIDER_DATA_TYPE_FLOAT,
        .color             = 0xFFFFFFFF,
        .size              = (ElementSize){.is_relative = true, .rel_width = 0.5f, .rel_height = 0.75f},
        .data_binding      = &camera.focal_length,
        .min               = 0.0f,
        .max               = FLT_MAX,
        .component_padding = 4,
        .center_x          = true,
        .center_y          = true});
  }
  window_pop_section(window);

  return false;
}

void window_entity_properties_create(Window** window) {
  MD_CHECK_NULL_ARGUMENT(window);

  window_create(window);

  (*window)->x             = 128;
  (*window)->y             = 128;
  (*window)->width         = 384;
  (*window)->height        = 512;
  (*window)->padding       = 8;
  (*window)->is_horizontal = false;
  (*window)->is_visible    = true;
  (*window)->background    = true;
  (*window)->action_func   = _window_entity_properties_action;

  window_allocate_memory(*window);
}

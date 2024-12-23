#include "entity_properties.h"

#include <float.h>

#include "elements/slider.h"
#include "elements/text.h"

static void _window_entity_properties_add_slider(
  Window* window, Display* display, const char* text, void* data_binding, ElementSliderDataType data_type, float min, float max) {
  MD_CHECK_NULL_ARGUMENT(window);
  MD_CHECK_NULL_ARGUMENT(display);

  window_push_section(window, 32, 0);
  {
    element_text(
      window, display,
      (ElementTextArgs){
        .color    = 0xFFFFFFFF,
        .size     = (ElementSize){.is_relative = true, .rel_width = 0.4f, .rel_height = 0.75f},
        .text     = text,
        .center_x = false,
        .center_y = true});

    element_slider(
      window, display,
      (ElementSliderArgs){
        .type              = data_type,
        .color             = 0xFFFFFFFF,
        .size              = (ElementSize){.is_relative = true, .rel_width = 0.6f, .rel_height = 0.75f},
        .data_binding      = data_binding,
        .min               = min,
        .max               = max,
        .component_padding = 4,
        .center_x          = true,
        .center_y          = true});
  }
  window_pop_section(window);
}

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

  _window_entity_properties_add_slider(window, display, "Position", &camera.pos, ELEMENT_SLIDER_DATA_TYPE_VECTOR, -FLT_MAX, FLT_MAX);
  _window_entity_properties_add_slider(window, display, "Rotation", &camera.rotation, ELEMENT_SLIDER_DATA_TYPE_VECTOR, -FLT_MAX, FLT_MAX);
  _window_entity_properties_add_slider(window, display, "Field of View", &camera.fov, ELEMENT_SLIDER_DATA_TYPE_FLOAT, 0.0f, FLT_MAX);
  _window_entity_properties_add_slider(
    window, display, "Focal Length", &camera.focal_length, ELEMENT_SLIDER_DATA_TYPE_FLOAT, 0.0f, FLT_MAX);

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

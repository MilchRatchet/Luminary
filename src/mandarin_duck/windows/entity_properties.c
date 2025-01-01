#include "entity_properties.h"

#include <float.h>

#include "elements/checkbox.h"
#include "elements/color.h"
#include "elements/dropdown.h"
#include "elements/slider.h"
#include "elements/text.h"

static bool _window_entity_properties_add_slider(
  Window* window, Display* display, const char* text, void* data_binding, ElementSliderDataType data_type, float min, float max,
  float change_rate) {
  MD_CHECK_NULL_ARGUMENT(window);
  MD_CHECK_NULL_ARGUMENT(display);

  bool update_data = false;

  window_push_section(window, 32, 0);
  {
    element_text(
      window, display,
      (ElementTextArgs){
        .color    = 0xFFFFFFFF,
        .size     = (ElementSize){.width = ELEMENT_SIZE_INVALID, .rel_width = 0.4f, .height = ELEMENT_SIZE_INVALID, .rel_height = 0.75f},
        .text     = text,
        .center_x = false,
        .center_y = true,
        .highlighting = false,
        .cache_text   = true});

    if (data_type == ELEMENT_SLIDER_DATA_TYPE_RGB) {
      LuminaryRGBF color = *(LuminaryRGBF*) data_binding;

      uint32_t color_bits = 0xFF000000;

      color_bits |= ((uint32_t) fminf(255.0f, fmaxf(0.0f, (color.r * 255.0f)))) << 16;
      color_bits |= ((uint32_t) fminf(255.0f, fmaxf(0.0f, (color.g * 255.0f)))) << 8;
      color_bits |= ((uint32_t) fminf(255.0f, fmaxf(0.0f, (color.b * 255.0f)))) << 0;

      element_color(window, display, (ElementColorArgs){.size = (ElementSize){.width = 24, .height = 24}, .color = color_bits});

      window_margin(window, 4);
    }

    if (element_slider(
          window, display,
          (ElementSliderArgs){
            .identifier = text,
            .type       = data_type,
            .color      = 0xFFFFFFFF,
            .size = (ElementSize){.width = ELEMENT_SIZE_INVALID, .rel_width = 1.0f, .height = ELEMENT_SIZE_INVALID, .rel_height = 0.75f},
            .data_binding      = data_binding,
            .min               = min,
            .max               = max,
            .change_rate       = change_rate,
            .component_padding = 4,
            .margins           = 4,
            .center_x          = true,
            .center_y          = true})) {
      update_data = true;
    }
  }
  window_pop_section(window);

  return update_data;
}

static bool _window_entity_properties_add_checkbox(Window* window, Display* display, const char* text, void* data_binding) {
  MD_CHECK_NULL_ARGUMENT(window);
  MD_CHECK_NULL_ARGUMENT(display);

  bool update_data = false;

  window_push_section(window, 32, 0);
  {
    element_text(
      window, display,
      (ElementTextArgs){
        .color    = 0xFFFFFFFF,
        .size     = (ElementSize){.width = ELEMENT_SIZE_INVALID, .rel_width = 0.4f, .height = ELEMENT_SIZE_INVALID, .rel_height = 0.75f},
        .text     = text,
        .center_x = false,
        .center_y = true,
        .highlighting = false,
        .cache_text   = true});

    if (element_checkbox(
          window, display, (ElementCheckBoxArgs){.size = (ElementSize){.width = 24, .height = 24}, .data_binding = data_binding})) {
      update_data = true;
    }
  }
  window_pop_section(window);

  return update_data;
}

static bool _window_entity_properties_add_dropdown(
  Window* window, Display* display, const char* text, uint32_t num_strings, char** strings, uint32_t* selected_index) {
  MD_CHECK_NULL_ARGUMENT(window);
  MD_CHECK_NULL_ARGUMENT(display);

  bool update_data = false;

  window_push_section(window, 32, 0);
  {
    element_text(
      window, display,
      (ElementTextArgs){
        .color    = 0xFFFFFFFF,
        .size     = (ElementSize){.width = ELEMENT_SIZE_INVALID, .rel_width = 0.4f, .height = ELEMENT_SIZE_INVALID, .rel_height = 0.75f},
        .text     = text,
        .center_x = false,
        .center_y = true,
        .highlighting = false,
        .cache_text   = true});

    if (element_dropdown(
          window, display,
          (ElementDropdownArgs){
            .identifier = text,
            .size = (ElementSize){.width = ELEMENT_SIZE_INVALID, .rel_width = 1.0f, .height = ELEMENT_SIZE_INVALID, .rel_height = 0.75f},
            .selected_index = selected_index,
            .num_strings    = num_strings,
            .strings        = strings})) {
      update_data = true;
    }
  }
  window_pop_section(window);

  return update_data;
}

static bool _window_entity_properties_action(Window* window, Display* display, LuminaryHost* host) {
  MD_CHECK_NULL_ARGUMENT(window);
  MD_CHECK_NULL_ARGUMENT(display);
  MD_CHECK_NULL_ARGUMENT(host);

  LuminaryCamera camera;
  LUM_FAILURE_HANDLE(luminary_host_get_camera(host, &camera));

  uint32_t tonemap        = (uint32_t) camera.tonemap;
  uint32_t filter         = (uint32_t) camera.filter;
  uint32_t aperture_shape = (uint32_t) camera.aperture_shape;

  element_text(
    window, display,
    (ElementTextArgs){
      .color        = 0xFFFFFFFF,
      .size         = (ElementSize){.width = ELEMENT_SIZE_INVALID, .rel_width = 1.0f, .height = 24},
      .text         = "Camera",
      .center_x     = true,
      .center_y     = true,
      .highlighting = false,
      .cache_text   = true});

  window_margin(window, 16);

  bool update_data = false;

  update_data |= _window_entity_properties_add_slider(
    window, display, "Position", &camera.pos, ELEMENT_SLIDER_DATA_TYPE_VECTOR, -FLT_MAX, FLT_MAX, 1.0f);
  update_data |= _window_entity_properties_add_slider(
    window, display, "Rotation", &camera.rotation, ELEMENT_SLIDER_DATA_TYPE_VECTOR, -FLT_MAX, FLT_MAX, 1.0f);
  update_data |= _window_entity_properties_add_slider(
    window, display, "Field of View", &camera.fov, ELEMENT_SLIDER_DATA_TYPE_FLOAT, 0.0f, FLT_MAX, 1.0f);
  update_data |= _window_entity_properties_add_slider(
    window, display, "Aperture Size", &camera.aperture_size, ELEMENT_SLIDER_DATA_TYPE_FLOAT, 0.0f, FLT_MAX, 1.0f);

  if (camera.aperture_size > 0.0f) {
    update_data |= _window_entity_properties_add_dropdown(
      window, display, "Aperture Shape", LUMINARY_APERTURE_COUNT, (char**) luminary_strings_aperture, &aperture_shape);

    if (aperture_shape == (uint32_t) LUMINARY_APERTURE_BLADED) {
      update_data |= _window_entity_properties_add_slider(
        window, display, "Aperture Blade Count", &camera.aperture_blade_count, ELEMENT_SLIDER_DATA_TYPE_UINT, 1.0f, FLT_MAX, 5.0f);
    }
  }

  update_data |= _window_entity_properties_add_slider(
    window, display, "Focal Length", &camera.focal_length, ELEMENT_SLIDER_DATA_TYPE_FLOAT, 0.0f, FLT_MAX, 1.0f);
  update_data |= _window_entity_properties_add_checkbox(window, display, "Firefly Clamping", &camera.do_firefly_clamping);
  update_data |= _window_entity_properties_add_slider(
    window, display, "Russian Roulette Threshold", &camera.russian_roulette_threshold, ELEMENT_SLIDER_DATA_TYPE_FLOAT, 0.0f, FLT_MAX, 1.0f);

  // TODO: Add separator

  update_data |=
    _window_entity_properties_add_dropdown(window, display, "Tonemap", LUMINARY_TONEMAP_COUNT, (char**) luminary_strings_tonemap, &tonemap);
  update_data |= _window_entity_properties_add_slider(
    window, display, "Exposure", &camera.exposure, ELEMENT_SLIDER_DATA_TYPE_FLOAT, 0.0f, FLT_MAX, 5.0f);
  update_data |= _window_entity_properties_add_slider(
    window, display, "Bloom", &camera.bloom_blend, ELEMENT_SLIDER_DATA_TYPE_FLOAT, 0.0f, FLT_MAX, 5.0f);
  update_data |= _window_entity_properties_add_slider(
    window, display, "Film Grain", &camera.film_grain, ELEMENT_SLIDER_DATA_TYPE_FLOAT, 0.0f, 1.0f, 0.5f);
  update_data |= _window_entity_properties_add_checkbox(window, display, "Lens Flare", &camera.lens_flare);
  update_data |= _window_entity_properties_add_slider(
    window, display, "Lens Flare Threshold", &camera.lens_flare_threshold, ELEMENT_SLIDER_DATA_TYPE_FLOAT, 0.0f, FLT_MAX, 5.0f);
  update_data |= _window_entity_properties_add_checkbox(window, display, "Purkinje Shift", &camera.purkinje);

  if (camera.purkinje) {
    update_data |= _window_entity_properties_add_slider(
      window, display, "Purkinje Blueness", &camera.purkinje_kappa1, ELEMENT_SLIDER_DATA_TYPE_FLOAT, 0.0f, FLT_MAX, 5.0f);
    update_data |= _window_entity_properties_add_slider(
      window, display, "Purkinje Brightness", &camera.purkinje_kappa2, ELEMENT_SLIDER_DATA_TYPE_FLOAT, 0.0f, FLT_MAX, 5.0f);
  }

  update_data |=
    _window_entity_properties_add_dropdown(window, display, "Filter", LUMINARY_FILTER_COUNT, (char**) luminary_strings_filter, &filter);
  update_data |= _window_entity_properties_add_checkbox(window, display, "Dithering", &camera.dithering);

  update_data |=
    _window_entity_properties_add_slider(window, display, "Test", &camera.color_correction, ELEMENT_SLIDER_DATA_TYPE_RGB, 0.0f, 1.0f, 1.0f);

  if (update_data) {
    camera.tonemap        = (LuminaryToneMap) tonemap;
    camera.filter         = (LuminaryFilter) filter;
    camera.aperture_shape = (LuminaryApertureShape) aperture_shape;

    LUM_FAILURE_HANDLE(luminary_host_set_camera(host, &camera));
  }

  return update_data;
}

void window_entity_properties_create(Window** window) {
  MD_CHECK_NULL_ARGUMENT(window);

  window_create(window);

  (*window)->x             = 128;
  (*window)->y             = 128;
  (*window)->width         = 512;
  (*window)->height        = 512;
  (*window)->padding       = 8;
  (*window)->is_horizontal = false;
  (*window)->is_visible    = true;
  (*window)->is_movable    = true;
  (*window)->background    = true;
  (*window)->auto_size     = true;
  (*window)->action_func   = _window_entity_properties_action;

  window_create_subwindow(*window);

  window_allocate_memory(*window);
}

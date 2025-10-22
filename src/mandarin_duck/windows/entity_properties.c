#include "entity_properties.h"

#include <float.h>

#include "display.h"
#include "elements/button.h"
#include "elements/checkbox.h"
#include "elements/color.h"
#include "elements/dropdown.h"
#include "elements/separator.h"
#include "elements/slider.h"
#include "elements/text.h"

struct WindowEntityPropertiesPassingData {
  Window* window;
  Display* display;
  const MouseState* mouse_state;
  const KeyboardState* keyboard_state;
} typedef WindowEntityPropertiesPassingData;

enum EntityPropertyButtonFunction {
  ENTITY_PROPERTY_BUTTON_FUNCTION_COMPUTE_HDRI,
  ENTITY_PROPERTY_BUTTON_FUNCTION_COUNT
} typedef EntityPropertyButtonFunction;

static bool _window_entity_properties_add_slider(
  WindowEntityPropertiesPassingData data, const char* text, void* data_binding, ElementSliderDataType data_type, float min, float max,
  float change_rate) {
  bool update_data = false;

  window_push_section(data.window, 32, 0);
  {
    element_text(
      data.window, data.display, data.mouse_state,
      (ElementTextArgs) {.color        = 0xFFFFFFFF,
                         .size         = (ElementSize) {.rel_width = 0.4f, .rel_height = 0.75f},
                         .text         = text,
                         .center_x     = false,
                         .center_y     = true,
                         .highlighting = false,
                         .cache_text   = true,
                         .auto_size    = false,
                         .is_clickable = false});

    if (data_type == ELEMENT_SLIDER_DATA_TYPE_RGB) {
      LuminaryRGBF color = *(LuminaryRGBF*) data_binding;

      uint32_t color_bits = 0xFF000000;

      color_bits |= ((uint32_t) fminf(255.0f, fmaxf(0.0f, (color.r * 255.0f)))) << 16;
      color_bits |= ((uint32_t) fminf(255.0f, fmaxf(0.0f, (color.g * 255.0f)))) << 8;
      color_bits |= ((uint32_t) fminf(255.0f, fmaxf(0.0f, (color.b * 255.0f)))) << 0;

      element_color(
        data.window, data.mouse_state, (ElementColorArgs) {.size = (ElementSize) {.width = 24, .height = 24}, .color = color_bits});

      window_margin(data.window, 4);
    }

    if (element_slider(
          data.window, data.display, data.mouse_state, data.keyboard_state,
          (ElementSliderArgs) {.identifier        = text,
                               .type              = data_type,
                               .color             = 0xFFFFFFFF,
                               .size              = (ElementSize) {.rel_width = 1.0f, .rel_height = 0.75f},
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
  window_pop_section(data.window);

  return update_data;
}

static bool _window_entity_properties_add_checkbox(WindowEntityPropertiesPassingData data, const char* text, void* data_binding) {
  bool update_data = false;

  window_push_section(data.window, 32, 0);
  {
    element_text(
      data.window, data.display, data.mouse_state,
      (ElementTextArgs) {.color        = 0xFFFFFFFF,
                         .size         = (ElementSize) {.rel_width = 0.4f, .rel_height = 0.75f},
                         .text         = text,
                         .center_x     = false,
                         .center_y     = true,
                         .highlighting = false,
                         .cache_text   = true,
                         .auto_size    = false,
                         .is_clickable = false});

    if (element_checkbox(
          data.window, data.display, data.mouse_state,
          (ElementCheckBoxArgs) {.size = (ElementSize) {.width = 24, .height = 24}, .data_binding = data_binding})) {
      update_data = true;
    }
  }
  window_pop_section(data.window);

  return update_data;
}

static bool _window_entity_properties_add_button(WindowEntityPropertiesPassingData data, const char* text) {
  bool update_data = false;

  window_push_section(data.window, 32, 0);
  {
    element_text(
      data.window, data.display, data.mouse_state,
      (ElementTextArgs) {.color        = 0xFFFFFFFF,
                         .size         = (ElementSize) {.rel_width = 0.4f, .rel_height = 0.75f},
                         .text         = text,
                         .center_x     = false,
                         .center_y     = true,
                         .highlighting = false,
                         .cache_text   = true,
                         .auto_size    = false,
                         .is_clickable = false});

    if (element_button(
          data.window, data.display, data.mouse_state,
          (ElementButtonArgs) {.size         = (ElementSize) {.width = 24, .height = 24},
                               .shape        = ELEMENT_BUTTON_SHAPE_IMAGE,
                               .image        = ELEMENT_BUTTON_IMAGE_SYNC,
                               .color        = MD_COLOR_GRAY,
                               .hover_color  = MD_COLOR_ACCENT_LIGHT_2,
                               .press_color  = MD_COLOR_WHITE,
                               .tooltip_text = text})) {
      update_data = true;
    }
  }
  window_pop_section(data.window);

  return update_data;
}

static bool _window_entity_properties_add_dropdown(
  WindowEntityPropertiesPassingData data, const char* text, uint32_t num_strings, char** strings, uint32_t* selected_index) {
  bool update_data = false;

  window_push_section(data.window, 32, 0);
  {
    element_text(
      data.window, data.display, data.mouse_state,
      (ElementTextArgs) {.color        = 0xFFFFFFFF,
                         .size         = (ElementSize) {.rel_width = 0.4f, .rel_height = 0.75f},
                         .text         = text,
                         .center_x     = false,
                         .center_y     = true,
                         .highlighting = false,
                         .cache_text   = true,
                         .auto_size    = false,
                         .is_clickable = false});

    if (element_dropdown(
          data.window, data.display, data.mouse_state,
          (ElementDropdownArgs) {.identifier     = text,
                                 .size           = (ElementSize) {.rel_width = 1.0f, .rel_height = 0.75f},
                                 .selected_index = selected_index,
                                 .num_strings    = num_strings,
                                 .strings        = strings})) {
      update_data = true;
    }
  }
  window_pop_section(data.window);

  return update_data;
}

static void _window_entity_properties_renderer_settings_action(
  Window* window, Display* display, LuminaryHost* host, const MouseState* mouse_state) {
  MD_CHECK_NULL_ARGUMENT(window);
  MD_CHECK_NULL_ARGUMENT(display);
  MD_CHECK_NULL_ARGUMENT(host);

  WindowEntityPropertiesPassingData data = {
    .window = window, .display = display, .mouse_state = mouse_state, .keyboard_state = display->keyboard_state};

  LuminaryRendererSettings settings;
  LUM_FAILURE_HANDLE(luminary_host_get_settings(host, &settings));

  uint32_t shading_mode = (uint32_t) settings.shading_mode;

  element_separator(
    window, mouse_state, (ElementSeparatorArgs) {.text = "Renderer Settings", .size = (ElementSize) {.rel_width = 1.0f, .height = 32}});

  bool update_data = false;

  update_data |= _window_entity_properties_add_slider(data, "Width", &settings.width, ELEMENT_SLIDER_DATA_TYPE_UINT, 0.0f, 16535.0f, 1.0f);
  update_data |=
    _window_entity_properties_add_slider(data, "Height", &settings.height, ELEMENT_SLIDER_DATA_TYPE_UINT, 0.0f, 16535.0f, 1.0f);

  if (display->sync_render_resolution) {
    if (settings.width != display->width || settings.height != display->height) {
      // If the display is maximized, we unmaximize it here
      if (display->is_maximized)
        display_handle_maximize(display, host, false);

      display_resize(display, settings.width, settings.height);
    }
  }

  update_data |=
    _window_entity_properties_add_slider(data, "Max Depth", &settings.max_ray_depth, ELEMENT_SLIDER_DATA_TYPE_UINT, 0.0f, 63.0f, 1.0f);
  update_data |= _window_entity_properties_add_slider(
    data, "Bridges Max Num Vertices", &settings.bridge_max_num_vertices, ELEMENT_SLIDER_DATA_TYPE_UINT, 1.0f, 15.0f, 1.0f);
  update_data |=
    _window_entity_properties_add_slider(data, "Undersampling", &settings.undersampling, ELEMENT_SLIDER_DATA_TYPE_UINT, 0.0f, 6.0f, 1.0f);
  update_data |=
    _window_entity_properties_add_slider(data, "Supersampling", &settings.supersampling, ELEMENT_SLIDER_DATA_TYPE_UINT, 0.0f, 3.0f, 1.0f);
  update_data |= _window_entity_properties_add_dropdown(
    data, "Shading Mode", LUMINARY_SHADING_MODE_COUNT, (char**) luminary_strings_shading_mode, &shading_mode);

  if (update_data) {
    settings.shading_mode = (LuminaryShadingMode) shading_mode;

    LUM_FAILURE_HANDLE(luminary_host_set_settings(host, &settings));
  }
}

static void _window_entity_properties_camera_action(Window* window, Display* display, LuminaryHost* host, const MouseState* mouse_state) {
  MD_CHECK_NULL_ARGUMENT(window);
  MD_CHECK_NULL_ARGUMENT(display);
  MD_CHECK_NULL_ARGUMENT(host);

  WindowEntityPropertiesPassingData data = {
    .window = window, .display = display, .mouse_state = mouse_state, .keyboard_state = display->keyboard_state};

  LuminaryCamera camera;
  LUM_FAILURE_HANDLE(luminary_host_get_camera(host, &camera));

  uint32_t tonemap        = (uint32_t) camera.tonemap;
  uint32_t filter         = (uint32_t) camera.filter;
  uint32_t aperture_shape = (uint32_t) camera.aperture_shape;

  element_separator(
    window, mouse_state, (ElementSeparatorArgs) {.text = "Camera", .size = (ElementSize) {.rel_width = 1.0f, .height = 32}});

  bool update_data = false;

  update_data |=
    _window_entity_properties_add_slider(data, "Position", &camera.pos, ELEMENT_SLIDER_DATA_TYPE_VECTOR, -FLT_MAX, FLT_MAX, 1.0f);
  update_data |=
    _window_entity_properties_add_slider(data, "Rotation", &camera.rotation, ELEMENT_SLIDER_DATA_TYPE_VECTOR, -FLT_MAX, FLT_MAX, 1.0f);

  update_data |= _window_entity_properties_add_checkbox(data, "Physical", &camera.use_physical_camera);

  if (camera.use_physical_camera) {
    update_data |= _window_entity_properties_add_slider(
      data, "Sensor Width", &camera.physical.sensor_width, ELEMENT_SLIDER_DATA_TYPE_FLOAT, 0.0f, FLT_MAX, 1.0f);
    update_data |= _window_entity_properties_add_slider(
      data, "Sensor Distance", &camera.physical.image_plane_distance, ELEMENT_SLIDER_DATA_TYPE_FLOAT, 0.0f, FLT_MAX, 1.0f);
    update_data |= _window_entity_properties_add_slider(
      data, "Aperture Diameter", &camera.physical.aperture_diameter, ELEMENT_SLIDER_DATA_TYPE_FLOAT, 0.0f, FLT_MAX, 1.0f);
    update_data |= _window_entity_properties_add_checkbox(data, "Reflections", &camera.physical.allow_reflections);
    update_data |= _window_entity_properties_add_checkbox(data, "Spectral", &camera.physical.use_spectral_rendering);
  }
  else {
    update_data |= _window_entity_properties_add_slider(
      data, "Field of View", &camera.thin_lens.fov, ELEMENT_SLIDER_DATA_TYPE_FLOAT, 0.0f, FLT_MAX, 1.0f);
    update_data |= _window_entity_properties_add_slider(
      data, "Aperture Size", &camera.thin_lens.aperture_size, ELEMENT_SLIDER_DATA_TYPE_FLOAT, 0.0f, FLT_MAX, 1.0f);
  }

  update_data |= _window_entity_properties_add_dropdown(
    data, "Aperture Shape", LUMINARY_APERTURE_COUNT, (char**) luminary_strings_aperture, &aperture_shape);

  if (aperture_shape == (uint32_t) LUMINARY_APERTURE_BLADED) {
    update_data |= _window_entity_properties_add_slider(
      data, "Aperture Blade Count", &camera.aperture_blade_count, ELEMENT_SLIDER_DATA_TYPE_UINT, 1.0f, FLT_MAX, 5.0f);
  }

  update_data |=
    _window_entity_properties_add_slider(data, "Scale", &camera.camera_scale, ELEMENT_SLIDER_DATA_TYPE_FLOAT, 0.01f, FLT_MAX, 1.0f);

  if (camera.use_physical_camera == false) {
    update_data |= _window_entity_properties_add_slider(
      data, "Object Distance", &camera.object_distance, ELEMENT_SLIDER_DATA_TYPE_FLOAT, 0.01f, FLT_MAX, 1.0f);
  }

  update_data |= _window_entity_properties_add_checkbox(data, "Firefly Rejection", &camera.do_firefly_rejection);
  update_data |= _window_entity_properties_add_checkbox(data, "Only Indirect Lighting", &camera.indirect_only);
  update_data |= _window_entity_properties_add_slider(
    data, "Russian Roulette Threshold", &camera.russian_roulette_threshold, ELEMENT_SLIDER_DATA_TYPE_FLOAT, 0.0f, FLT_MAX, 1.0f);

  element_separator(
    window, mouse_state, (ElementSeparatorArgs) {.text = "Post Process", .size = (ElementSize) {.rel_width = 1.0f, .height = 32}});

  update_data |=
    _window_entity_properties_add_dropdown(data, "Tonemap", LUMINARY_TONEMAP_COUNT, (char**) luminary_strings_tonemap, &tonemap);

  if (tonemap == LUMINARY_TONEMAP_AGX_CUSTOM) {
    update_data |= _window_entity_properties_add_slider(
      data, "AGX Power", &camera.agx_custom_power, ELEMENT_SLIDER_DATA_TYPE_FLOAT, 0.0f, FLT_MAX, 1.0f);
    update_data |= _window_entity_properties_add_slider(
      data, "AGX Saturation", &camera.agx_custom_saturation, ELEMENT_SLIDER_DATA_TYPE_FLOAT, 0.0f, FLT_MAX, 1.0f);
    update_data |= _window_entity_properties_add_slider(
      data, "AGX Slope", &camera.agx_custom_slope, ELEMENT_SLIDER_DATA_TYPE_FLOAT, 0.0f, FLT_MAX, 1.0f);
  }

  update_data |=
    _window_entity_properties_add_slider(data, "Exposure", &camera.exposure, ELEMENT_SLIDER_DATA_TYPE_FLOAT, -16.0f, 16.0f, 1.0f);
  update_data |= _window_entity_properties_add_slider(data, "Bloom", &camera.bloom_blend, ELEMENT_SLIDER_DATA_TYPE_FLOAT, 0.0f, 1.0f, 1.0f);
  update_data |=
    _window_entity_properties_add_slider(data, "Film Grain", &camera.film_grain, ELEMENT_SLIDER_DATA_TYPE_FLOAT, 0.0f, 1.0f, 0.5f);
  update_data |= _window_entity_properties_add_checkbox(data, "Lens Flare", &camera.lens_flare);
  update_data |= _window_entity_properties_add_slider(
    data, "Lens Flare Threshold", &camera.lens_flare_threshold, ELEMENT_SLIDER_DATA_TYPE_FLOAT, 0.0f, FLT_MAX, 5.0f);

  if (camera.use_physical_camera == false) {
    update_data |= _window_entity_properties_add_checkbox(data, "Purkinje Shift", &camera.purkinje);

    if (camera.purkinje) {
      update_data |= _window_entity_properties_add_slider(
        data, "Purkinje Blueness", &camera.purkinje_kappa1, ELEMENT_SLIDER_DATA_TYPE_FLOAT, 0.0f, FLT_MAX, 5.0f);
      update_data |= _window_entity_properties_add_slider(
        data, "Purkinje Brightness", &camera.purkinje_kappa2, ELEMENT_SLIDER_DATA_TYPE_FLOAT, 0.0f, FLT_MAX, 5.0f);
    }
  }

  update_data |= _window_entity_properties_add_dropdown(data, "Filter", LUMINARY_FILTER_COUNT, (char**) luminary_strings_filter, &filter);
  update_data |= _window_entity_properties_add_checkbox(data, "Dithering", &camera.dithering);

  update_data |= _window_entity_properties_add_checkbox(data, "Color Correction", &camera.use_color_correction);

  if (camera.use_color_correction) {
    update_data |=
      _window_entity_properties_add_slider(data, "Hue", &camera.color_correction.r, ELEMENT_SLIDER_DATA_TYPE_FLOAT, -1.0f, 1.0f, 1.0f);
    update_data |= _window_entity_properties_add_slider(
      data, "Saturation", &camera.color_correction.g, ELEMENT_SLIDER_DATA_TYPE_FLOAT, -1.0f, 1.0f, 1.0f);
    update_data |=
      _window_entity_properties_add_slider(data, "Value", &camera.color_correction.b, ELEMENT_SLIDER_DATA_TYPE_FLOAT, -1.0f, 1.0f, 1.0f);
  }

  if (update_data) {
    camera.tonemap        = (LuminaryToneMap) tonemap;
    camera.filter         = (LuminaryFilter) filter;
    camera.aperture_shape = (LuminaryApertureShape) aperture_shape;

    LUM_FAILURE_HANDLE(luminary_host_set_camera(host, &camera));
  }
}

static void _window_entity_properties_ocean_action(Window* window, Display* display, LuminaryHost* host, const MouseState* mouse_state) {
  MD_CHECK_NULL_ARGUMENT(window);
  MD_CHECK_NULL_ARGUMENT(display);
  MD_CHECK_NULL_ARGUMENT(host);

  WindowEntityPropertiesPassingData data = {
    .window = window, .display = display, .mouse_state = mouse_state, .keyboard_state = display->keyboard_state};

  LuminaryOcean ocean;
  LUM_FAILURE_HANDLE(luminary_host_get_ocean(host, &ocean));

  uint32_t water_type = (uint32_t) ocean.water_type;

  element_separator(window, mouse_state, (ElementSeparatorArgs) {.text = "Ocean", .size = (ElementSize) {.rel_width = 1.0f, .height = 32}});

  bool update_data = false;

  update_data |= _window_entity_properties_add_checkbox(data, "Active", &ocean.active);

  if (ocean.active) {
    update_data |=
      _window_entity_properties_add_slider(data, "Height", &ocean.height, ELEMENT_SLIDER_DATA_TYPE_FLOAT, -FLT_MAX, FLT_MAX, 1.0f);
    update_data |=
      _window_entity_properties_add_slider(data, "Amplitude", &ocean.amplitude, ELEMENT_SLIDER_DATA_TYPE_FLOAT, 0.0f, 1.0f, 1.0f);
    update_data |=
      _window_entity_properties_add_slider(data, "Frequency", &ocean.frequency, ELEMENT_SLIDER_DATA_TYPE_FLOAT, 0.0f, FLT_MAX, 1.0f);
    update_data |=
      _window_entity_properties_add_slider(data, "IOR", &ocean.refractive_index, ELEMENT_SLIDER_DATA_TYPE_FLOAT, 1.0f, 3.0f, 1.0f);
    update_data |= _window_entity_properties_add_dropdown(
      data, "Jerlov Water Type", LUMINARY_JERLOV_WATER_TYPE_COUNT, (char**) luminary_strings_jerlov_water_type, &water_type);

    update_data |= _window_entity_properties_add_checkbox(data, "Caustics", &ocean.caustics_active);

    if (ocean.caustics_active) {
      update_data |= _window_entity_properties_add_slider(
        data, "Caustics RIS Samples", &ocean.caustics_ris_sample_count, ELEMENT_SLIDER_DATA_TYPE_UINT, 1.0f, 128.0f, 1.0f);
      update_data |= _window_entity_properties_add_slider(
        data, "Caustics Domain Scale", &ocean.caustics_domain_scale, ELEMENT_SLIDER_DATA_TYPE_FLOAT, 0.0f, FLT_MAX, 1.0f);
    }

    update_data |= _window_entity_properties_add_checkbox(data, "Multiscattering", &ocean.multiscattering);
    update_data |= _window_entity_properties_add_checkbox(data, "Triangle Light Contribution", &ocean.triangle_light_contribution);
  }

  if (update_data) {
    ocean.water_type = (LuminaryJerlovWaterType) water_type;

    LUM_FAILURE_HANDLE(luminary_host_set_ocean(host, &ocean));
  }
}

static void _window_entity_properties_sky_action(Window* window, Display* display, LuminaryHost* host, const MouseState* mouse_state) {
  MD_CHECK_NULL_ARGUMENT(window);
  MD_CHECK_NULL_ARGUMENT(display);
  MD_CHECK_NULL_ARGUMENT(host);

  WindowEntityPropertiesPassingData data = {
    .window = window, .display = display, .mouse_state = mouse_state, .keyboard_state = display->keyboard_state};

  LuminarySky sky;
  LUM_FAILURE_HANDLE(luminary_host_get_sky(host, &sky));

  uint32_t mode = (uint32_t) sky.mode;

  element_separator(window, mouse_state, (ElementSeparatorArgs) {.text = "Sky", .size = (ElementSize) {.rel_width = 1.0f, .height = 32}});

  bool update_data = false;

  update_data |= _window_entity_properties_add_dropdown(data, "Mode", LUMINARY_SKY_MODE_COUNT, (char**) luminary_strings_sky_mode, &mode);

  switch ((LuminarySkyMode) mode) {
    case LUMINARY_SKY_MODE_HDRI:
      element_separator(
        window, mouse_state, (ElementSeparatorArgs) {.text = "HDRI Settings", .size = (ElementSize) {.rel_width = 1.0f, .height = 32}});

      update_data |=
        _window_entity_properties_add_slider(data, "Resolution", &sky.hdri_dim, ELEMENT_SLIDER_DATA_TYPE_UINT, 1.0f, FLT_MAX, 1.0f);
      update_data |=
        _window_entity_properties_add_slider(data, "Sample Count", &sky.hdri_samples, ELEMENT_SLIDER_DATA_TYPE_UINT, 0, FLT_MAX, 1.0f);
      if (_window_entity_properties_add_button(data, "Build")) {
        LUM_FAILURE_HANDLE(luminary_host_request_sky_hdri_build(host));
      }
    case LUMINARY_SKY_MODE_DEFAULT:
    default:
      element_separator(
        window, mouse_state,
        (ElementSeparatorArgs) {.text = "Atmosphere and Celestials", .size = (ElementSize) {.rel_width = 1.0f, .height = 32}});

      update_data |= _window_entity_properties_add_slider(
        data, "Geometry Offset", &sky.geometry_offset, ELEMENT_SLIDER_DATA_TYPE_VECTOR, -FLT_MAX, FLT_MAX, 1.0f);
      update_data |=
        _window_entity_properties_add_slider(data, "Azimuth", &sky.azimuth, ELEMENT_SLIDER_DATA_TYPE_FLOAT, -FLT_MAX, FLT_MAX, 1.0f);
      update_data |=
        _window_entity_properties_add_slider(data, "Altitude", &sky.altitude, ELEMENT_SLIDER_DATA_TYPE_FLOAT, -FLT_MAX, FLT_MAX, 1.0f);
      update_data |= _window_entity_properties_add_slider(
        data, "Moon Azimuth", &sky.moon_azimuth, ELEMENT_SLIDER_DATA_TYPE_FLOAT, -FLT_MAX, FLT_MAX, 1.0f);
      update_data |= _window_entity_properties_add_slider(
        data, "Moon Altitude", &sky.moon_altitude, ELEMENT_SLIDER_DATA_TYPE_FLOAT, -FLT_MAX, FLT_MAX, 1.0f);
      update_data |= _window_entity_properties_add_slider(
        data, "Moon Texture Offset", &sky.moon_tex_offset, ELEMENT_SLIDER_DATA_TYPE_FLOAT, -FLT_MAX, FLT_MAX, 1.0f);
      update_data |=
        _window_entity_properties_add_slider(data, "Sun Intensity", &sky.sun_strength, ELEMENT_SLIDER_DATA_TYPE_FLOAT, 0.0f, FLT_MAX, 1.0f);
      update_data |=
        _window_entity_properties_add_slider(data, "Density", &sky.base_density, ELEMENT_SLIDER_DATA_TYPE_FLOAT, 0.0f, FLT_MAX, 1.0f);

      update_data |=
        _window_entity_properties_add_slider(data, "Num Steps", &sky.steps, ELEMENT_SLIDER_DATA_TYPE_UINT, 0.0f, FLT_MAX, 1.0f);

      update_data |= _window_entity_properties_add_slider(
        data, "Rayleigh Density", &sky.rayleigh_density, ELEMENT_SLIDER_DATA_TYPE_FLOAT, 0.0f, FLT_MAX, 1.0f);
      update_data |= _window_entity_properties_add_slider(
        data, "Rayleigh Falloff", &sky.rayleigh_falloff, ELEMENT_SLIDER_DATA_TYPE_FLOAT, 0.0f, FLT_MAX, 1.0f);

      update_data |=
        _window_entity_properties_add_slider(data, "Mie Density", &sky.mie_density, ELEMENT_SLIDER_DATA_TYPE_FLOAT, 0.0f, FLT_MAX, 1.0f);
      update_data |=
        _window_entity_properties_add_slider(data, "Mie Falloff", &sky.mie_falloff, ELEMENT_SLIDER_DATA_TYPE_FLOAT, 0.0f, FLT_MAX, 1.0f);
      update_data |=
        _window_entity_properties_add_slider(data, "Mie Diameter", &sky.mie_diameter, ELEMENT_SLIDER_DATA_TYPE_FLOAT, 0.0f, FLT_MAX, 1.0f);

      update_data |= _window_entity_properties_add_checkbox(data, "Ozone Absorption", &sky.ozone_absorption);

      if (sky.ozone_absorption) {
        update_data |= _window_entity_properties_add_slider(
          data, "Ozone Density", &sky.ozone_density, ELEMENT_SLIDER_DATA_TYPE_FLOAT, 0.0f, FLT_MAX, 1.0f);
        update_data |= _window_entity_properties_add_slider(
          data, "Ozone Layer Thickness", &sky.ozone_layer_thickness, ELEMENT_SLIDER_DATA_TYPE_FLOAT, 0.0f, FLT_MAX, 1.0f);
      }

      update_data |= _window_entity_properties_add_slider(
        data, "Ground Visibility", &sky.ground_visibility, ELEMENT_SLIDER_DATA_TYPE_FLOAT, 0.0f, FLT_MAX, 1.0f);
      update_data |= _window_entity_properties_add_slider(
        data, "Multiscattering Factor", &sky.multiscattering_factor, ELEMENT_SLIDER_DATA_TYPE_FLOAT, 0.0f, FLT_MAX, 1.0f);

      update_data |=
        _window_entity_properties_add_slider(data, "Stars Seed", &sky.stars_seed, ELEMENT_SLIDER_DATA_TYPE_UINT, 0.0f, FLT_MAX, 1.0f);
      update_data |=
        _window_entity_properties_add_slider(data, "Stars Count", &sky.stars_count, ELEMENT_SLIDER_DATA_TYPE_UINT, 0.0f, FLT_MAX, 1.0f);
      update_data |= _window_entity_properties_add_slider(
        data, "Stars Intensity", &sky.stars_intensity, ELEMENT_SLIDER_DATA_TYPE_FLOAT, 0.0f, FLT_MAX, 1.0f);

      update_data |= _window_entity_properties_add_checkbox(data, "Aerial Perspective", &sky.aerial_perspective);
      break;
    case LUMINARY_SKY_MODE_CONSTANT_COLOR:
      update_data |=
        _window_entity_properties_add_slider(data, "Color", &sky.constant_color, ELEMENT_SLIDER_DATA_TYPE_RGB, 0.0f, FLT_MAX, 1.0f);
      break;
  }

  if (update_data) {
    sky.mode = (LuminarySkyMode) mode;

    LUM_FAILURE_HANDLE(luminary_host_set_sky(host, &sky));
  }
}

static bool _window_entity_properties_cloud_layer_action(
  Window* window, Display* display, LuminaryHost* host, LuminaryCloudLayer* layer, const MouseState* mouse_state) {
  MD_CHECK_NULL_ARGUMENT(window);
  MD_CHECK_NULL_ARGUMENT(display);
  MD_CHECK_NULL_ARGUMENT(host);
  MD_CHECK_NULL_ARGUMENT(layer);

  WindowEntityPropertiesPassingData data = {
    .window = window, .display = display, .mouse_state = mouse_state, .keyboard_state = display->keyboard_state};

  bool update_data = false;

  update_data |= _window_entity_properties_add_checkbox(data, "Layer Active", &layer->active);

  if (layer->active) {
    update_data |=
      _window_entity_properties_add_slider(data, "Height Max", &layer->height_max, ELEMENT_SLIDER_DATA_TYPE_FLOAT, 0.0f, FLT_MAX, 1.0f);
    update_data |=
      _window_entity_properties_add_slider(data, "Height Min", &layer->height_min, ELEMENT_SLIDER_DATA_TYPE_FLOAT, 0.0f, FLT_MAX, 1.0f);
    update_data |=
      _window_entity_properties_add_slider(data, "Coverage", &layer->coverage, ELEMENT_SLIDER_DATA_TYPE_FLOAT, 0.0f, FLT_MAX, 1.0f);
    update_data |=
      _window_entity_properties_add_slider(data, "Coverage Min", &layer->coverage_min, ELEMENT_SLIDER_DATA_TYPE_FLOAT, 0.0f, FLT_MAX, 1.0f);
    update_data |= _window_entity_properties_add_slider(data, "Type", &layer->type, ELEMENT_SLIDER_DATA_TYPE_FLOAT, 0.0f, FLT_MAX, 1.0f);
    update_data |=
      _window_entity_properties_add_slider(data, "Type Min", &layer->type_min, ELEMENT_SLIDER_DATA_TYPE_FLOAT, 0.0f, FLT_MAX, 1.0f);
    update_data |=
      _window_entity_properties_add_slider(data, "Wind Speed", &layer->wind_speed, ELEMENT_SLIDER_DATA_TYPE_FLOAT, 0.0f, FLT_MAX, 1.0f);
    update_data |=
      _window_entity_properties_add_slider(data, "Wind Angle", &layer->wind_angle, ELEMENT_SLIDER_DATA_TYPE_FLOAT, -FLT_MAX, FLT_MAX, 1.0f);
  }

  return update_data;
}

static void _window_entity_properties_cloud_action(Window* window, Display* display, LuminaryHost* host, const MouseState* mouse_state) {
  MD_CHECK_NULL_ARGUMENT(window);
  MD_CHECK_NULL_ARGUMENT(display);
  MD_CHECK_NULL_ARGUMENT(host);

  WindowEntityPropertiesPassingData data = {
    .window = window, .display = display, .mouse_state = mouse_state, .keyboard_state = display->keyboard_state};

  LuminaryCloud cloud;
  LUM_FAILURE_HANDLE(luminary_host_get_cloud(host, &cloud));

  element_separator(
    window, mouse_state, (ElementSeparatorArgs) {.text = "Procedural Clouds", .size = (ElementSize) {.rel_width = 1.0f, .height = 32}});

  bool update_data = false;

  update_data |= _window_entity_properties_add_checkbox(data, "Active", &cloud.active);

  if (cloud.active) {
    update_data |= _window_entity_properties_add_checkbox(data, "Atmosphere Scattering", &cloud.atmosphere_scattering);

    update_data |=
      _window_entity_properties_add_slider(data, "Offset X", &cloud.offset_x, ELEMENT_SLIDER_DATA_TYPE_FLOAT, -FLT_MAX, FLT_MAX, 1.0f);
    update_data |=
      _window_entity_properties_add_slider(data, "Offset Z", &cloud.offset_z, ELEMENT_SLIDER_DATA_TYPE_FLOAT, -FLT_MAX, FLT_MAX, 1.0f);
    update_data |=
      _window_entity_properties_add_slider(data, "Density", &cloud.density, ELEMENT_SLIDER_DATA_TYPE_FLOAT, 0.0f, FLT_MAX, 1.0f);
    update_data |= _window_entity_properties_add_slider(
      data, "Droplet Diameter", &cloud.droplet_diameter, ELEMENT_SLIDER_DATA_TYPE_FLOAT, 0.0f, FLT_MAX, 1.0f);
    update_data |= _window_entity_properties_add_slider(
      data, "Shape Scale", &cloud.noise_shape_scale, ELEMENT_SLIDER_DATA_TYPE_FLOAT, 0.0f, FLT_MAX, 1.0f);
    update_data |= _window_entity_properties_add_slider(
      data, "Detail Scale", &cloud.noise_detail_scale, ELEMENT_SLIDER_DATA_TYPE_FLOAT, 0.0f, FLT_MAX, 1.0f);
    update_data |= _window_entity_properties_add_slider(
      data, "Weather Scale", &cloud.noise_weather_scale, ELEMENT_SLIDER_DATA_TYPE_FLOAT, 0.0f, FLT_MAX, 1.0f);

    element_separator(
      window, mouse_state, (ElementSeparatorArgs) {.text = "Top Layer", .size = (ElementSize) {.rel_width = 1.0f, .height = 32}});
    update_data |= _window_entity_properties_cloud_layer_action(window, display, host, &cloud.top, mouse_state);

    element_separator(
      window, mouse_state, (ElementSeparatorArgs) {.text = "Mid Layer", .size = (ElementSize) {.rel_width = 1.0f, .height = 32}});
    update_data |= _window_entity_properties_cloud_layer_action(window, display, host, &cloud.mid, mouse_state);

    element_separator(
      window, mouse_state, (ElementSeparatorArgs) {.text = "Bottom Layer", .size = (ElementSize) {.rel_width = 1.0f, .height = 32}});
    update_data |= _window_entity_properties_cloud_layer_action(window, display, host, &cloud.low, mouse_state);
  }

  if (update_data) {
    LUM_FAILURE_HANDLE(luminary_host_set_cloud(host, &cloud));
  }
}

static void _window_entity_properties_fog_action(Window* window, Display* display, LuminaryHost* host, const MouseState* mouse_state) {
  MD_CHECK_NULL_ARGUMENT(window);
  MD_CHECK_NULL_ARGUMENT(display);
  MD_CHECK_NULL_ARGUMENT(host);

  WindowEntityPropertiesPassingData data = {
    .window = window, .display = display, .mouse_state = mouse_state, .keyboard_state = display->keyboard_state};

  LuminaryFog fog;
  LUM_FAILURE_HANDLE(luminary_host_get_fog(host, &fog));

  element_separator(window, mouse_state, (ElementSeparatorArgs) {.text = "Fog", .size = (ElementSize) {.rel_width = 1.0f, .height = 32}});

  bool update_data = false;

  update_data |= _window_entity_properties_add_checkbox(data, "Active", &fog.active);

  if (fog.active) {
    update_data |= _window_entity_properties_add_slider(data, "Density", &fog.density, ELEMENT_SLIDER_DATA_TYPE_FLOAT, 0.0f, FLT_MAX, 1.0f);
    update_data |=
      _window_entity_properties_add_slider(data, "Height", &fog.height, ELEMENT_SLIDER_DATA_TYPE_FLOAT, -FLT_MAX, FLT_MAX, 1.0f);
    update_data |= _window_entity_properties_add_slider(data, "Distance", &fog.dist, ELEMENT_SLIDER_DATA_TYPE_FLOAT, 0.0f, FLT_MAX, 1.0f);
    update_data |= _window_entity_properties_add_slider(
      data, "Droplet Diameter", &fog.droplet_diameter, ELEMENT_SLIDER_DATA_TYPE_FLOAT, 0.0f, FLT_MAX, 1.0f);
  }

  if (update_data) {
    LUM_FAILURE_HANDLE(luminary_host_set_fog(host, &fog));
  }
}

static void _window_entity_properties_particles_action(
  Window* window, Display* display, LuminaryHost* host, const MouseState* mouse_state) {
  MD_CHECK_NULL_ARGUMENT(window);
  MD_CHECK_NULL_ARGUMENT(display);
  MD_CHECK_NULL_ARGUMENT(host);

  WindowEntityPropertiesPassingData data = {
    .window = window, .display = display, .mouse_state = mouse_state, .keyboard_state = display->keyboard_state};

  LuminaryParticles particles;
  LUM_FAILURE_HANDLE(luminary_host_get_particles(host, &particles));

  element_separator(
    window, mouse_state, (ElementSeparatorArgs) {.text = "Procedural Particles", .size = (ElementSize) {.rel_width = 1.0f, .height = 32}});

  bool update_data = false;

  update_data |= _window_entity_properties_add_checkbox(data, "Active", &particles.active);

  if (particles.active) {
    update_data |=
      _window_entity_properties_add_slider(data, "Count", &particles.count, ELEMENT_SLIDER_DATA_TYPE_UINT, 0.0f, 32767.0f, 1.0f);
    update_data |= _window_entity_properties_add_slider(data, "Seed", &particles.seed, ELEMENT_SLIDER_DATA_TYPE_UINT, 0.0f, FLT_MAX, 1.0f);
    update_data |= _window_entity_properties_add_slider(data, "Albedo", &particles.albedo, ELEMENT_SLIDER_DATA_TYPE_RGB, 0.0f, 1.0f, 1.0f);
    update_data |=
      _window_entity_properties_add_slider(data, "Speed", &particles.speed, ELEMENT_SLIDER_DATA_TYPE_FLOAT, 0.0f, FLT_MAX, 1.0f);
    update_data |= _window_entity_properties_add_slider(
      data, "Direction Altitude", &particles.direction_altitude, ELEMENT_SLIDER_DATA_TYPE_FLOAT, -FLT_MAX, FLT_MAX, 1.0f);
    update_data |= _window_entity_properties_add_slider(
      data, "Direction Azimuth", &particles.direction_azimuth, ELEMENT_SLIDER_DATA_TYPE_FLOAT, -FLT_MAX, FLT_MAX, 1.0f);
    update_data |= _window_entity_properties_add_slider(
      data, "Phase Diameter", &particles.phase_diameter, ELEMENT_SLIDER_DATA_TYPE_FLOAT, 0.0f, FLT_MAX, 1.0f);
    update_data |=
      _window_entity_properties_add_slider(data, "Scale", &particles.scale, ELEMENT_SLIDER_DATA_TYPE_FLOAT, 0.0f, FLT_MAX, 1.0f);
    update_data |= _window_entity_properties_add_slider(data, "Size", &particles.size, ELEMENT_SLIDER_DATA_TYPE_FLOAT, 0.0f, FLT_MAX, 1.0f);
    update_data |= _window_entity_properties_add_slider(
      data, "Size Variation", &particles.size_variation, ELEMENT_SLIDER_DATA_TYPE_FLOAT, 0.0f, FLT_MAX, 1.0f);
  }

  if (update_data) {
    LUM_FAILURE_HANDLE(luminary_host_set_particles(host, &particles));
  }
}

static void _window_entity_properties_material_action(Window* window, Display* display, LuminaryHost* host, const MouseState* mouse_state) {
  MD_CHECK_NULL_ARGUMENT(window);
  MD_CHECK_NULL_ARGUMENT(display);
  MD_CHECK_NULL_ARGUMENT(host);

  WindowEntityPropertiesPassingData data = {
    .window = window, .display = display, .mouse_state = mouse_state, .keyboard_state = display->keyboard_state};

  element_separator(
    window, mouse_state, (ElementSeparatorArgs) {.text = "Material", .size = (ElementSize) {.rel_width = 1.0f, .height = 32}});

  const bool material_is_selected = display->select_pixel_data.pixel_query_is_valid && display->select_pixel_data.material_id != 0xFFFF;

  if (!material_is_selected) {
    element_text(
      window, display, mouse_state,
      (ElementTextArgs) {.color        = 0xFFFFFFFF,
                         .size         = (ElementSize) {.rel_width = 1.0f, .height = 24},
                         .text         = "No material is selected.",
                         .center_x     = true,
                         .center_y     = true,
                         .highlighting = false,
                         .cache_text   = true,
                         .auto_size    = false,
                         .is_clickable = false});

    return;
  }

  LuminaryMaterial material;
  LUM_FAILURE_HANDLE(luminary_host_get_material(host, display->select_pixel_data.material_id, &material));

  uint32_t base_substrate = (uint32_t) material.base_substrate;

  bool update_data = false;

  update_data |= _window_entity_properties_add_dropdown(
    data, "Base Substrate", LUMINARY_MATERIAL_BASE_SUBSTRATE_COUNT, (char**) luminary_strings_material_base_substrate, &base_substrate);

  if (material.albedo_tex == 0xFFFF) {
    update_data |= _window_entity_properties_add_slider(data, "Albedo", &material.albedo, ELEMENT_SLIDER_DATA_TYPE_RGB, 0.0f, 1.0f, 1.0f);
    update_data |=
      _window_entity_properties_add_slider(data, "Opacity", &material.albedo.a, ELEMENT_SLIDER_DATA_TYPE_FLOAT, 0.0f, 1.0f, 1.0f);
  }

  update_data |= _window_entity_properties_add_checkbox(data, "Emission Active", &material.emission_active);

  if (material.emission_active) {
    if (material.luminance_tex != 0xFFFF) {
      update_data |= _window_entity_properties_add_slider(
        data, "Emission Scale", &material.emission_scale, ELEMENT_SLIDER_DATA_TYPE_FLOAT, 0.0f, 1023.0f, 1.0f);
    }
    else {
      update_data |=
        _window_entity_properties_add_slider(data, "Emission", &material.emission, ELEMENT_SLIDER_DATA_TYPE_RGB, 0.0f, 1023.0f, 1.0f);
    }

    update_data |= _window_entity_properties_add_checkbox(data, "Bidirectional Emission", &material.bidirectional_emission);
  }

  if (material.roughness_tex == 0xFFFF) {
    update_data |=
      _window_entity_properties_add_slider(data, "Roughness", &material.roughness, ELEMENT_SLIDER_DATA_TYPE_FLOAT, 0.0f, 1.0f, 1.0f);
  }

  if ((LuminaryMaterialBaseSubstrate) base_substrate == LUMINARY_MATERIAL_BASE_SUBSTRATE_OPAQUE) {
    if (material.metallic_tex == 0xFFFF) {
      update_data |= _window_entity_properties_add_checkbox(data, "Metallic", &material.metallic);
    }
  }

  update_data |= _window_entity_properties_add_slider(
    data, "Roughness Clamp", &material.roughness_clamp, ELEMENT_SLIDER_DATA_TYPE_FLOAT, 0.0f, 1.0f, 1.0f);

  if ((LuminaryMaterialBaseSubstrate) base_substrate == LUMINARY_MATERIAL_BASE_SUBSTRATE_TRANSLUCENT) {
    update_data |=
      _window_entity_properties_add_slider(data, "IOR", &material.refraction_index, ELEMENT_SLIDER_DATA_TYPE_FLOAT, 1.0f, 3.0f, 1.0f);
  }

  if (material.normal_tex != 0xFFFF) {
    update_data |= _window_entity_properties_add_checkbox(data, "Compressed Normal Map", &material.normal_map_is_compressed);
  }

  update_data |= _window_entity_properties_add_checkbox(data, "Colored Transparency", &material.colored_transparency);
  update_data |= _window_entity_properties_add_checkbox(data, "Roughness as Smoothness", &material.roughness_as_smoothness);

  if (update_data) {
    material.base_substrate = (LuminaryMaterialBaseSubstrate) base_substrate;

    LUM_FAILURE_HANDLE(luminary_host_set_material(host, display->select_pixel_data.material_id, &material));
  }

  return;
}

static void _window_entity_properties_instance_action(Window* window, Display* display, LuminaryHost* host, const MouseState* mouse_state) {
  MD_CHECK_NULL_ARGUMENT(window);
  MD_CHECK_NULL_ARGUMENT(display);
  MD_CHECK_NULL_ARGUMENT(host);

  WindowEntityPropertiesPassingData data = {
    .window = window, .display = display, .mouse_state = mouse_state, .keyboard_state = display->keyboard_state};

  element_separator(
    window, mouse_state, (ElementSeparatorArgs) {.text = "Instance", .size = (ElementSize) {.rel_width = 1.0f, .height = 32}});

  const bool instance_is_selected = display->select_pixel_data.pixel_query_is_valid && display->select_pixel_data.instance_id != 0xFFFFFFFF;

  if (!instance_is_selected) {
    element_text(
      window, display, mouse_state,
      (ElementTextArgs) {.color        = 0xFFFFFFFF,
                         .size         = (ElementSize) {.rel_width = 1.0f, .height = 24},
                         .text         = "No instance is selected.",
                         .center_x     = true,
                         .center_y     = true,
                         .highlighting = false,
                         .cache_text   = true,
                         .auto_size    = false,
                         .is_clickable = false});

    return;
  }

  LuminaryInstance instance;
  LUM_FAILURE_HANDLE(luminary_host_get_instance(host, display->select_pixel_data.instance_id, &instance));

  bool update_data = false;

  update_data |=
    _window_entity_properties_add_slider(data, "Position", &instance.position, ELEMENT_SLIDER_DATA_TYPE_VECTOR, -FLT_MAX, FLT_MAX, 1.0f);
  update_data |=
    _window_entity_properties_add_slider(data, "Rotation", &instance.rotation, ELEMENT_SLIDER_DATA_TYPE_VECTOR, -FLT_MAX, FLT_MAX, 1.0f);
  update_data |=
    _window_entity_properties_add_slider(data, "Scale", &instance.scale, ELEMENT_SLIDER_DATA_TYPE_VECTOR, -FLT_MAX, FLT_MAX, 1.0f);

  if (update_data) {
    LUM_FAILURE_HANDLE(luminary_host_set_instance(host, &instance));
  }

  return;
}

static void (*const action_funcs[WINDOW_ENTITY_PROPERTIES_TYPE_COUNT])(
  Window* window, Display* display, LuminaryHost* host, const MouseState* mouse_state) = {
  [WINDOW_ENTITY_PROPERTIES_TYPE_SETTINGS]  = _window_entity_properties_renderer_settings_action,
  [WINDOW_ENTITY_PROPERTIES_TYPE_CAMERA]    = _window_entity_properties_camera_action,
  [WINDOW_ENTITY_PROPERTIES_TYPE_OCEAN]     = _window_entity_properties_ocean_action,
  [WINDOW_ENTITY_PROPERTIES_TYPE_SKY]       = _window_entity_properties_sky_action,
  [WINDOW_ENTITY_PROPERTIES_TYPE_CLOUD]     = _window_entity_properties_cloud_action,
  [WINDOW_ENTITY_PROPERTIES_TYPE_FOG]       = _window_entity_properties_fog_action,
  [WINDOW_ENTITY_PROPERTIES_TYPE_PARTICLES] = _window_entity_properties_particles_action,
  [WINDOW_ENTITY_PROPERTIES_TYPE_MATERIAL]  = _window_entity_properties_material_action,
  [WINDOW_ENTITY_PROPERTIES_TYPE_INSTANCE]  = _window_entity_properties_instance_action};

void window_entity_properties_create(Window** window, WindowEntityPropertiesType type) {
  MD_CHECK_NULL_ARGUMENT(window);

  if (type >= WINDOW_ENTITY_PROPERTIES_TYPE_COUNT) {
    crash_message("Invalid entity properties window type was passed.");
  }

  window_create(window);

  (*window)->type            = WINDOW_TYPE_ENTITY_PROPERTIES;
  (*window)->visibility_mask = WINDOW_VISIBILITY_UTILITIES;
  (*window)->x               = 128 + (((uint32_t) type) * 64);
  (*window)->y               = 128 + (((uint32_t) type) * 64);
  (*window)->width           = 512;
  (*window)->height          = 512;
  (*window)->padding         = 8;
  (*window)->is_horizontal   = false;
  (*window)->is_visible      = false;
  (*window)->is_movable      = true;
  (*window)->background      = true;
  (*window)->auto_size       = true;
  (*window)->action_func     = action_funcs[type];
  (*window)->fixed_depth     = false;

  window_create_subwindow(*window);

  window_allocate_memory(*window);
}

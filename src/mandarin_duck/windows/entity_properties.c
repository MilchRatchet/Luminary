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

static bool _window_entity_properties_renderer_settings_action(Window* window, Display* display, LuminaryHost* host) {
  MD_CHECK_NULL_ARGUMENT(window);
  MD_CHECK_NULL_ARGUMENT(display);
  MD_CHECK_NULL_ARGUMENT(host);

  LuminaryRendererSettings settings;
  LUM_FAILURE_HANDLE(luminary_host_get_settings(host, &settings));

  uint32_t shading_mode = (uint32_t) settings.shading_mode;

  element_text(
    window, display,
    (ElementTextArgs){
      .color        = 0xFFFFFFFF,
      .size         = (ElementSize){.width = ELEMENT_SIZE_INVALID, .rel_width = 1.0f, .height = 24},
      .text         = "Renderer Settings",
      .center_x     = true,
      .center_y     = true,
      .highlighting = false,
      .cache_text   = true});

  window_margin(window, 16);

  bool update_data = false;

  update_data |=
    _window_entity_properties_add_slider(window, display, "Width", &settings.width, ELEMENT_SLIDER_DATA_TYPE_UINT, 0.0f, FLT_MAX, 1.0f);
  update_data |=
    _window_entity_properties_add_slider(window, display, "Height", &settings.height, ELEMENT_SLIDER_DATA_TYPE_UINT, 0.0f, FLT_MAX, 1.0f);
  update_data |= _window_entity_properties_add_slider(
    window, display, "Max Depth", &settings.max_ray_depth, ELEMENT_SLIDER_DATA_TYPE_UINT, 0.0f, FLT_MAX, 1.0f);
  update_data |= _window_entity_properties_add_slider(
    window, display, "Num Light RIS Samples", &settings.light_num_ris_samples, ELEMENT_SLIDER_DATA_TYPE_UINT, 0.0f, FLT_MAX, 1.0f);
  update_data |= _window_entity_properties_add_slider(
    window, display, "Num Light Rays", &settings.light_num_rays, ELEMENT_SLIDER_DATA_TYPE_UINT, 0.0f, FLT_MAX, 1.0f);
  update_data |= _window_entity_properties_add_slider(
    window, display, "Bridges Max Num Vertices", &settings.bridge_max_num_vertices, ELEMENT_SLIDER_DATA_TYPE_UINT, 0.0f, FLT_MAX, 1.0f);
  update_data |= _window_entity_properties_add_slider(
    window, display, "Bridges Num RIS Samples", &settings.bridge_num_ris_samples, ELEMENT_SLIDER_DATA_TYPE_UINT, 0.0f, FLT_MAX, 1.0f);
  update_data |= _window_entity_properties_add_slider(
    window, display, "Undersampling", &settings.undersampling, ELEMENT_SLIDER_DATA_TYPE_UINT, 0.0f, FLT_MAX, 1.0f);
  update_data |= _window_entity_properties_add_dropdown(
    window, display, "Shading Mode", LUMINARY_SHADING_MODE_COUNT, (char**) luminary_strings_shading_mode, &shading_mode);

  if (update_data) {
    settings.shading_mode = (LuminaryShadingMode) shading_mode;

    LUM_FAILURE_HANDLE(luminary_host_set_settings(host, &settings));
  }

  return update_data;
}

static bool _window_entity_properties_camera_action(Window* window, Display* display, LuminaryHost* host) {
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

    update_data |= _window_entity_properties_add_slider(
      window, display, "Focal Length", &camera.focal_length, ELEMENT_SLIDER_DATA_TYPE_FLOAT, 0.0f, FLT_MAX, 1.0f);
  }

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

static bool _window_entity_properties_ocean_action(Window* window, Display* display, LuminaryHost* host) {
  MD_CHECK_NULL_ARGUMENT(window);
  MD_CHECK_NULL_ARGUMENT(display);
  MD_CHECK_NULL_ARGUMENT(host);

  LuminaryOcean ocean;
  LUM_FAILURE_HANDLE(luminary_host_get_ocean(host, &ocean));

  uint32_t water_type = (uint32_t) ocean.water_type;

  element_text(
    window, display,
    (ElementTextArgs){
      .color        = 0xFFFFFFFF,
      .size         = (ElementSize){.width = ELEMENT_SIZE_INVALID, .rel_width = 1.0f, .height = 24},
      .text         = "Ocean",
      .center_x     = true,
      .center_y     = true,
      .highlighting = false,
      .cache_text   = true});

  window_margin(window, 16);

  bool update_data = false;

  update_data |= _window_entity_properties_add_checkbox(window, display, "Active", &ocean.active);

  if (ocean.active) {
    update_data |= _window_entity_properties_add_slider(
      window, display, "Height", &ocean.height, ELEMENT_SLIDER_DATA_TYPE_FLOAT, -FLT_MAX, FLT_MAX, 1.0f);
    update_data |= _window_entity_properties_add_slider(
      window, display, "Amplitude", &ocean.amplitude, ELEMENT_SLIDER_DATA_TYPE_FLOAT, -FLT_MAX, FLT_MAX, 1.0f);
    update_data |= _window_entity_properties_add_slider(
      window, display, "Frequency", &ocean.frequency, ELEMENT_SLIDER_DATA_TYPE_FLOAT, -FLT_MAX, FLT_MAX, 1.0f);
    update_data |= _window_entity_properties_add_slider(
      window, display, "Choppyness", &ocean.choppyness, ELEMENT_SLIDER_DATA_TYPE_FLOAT, -FLT_MAX, FLT_MAX, 1.0f);
    update_data |= _window_entity_properties_add_slider(
      window, display, "IOR", &ocean.refractive_index, ELEMENT_SLIDER_DATA_TYPE_FLOAT, 1.0f, 3.0f, 1.0f);
    update_data |= _window_entity_properties_add_dropdown(
      window, display, "Jerlov Water Type", LUMINARY_JERLOV_WATER_TYPE_COUNT, (char**) luminary_strings_jerlov_water_type, &water_type);

    update_data |= _window_entity_properties_add_checkbox(window, display, "Caustics", &ocean.caustics_active);

    if (ocean.caustics_active) {
      update_data |= _window_entity_properties_add_slider(
        window, display, "Caustics RIS Samples", &ocean.caustics_ris_sample_count, ELEMENT_SLIDER_DATA_TYPE_UINT, 0.0f, FLT_MAX, 1.0f);
      update_data |= _window_entity_properties_add_slider(
        window, display, "Caustics Domain Scale", &ocean.caustics_domain_scale, ELEMENT_SLIDER_DATA_TYPE_FLOAT, 0.0f, FLT_MAX, 1.0f);
    }

    update_data |= _window_entity_properties_add_checkbox(window, display, "Multiscattering", &ocean.multiscattering);
    update_data |=
      _window_entity_properties_add_checkbox(window, display, "Triangle Light Contribution", &ocean.triangle_light_contribution);
  }

  if (update_data) {
    ocean.water_type = (LuminaryJerlovWaterType) water_type;

    LUM_FAILURE_HANDLE(luminary_host_set_ocean(host, &ocean));
  }

  return update_data;
}

static bool _window_entity_properties_sky_action(Window* window, Display* display, LuminaryHost* host) {
  MD_CHECK_NULL_ARGUMENT(window);
  MD_CHECK_NULL_ARGUMENT(display);
  MD_CHECK_NULL_ARGUMENT(host);

  LuminarySky sky;
  LUM_FAILURE_HANDLE(luminary_host_get_sky(host, &sky));

  uint32_t mode = (uint32_t) sky.mode;

  element_text(
    window, display,
    (ElementTextArgs){
      .color        = 0xFFFFFFFF,
      .size         = (ElementSize){.width = ELEMENT_SIZE_INVALID, .rel_width = 1.0f, .height = 24},
      .text         = "Sky",
      .center_x     = true,
      .center_y     = true,
      .highlighting = false,
      .cache_text   = true});

  window_margin(window, 16);

  bool update_data = false;

  update_data |= _window_entity_properties_add_slider(
    window, display, "Geometry Offset", &sky.geometry_offset, ELEMENT_SLIDER_DATA_TYPE_VECTOR, -FLT_MAX, FLT_MAX, 1.0f);
  update_data |=
    _window_entity_properties_add_slider(window, display, "Azimuth", &sky.azimuth, ELEMENT_SLIDER_DATA_TYPE_FLOAT, -FLT_MAX, FLT_MAX, 1.0f);
  update_data |= _window_entity_properties_add_slider(
    window, display, "Altitude", &sky.altitude, ELEMENT_SLIDER_DATA_TYPE_FLOAT, -FLT_MAX, FLT_MAX, 1.0f);
  update_data |= _window_entity_properties_add_slider(
    window, display, "Moon Azimuth", &sky.moon_azimuth, ELEMENT_SLIDER_DATA_TYPE_FLOAT, -FLT_MAX, FLT_MAX, 1.0f);
  update_data |= _window_entity_properties_add_slider(
    window, display, "Moon Altitude", &sky.moon_altitude, ELEMENT_SLIDER_DATA_TYPE_FLOAT, -FLT_MAX, FLT_MAX, 1.0f);
  update_data |= _window_entity_properties_add_slider(
    window, display, "Moon Texture Offset", &sky.moon_tex_offset, ELEMENT_SLIDER_DATA_TYPE_FLOAT, -FLT_MAX, FLT_MAX, 1.0f);
  update_data |= _window_entity_properties_add_slider(
    window, display, "Sun Intensity", &sky.sun_strength, ELEMENT_SLIDER_DATA_TYPE_FLOAT, 0.0f, FLT_MAX, 1.0f);
  update_data |= _window_entity_properties_add_slider(
    window, display, "Density", &sky.base_density, ELEMENT_SLIDER_DATA_TYPE_FLOAT, 0.0f, FLT_MAX, 1.0f);

  update_data |=
    _window_entity_properties_add_slider(window, display, "Num Steps", &sky.steps, ELEMENT_SLIDER_DATA_TYPE_UINT, 0.0f, FLT_MAX, 1.0f);

  update_data |= _window_entity_properties_add_slider(
    window, display, "Rayleigh Density", &sky.rayleigh_density, ELEMENT_SLIDER_DATA_TYPE_FLOAT, 0.0f, FLT_MAX, 1.0f);
  update_data |= _window_entity_properties_add_slider(
    window, display, "Rayleigh Falloff", &sky.rayleigh_falloff, ELEMENT_SLIDER_DATA_TYPE_FLOAT, 0.0f, FLT_MAX, 1.0f);

  update_data |= _window_entity_properties_add_slider(
    window, display, "Mie Density", &sky.mie_density, ELEMENT_SLIDER_DATA_TYPE_FLOAT, 0.0f, FLT_MAX, 1.0f);
  update_data |= _window_entity_properties_add_slider(
    window, display, "Mie Falloff", &sky.mie_falloff, ELEMENT_SLIDER_DATA_TYPE_FLOAT, 0.0f, FLT_MAX, 1.0f);
  update_data |= _window_entity_properties_add_slider(
    window, display, "Mie Diameter", &sky.mie_diameter, ELEMENT_SLIDER_DATA_TYPE_FLOAT, 0.0f, FLT_MAX, 1.0f);

  update_data |= _window_entity_properties_add_checkbox(window, display, "Ozone Absorption", &sky.ozone_absorption);

  if (sky.ozone_absorption) {
    update_data |= _window_entity_properties_add_slider(
      window, display, "Ozone Density", &sky.ozone_density, ELEMENT_SLIDER_DATA_TYPE_FLOAT, 0.0f, FLT_MAX, 1.0f);
    update_data |= _window_entity_properties_add_slider(
      window, display, "Ozone Layer Thickness", &sky.ozone_layer_thickness, ELEMENT_SLIDER_DATA_TYPE_FLOAT, 0.0f, FLT_MAX, 1.0f);
  }

  update_data |= _window_entity_properties_add_slider(
    window, display, "Ground Visibility", &sky.base_density, ELEMENT_SLIDER_DATA_TYPE_FLOAT, 0.0f, FLT_MAX, 1.0f);
  update_data |= _window_entity_properties_add_slider(
    window, display, "Multiscattering Factor", &sky.base_density, ELEMENT_SLIDER_DATA_TYPE_FLOAT, 0.0f, FLT_MAX, 1.0f);

  update_data |= _window_entity_properties_add_checkbox(window, display, "Aerial Perspective", &sky.aerial_perspective);
  update_data |= _window_entity_properties_add_checkbox(window, display, "Ambient Sampling", &sky.ambient_sampling);

  update_data |=
    _window_entity_properties_add_dropdown(window, display, "Mode", LUMINARY_SKY_MODE_COUNT, (char**) luminary_strings_sky_mode, &mode);

  if ((LuminarySkyMode) mode == LUMINARY_SKY_MODE_CONSTANT_COLOR) {
    update_data |= _window_entity_properties_add_slider(
      window, display, "Color", &sky.constant_color, ELEMENT_SLIDER_DATA_TYPE_RGB, 0.0f, FLT_MAX, 1.0f);
  }

  if (update_data) {
    sky.mode = (LuminarySkyMode) mode;

    LUM_FAILURE_HANDLE(luminary_host_set_sky(host, &sky));
  }

  return update_data;
}

static bool _window_entity_properties_cloud_layer_action(Window* window, Display* display, LuminaryHost* host, LuminaryCloudLayer* layer) {
  MD_CHECK_NULL_ARGUMENT(window);
  MD_CHECK_NULL_ARGUMENT(display);
  MD_CHECK_NULL_ARGUMENT(host);
  MD_CHECK_NULL_ARGUMENT(layer);

  bool update_data = false;

  update_data |= _window_entity_properties_add_checkbox(window, display, "Layer Active", &layer->active);

  if (layer->active) {
    update_data |= _window_entity_properties_add_slider(
      window, display, "Height Max", &layer->height_max, ELEMENT_SLIDER_DATA_TYPE_FLOAT, 0.0f, FLT_MAX, 1.0f);
    update_data |= _window_entity_properties_add_slider(
      window, display, "Height Min", &layer->height_min, ELEMENT_SLIDER_DATA_TYPE_FLOAT, 0.0f, FLT_MAX, 1.0f);
    update_data |= _window_entity_properties_add_slider(
      window, display, "Coverage", &layer->coverage, ELEMENT_SLIDER_DATA_TYPE_FLOAT, 0.0f, FLT_MAX, 1.0f);
    update_data |= _window_entity_properties_add_slider(
      window, display, "Coverage Min", &layer->coverage_min, ELEMENT_SLIDER_DATA_TYPE_FLOAT, 0.0f, FLT_MAX, 1.0f);
    update_data |=
      _window_entity_properties_add_slider(window, display, "Type", &layer->type, ELEMENT_SLIDER_DATA_TYPE_FLOAT, 0.0f, FLT_MAX, 1.0f);
    update_data |= _window_entity_properties_add_slider(
      window, display, "Type Min", &layer->type_min, ELEMENT_SLIDER_DATA_TYPE_FLOAT, 0.0f, FLT_MAX, 1.0f);
    update_data |= _window_entity_properties_add_slider(
      window, display, "Wind Speed", &layer->wind_speed, ELEMENT_SLIDER_DATA_TYPE_FLOAT, 0.0f, FLT_MAX, 1.0f);
    update_data |= _window_entity_properties_add_slider(
      window, display, "Wind Angle", &layer->wind_angle, ELEMENT_SLIDER_DATA_TYPE_FLOAT, -FLT_MAX, FLT_MAX, 1.0f);
  }

  return update_data;
}

static bool _window_entity_properties_cloud_action(Window* window, Display* display, LuminaryHost* host) {
  MD_CHECK_NULL_ARGUMENT(window);
  MD_CHECK_NULL_ARGUMENT(display);
  MD_CHECK_NULL_ARGUMENT(host);

  LuminaryCloud cloud;
  LUM_FAILURE_HANDLE(luminary_host_get_cloud(host, &cloud));

  element_text(
    window, display,
    (ElementTextArgs){
      .color        = 0xFFFFFFFF,
      .size         = (ElementSize){.width = ELEMENT_SIZE_INVALID, .rel_width = 1.0f, .height = 24},
      .text         = "Procedural Clouds",
      .center_x     = true,
      .center_y     = true,
      .highlighting = false,
      .cache_text   = true});

  window_margin(window, 16);

  bool update_data = false;

  update_data |= _window_entity_properties_add_checkbox(window, display, "Active", &cloud.active);

  if (cloud.active) {
    update_data |= _window_entity_properties_add_checkbox(window, display, "Atmosphere Scattering", &cloud.atmosphere_scattering);

    update_data |= _window_entity_properties_add_slider(
      window, display, "Offset X", &cloud.offset_x, ELEMENT_SLIDER_DATA_TYPE_FLOAT, -FLT_MAX, FLT_MAX, 1.0f);
    update_data |= _window_entity_properties_add_slider(
      window, display, "Offset Z", &cloud.offset_z, ELEMENT_SLIDER_DATA_TYPE_FLOAT, -FLT_MAX, FLT_MAX, 1.0f);
    update_data |=
      _window_entity_properties_add_slider(window, display, "Density", &cloud.density, ELEMENT_SLIDER_DATA_TYPE_FLOAT, 0.0f, FLT_MAX, 1.0f);
    update_data |= _window_entity_properties_add_slider(
      window, display, "Droplet Diameter", &cloud.droplet_diameter, ELEMENT_SLIDER_DATA_TYPE_FLOAT, 0.0f, FLT_MAX, 1.0f);
    update_data |= _window_entity_properties_add_slider(
      window, display, "Shape Scale", &cloud.noise_shape_scale, ELEMENT_SLIDER_DATA_TYPE_FLOAT, 0.0f, FLT_MAX, 1.0f);
    update_data |= _window_entity_properties_add_slider(
      window, display, "Detail Scale", &cloud.noise_detail_scale, ELEMENT_SLIDER_DATA_TYPE_FLOAT, 0.0f, FLT_MAX, 1.0f);
    update_data |= _window_entity_properties_add_slider(
      window, display, "Weather Scale", &cloud.noise_weather_scale, ELEMENT_SLIDER_DATA_TYPE_FLOAT, 0.0f, FLT_MAX, 1.0f);

    // TODO: Add separator with title specifying which layer we are talking about.
    update_data |= _window_entity_properties_cloud_layer_action(window, display, host, &cloud.top);
    update_data |= _window_entity_properties_cloud_layer_action(window, display, host, &cloud.mid);
    update_data |= _window_entity_properties_cloud_layer_action(window, display, host, &cloud.low);
  }

  if (update_data) {
    LUM_FAILURE_HANDLE(luminary_host_set_cloud(host, &cloud));
  }

  return update_data;
}

static bool _window_entity_properties_fog_action(Window* window, Display* display, LuminaryHost* host) {
  MD_CHECK_NULL_ARGUMENT(window);
  MD_CHECK_NULL_ARGUMENT(display);
  MD_CHECK_NULL_ARGUMENT(host);

  LuminaryFog fog;
  LUM_FAILURE_HANDLE(luminary_host_get_fog(host, &fog));

  element_text(
    window, display,
    (ElementTextArgs){
      .color        = 0xFFFFFFFF,
      .size         = (ElementSize){.width = ELEMENT_SIZE_INVALID, .rel_width = 1.0f, .height = 24},
      .text         = "Fog",
      .center_x     = true,
      .center_y     = true,
      .highlighting = false,
      .cache_text   = true});

  window_margin(window, 16);

  bool update_data = false;

  update_data |= _window_entity_properties_add_checkbox(window, display, "Active", &fog.active);

  if (fog.active) {
    update_data |=
      _window_entity_properties_add_slider(window, display, "Density", &fog.density, ELEMENT_SLIDER_DATA_TYPE_FLOAT, 0.0f, FLT_MAX, 1.0f);
    update_data |=
      _window_entity_properties_add_slider(window, display, "Height", &fog.height, ELEMENT_SLIDER_DATA_TYPE_FLOAT, -FLT_MAX, FLT_MAX, 1.0f);
    update_data |=
      _window_entity_properties_add_slider(window, display, "Distance", &fog.dist, ELEMENT_SLIDER_DATA_TYPE_FLOAT, 0.0f, FLT_MAX, 1.0f);
    update_data |= _window_entity_properties_add_slider(
      window, display, "Droplet Diameter", &fog.droplet_diameter, ELEMENT_SLIDER_DATA_TYPE_FLOAT, 0.0f, FLT_MAX, 1.0f);
  }

  if (update_data) {
    LUM_FAILURE_HANDLE(luminary_host_set_fog(host, &fog));
  }

  return update_data;
}

static bool _window_entity_properties_particles_action(Window* window, Display* display, LuminaryHost* host) {
  MD_CHECK_NULL_ARGUMENT(window);
  MD_CHECK_NULL_ARGUMENT(display);
  MD_CHECK_NULL_ARGUMENT(host);

  LuminaryParticles particles;
  LUM_FAILURE_HANDLE(luminary_host_get_particles(host, &particles));

  element_text(
    window, display,
    (ElementTextArgs){
      .color        = 0xFFFFFFFF,
      .size         = (ElementSize){.width = ELEMENT_SIZE_INVALID, .rel_width = 1.0f, .height = 24},
      .text         = "Procedural Particles",
      .center_x     = true,
      .center_y     = true,
      .highlighting = false,
      .cache_text   = true});

  window_margin(window, 16);

  bool update_data = false;

  update_data |= _window_entity_properties_add_checkbox(window, display, "Active", &particles.active);

  if (particles.active) {
    update_data |=
      _window_entity_properties_add_slider(window, display, "Albedo", &particles.albedo, ELEMENT_SLIDER_DATA_TYPE_RGB, 0.0f, 1.0f, 1.0f);
    update_data |=
      _window_entity_properties_add_slider(window, display, "Speed", &particles.speed, ELEMENT_SLIDER_DATA_TYPE_FLOAT, 0.0f, FLT_MAX, 1.0f);
    update_data |= _window_entity_properties_add_slider(
      window, display, "Direction Altitude", &particles.direction_altitude, ELEMENT_SLIDER_DATA_TYPE_FLOAT, -FLT_MAX, FLT_MAX, 1.0f);
    update_data |= _window_entity_properties_add_slider(
      window, display, "Direction Azimuth", &particles.direction_azimuth, ELEMENT_SLIDER_DATA_TYPE_FLOAT, -FLT_MAX, FLT_MAX, 1.0f);
    update_data |= _window_entity_properties_add_slider(
      window, display, "Phase Diameter", &particles.phase_diameter, ELEMENT_SLIDER_DATA_TYPE_FLOAT, 0.0f, FLT_MAX, 1.0f);
    update_data |=
      _window_entity_properties_add_slider(window, display, "Scale", &particles.scale, ELEMENT_SLIDER_DATA_TYPE_FLOAT, 0.0f, FLT_MAX, 1.0f);
    update_data |=
      _window_entity_properties_add_slider(window, display, "Size", &particles.size, ELEMENT_SLIDER_DATA_TYPE_FLOAT, 0.0f, FLT_MAX, 1.0f);
    update_data |= _window_entity_properties_add_slider(
      window, display, "Size Variation", &particles.size_variation, ELEMENT_SLIDER_DATA_TYPE_FLOAT, 0.0f, FLT_MAX, 1.0f);
  }

  if (update_data) {
    LUM_FAILURE_HANDLE(luminary_host_set_particles(host, &particles));
  }

  return update_data;
}

static bool _window_entity_properties_material_action(Window* window, Display* display, LuminaryHost* host) {
  MD_CHECK_NULL_ARGUMENT(window);
  MD_CHECK_NULL_ARGUMENT(display);
  MD_CHECK_NULL_ARGUMENT(host);

  element_text(
    window, display,
    (ElementTextArgs){
      .color        = 0xFFFFFFFF,
      .size         = (ElementSize){.width = ELEMENT_SIZE_INVALID, .rel_width = 1.0f, .height = 24},
      .text         = "Material",
      .center_x     = true,
      .center_y     = true,
      .highlighting = false,
      .cache_text   = true});

  window_margin(window, 16);

  // TODO: If there is a selected material.
  if (true) {
    element_text(
      window, display,
      (ElementTextArgs){
        .color        = 0xFFFFFFFF,
        .size         = (ElementSize){.width = ELEMENT_SIZE_INVALID, .rel_width = 1.0f, .height = 24},
        .text         = "No material is selected.",
        .center_x     = true,
        .center_y     = true,
        .highlighting = false,
        .cache_text   = true});

    return false;
  }

  LuminaryMaterial material;

  bool emission_active = material.flags.emission_active != 0;

  bool update_data = false;

  update_data |=
    _window_entity_properties_add_slider(window, display, "Albedo", &material.albedo, ELEMENT_SLIDER_DATA_TYPE_RGB, 0.0f, 1.0f, 1.0f);
  update_data |=
    _window_entity_properties_add_slider(window, display, "Opacity", &material.albedo.a, ELEMENT_SLIDER_DATA_TYPE_FLOAT, 0.0f, 1.0f, 1.0f);

  update_data |= _window_entity_properties_add_checkbox(window, display, "Emission Active", &emission_active);

  if (emission_active) {
    if (material.luminance_tex != 0xFFFF) {
      update_data |= _window_entity_properties_add_slider(
        window, display, "Emission Scale", &material.emission_scale, ELEMENT_SLIDER_DATA_TYPE_FLOAT, 0.0f, FLT_MAX, 1.0f);
    }
    else {
      update_data |= _window_entity_properties_add_slider(
        window, display, "Emission", &material.emission, ELEMENT_SLIDER_DATA_TYPE_RGB, 0.0f, FLT_MAX, 1.0f);
    }
  }

  update_data |=
    _window_entity_properties_add_slider(window, display, "Metallic", &material.metallic, ELEMENT_SLIDER_DATA_TYPE_FLOAT, 0.0f, 1.0f, 1.0f);
  update_data |= _window_entity_properties_add_slider(
    window, display, "Roughness", &material.roughness, ELEMENT_SLIDER_DATA_TYPE_FLOAT, 0.0f, 1.0f, 1.0f);
  update_data |= _window_entity_properties_add_slider(
    window, display, "Roughness Clamp", &material.roughness_clamp, ELEMENT_SLIDER_DATA_TYPE_FLOAT, 0.0f, 1.0f, 1.0f);
  update_data |= _window_entity_properties_add_slider(
    window, display, "IOR", &material.refraction_index, ELEMENT_SLIDER_DATA_TYPE_FLOAT, 0.0f, 1.0f, 1.0f);

  material.flags.emission_active = (emission_active) ? 1 : 0;

  if (update_data) {
  }

  return update_data;
}

static bool (*const action_funcs[WINDOW_ENTITY_PROPERTIES_TYPE_COUNT])(Window* window, Display* display, LuminaryHost* host) = {
  [WINDOW_ENTITY_PROPERTIES_TYPE_SETTINGS]  = _window_entity_properties_renderer_settings_action,
  [WINDOW_ENTITY_PROPERTIES_TYPE_CAMERA]    = _window_entity_properties_camera_action,
  [WINDOW_ENTITY_PROPERTIES_TYPE_OCEAN]     = _window_entity_properties_ocean_action,
  [WINDOW_ENTITY_PROPERTIES_TYPE_SKY]       = _window_entity_properties_sky_action,
  [WINDOW_ENTITY_PROPERTIES_TYPE_CLOUD]     = _window_entity_properties_cloud_action,
  [WINDOW_ENTITY_PROPERTIES_TYPE_FOG]       = _window_entity_properties_fog_action,
  [WINDOW_ENTITY_PROPERTIES_TYPE_PARTICLES] = _window_entity_properties_particles_action,
  [WINDOW_ENTITY_PROPERTIES_TYPE_MATERIAL]  = _window_entity_properties_material_action};

void window_entity_properties_create(Window** window, WindowEntityPropertiesType type) {
  MD_CHECK_NULL_ARGUMENT(window);

  if (type >= WINDOW_ENTITY_PROPERTIES_TYPE_COUNT) {
    crash_message("Invalid entity properties window type was passed.");
  }

  window_create(window);

  (*window)->x             = 128 + (((uint32_t) type) * 64);
  (*window)->y             = 128 + (((uint32_t) type) * 64);
  (*window)->width         = 512;
  (*window)->height        = 512;
  (*window)->padding       = 8;
  (*window)->is_horizontal = false;
  (*window)->is_visible    = false;
  (*window)->is_movable    = true;
  (*window)->background    = true;
  (*window)->auto_size     = true;
  (*window)->action_func   = action_funcs[type];

  window_create_subwindow(*window);

  window_allocate_memory(*window);
}

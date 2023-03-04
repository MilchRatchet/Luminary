#include "UI.h"

#include <float.h>
#include <immintrin.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>

#include "UI_blit.h"
#include "UI_blur.h"
#include "UI_dropdown.h"
#include "UI_info.h"
#include "UI_panel.h"
#include "UI_text.h"
#include "baked.h"
#include "device.h"
#include "output.h"
#include "raytrace.h"
#include "scene.h"
#include "stars.h"

#define MOUSE_LEFT_BLOCKED 0b1
#define MOUSE_DRAGGING_WINDOW 0b10
#define MOUSE_DRAGGING_SLIDER 0b100

#define MOUSE_SCROLL_SPEED 10

#define TAB_PANEL_DEFAULT_ALLOCATION 128

static size_t compute_scratch_space() {
  size_t val = blur_scratch_needed();

  return val;
}

/*
 * Requires the font of the UI to be initialized.
 */
static UITab create_general_renderer_panels(UI* ui, RaytraceInstance* instance) {
  UITab tab;

  tab.count   = 1;
  tab.subtabs = (UITab*) 0;

  UIPanel* panels = (UIPanel*) malloc(sizeof(UIPanel) * TAB_PANEL_DEFAULT_ALLOCATION);

  int i = 0;

  panels[i++] = create_tab(ui, &(ui->tab), "General\nCamera\nSky\nOcean\nToy");
  panels[i++] = create_tab(ui, &(ui->subtab), "Renderer\nMaterials\nExport");
  panels[i++] = create_slider(ui, "Width", &(instance->settings.width), 0, 0.9f, 16.0f, 16384.0f, 0, 1);
  panels[i++] = create_slider(ui, "Height", &(instance->settings.height), 0, 0.9f, 16.0f, 16384.0f, 0, 1);
  panels[i++] = create_slider(ui, "Max Ray Depth", &(instance->settings.max_ray_depth), 0, 0.02f, 0.0f, 1024.0f, 0, 1);
  panels[i++] = create_slider(ui, "Reservoir Size", &(instance->settings.reservoir_size), 0, 0.02f, 1.0f, 1024.0f, 0, 1);
  panels[i++] = create_dropdown(ui, "Optix Denoiser", &(instance->settings.denoiser), 0, 3, "Off\0On\0Upscaling 4x", 5);
  panels[i++] = create_button(ui, "Reset Renderer", instance, (void (*)(void*)) raytrace_reset, 1);
  panels[i++] = create_info(ui, "Triangle Count", &(instance->scene.triangles_length), PANEL_INFO_TYPE_INT32, PANEL_INFO_STATIC);
  panels[i++] = create_dropdown(
    ui, "Shading Mode", &(instance->shading_mode), 1, 7, "Default\0Albedo\0Depth\0Normal\0Trace Heatmap\0Wireframe\0Lights", 8);
  panels[i++] = create_dropdown(ui, "Accumulation Mode", &(instance->accum_mode), 1, 3, "Off\0Accumulation\0Reprojection", 9);
  panels[i++] = create_info(ui, "Temporal Frames", &(instance->temporal_frames), PANEL_INFO_TYPE_INT32, PANEL_INFO_DYNAMIC);
  panels[i++] = create_info(ui, "Light Source Count", &(instance->scene.triangle_lights_length), PANEL_INFO_TYPE_INT32, PANEL_INFO_STATIC);
  panels[i++] = create_slider(ui, "Spatial Resampling Samples", &(instance->spatial_samples), 0, 0.02f, 0.0f, 16.0f, 0, 1);
  panels[i++] = create_slider(ui, "Spatial Resampling Iterations", &(instance->spatial_iterations), 0, 0.02f, 0.0f, 16.0f, 0, 1);

  tab.panels      = panels;
  tab.panel_count = i;
  tab.panels      = safe_realloc(tab.panels, sizeof(UIPanel) * tab.panel_count);

  return tab;
}

static UITab create_general_material_panels(UI* ui, RaytraceInstance* instance) {
  UITab tab;

  tab.count   = 1;
  tab.subtabs = (UITab*) 0;

  UIPanel* panels = (UIPanel*) malloc(sizeof(UIPanel) * TAB_PANEL_DEFAULT_ALLOCATION);

  int i = 0;

  panels[i++] = create_tab(ui, &(ui->tab), "General\nCamera\nSky\nOcean\nToy");
  panels[i++] = create_tab(ui, &(ui->subtab), "Renderer\nMaterials\nExport");
  panels[i++] = create_check(ui, "Lights", &(instance->scene.material.lights_active), 1);
  panels[i++] = create_slider(ui, "Alpha Cutoff", &(instance->scene.material.alpha_cutoff), 1, 0.0005f, 0.0f, 1.0f, 0, 0);
  panels[i++] = create_slider(ui, "Default Smoothness", &(instance->scene.material.default_material.r), 1, 0.001f, 0.0f, 1.0f, 0, 0);
  panels[i++] = create_slider(ui, "Default Metallic", &(instance->scene.material.default_material.g), 1, 0.001f, 0.0f, 1.0f, 0, 0);
  panels[i++] =
    create_slider(ui, "Default Light Intensity", &(instance->scene.material.default_material.b), 1, 0.001f, 0.0f, FLT_MAX, 0, 0);
  panels[i++] = create_dropdown(ui, "Fresnel Approximation", &(instance->scene.material.fresnel), 1, 2, "Schlick\0Fdez-Aguera", 7);

  tab.panels      = panels;
  tab.panel_count = i;
  tab.panels      = safe_realloc(tab.panels, sizeof(UIPanel) * tab.panel_count);

  return tab;
}

static UITab create_general_export_panels(UI* ui, RaytraceInstance* instance) {
  UITab tab;

  tab.count   = 1;
  tab.subtabs = (UITab*) 0;

  UIPanel* panels = (UIPanel*) malloc(sizeof(UIPanel) * TAB_PANEL_DEFAULT_ALLOCATION);

  int i = 0;

  panels[i++] = create_tab(ui, &(ui->tab), "General\nCamera\nSky\nOcean\nToy");
  panels[i++] = create_tab(ui, &(ui->subtab), "Renderer\nMaterials\nExport");
  panels[i++] = create_dropdown(ui, "Snapshot Resolution", &(instance->snap_resolution), 0, 2, "Window\0Render", 2);
  panels[i++] = create_dropdown(ui, "Output Image Format", &(instance->image_format), 0, 2, "PNG\0QOI", 3);
  panels[i++] = create_button(ui, "Export Settings", instance, (void (*)(void*)) scene_serialize, 0);
  panels[i++] = create_button(ui, "Export Baked File", instance, (void (*)(void*)) serialize_baked, 0);

  tab.panels      = panels;
  tab.panel_count = i;
  tab.panels      = safe_realloc(tab.panels, sizeof(UIPanel) * tab.panel_count);

  return tab;
}

static UITab create_general_panels(UI* ui, RaytraceInstance* instance) {
  UITab tab;

  tab.count       = 3;
  tab.panel_count = 0;
  tab.panels      = (UIPanel*) 0;

  UITab* tabs = (UITab*) malloc(sizeof(UITab) * tab.count);

  tabs[0] = create_general_renderer_panels(ui, instance);
  tabs[1] = create_general_material_panels(ui, instance);
  tabs[2] = create_general_export_panels(ui, instance);

  tab.subtabs = tabs;

  return tab;
}

static UITab create_camera_prop_panels(UI* ui, RaytraceInstance* instance) {
  UITab tab;

  tab.count   = 1;
  tab.subtabs = (UITab*) 0;

  UIPanel* panels = (UIPanel*) malloc(sizeof(UIPanel) * TAB_PANEL_DEFAULT_ALLOCATION);

  int i = 0;

  panels[i++] = create_tab(ui, &(ui->tab), "General\nCamera\nSky\nOcean\nToy");
  panels[i++] = create_tab(ui, &(ui->subtab), "Properties\nPost Processing");
  panels[i++] = create_slider(ui, "Position X", &(instance->scene.camera.pos.x), 1, 0.001f, -FLT_MAX, FLT_MAX, 1, 0);
  panels[i++] = create_slider(ui, "Position Y", &(instance->scene.camera.pos.y), 1, 0.001f, -FLT_MAX, FLT_MAX, 1, 0);
  panels[i++] = create_slider(ui, "Position Z", &(instance->scene.camera.pos.z), 1, 0.001f, -FLT_MAX, FLT_MAX, 1, 0);
  panels[i++] = create_slider(ui, "Rotation X", &(instance->scene.camera.rotation.x), 1, 0.001f, -FLT_MAX, FLT_MAX, 1, 0);
  panels[i++] = create_slider(ui, "Rotation Y", &(instance->scene.camera.rotation.y), 1, 0.001f, -FLT_MAX, FLT_MAX, 1, 0);
  panels[i++] = create_slider(ui, "Rotation Z", &(instance->scene.camera.rotation.z), 1, 0.001f, -FLT_MAX, FLT_MAX, 1, 0);
  panels[i++] = create_slider(ui, "Field of View", &(instance->scene.camera.fov), 1, 0.001f, 0.0001f, FLT_MAX, 0, 0);
  panels[i++] = create_slider(ui, "Aperture Size", &(instance->scene.camera.aperture_size), 1, 0.0005f, 0.0f, FLT_MAX, 0, 0);
  panels[i++] = create_slider(ui, "Focal Length", &(instance->scene.camera.focal_length), 1, 0.001f, 0.0f, FLT_MAX, 0, 0);
  panels[i++] = create_slider(ui, "Far Clip Distance", &(instance->scene.camera.far_clip_distance), 1, 0.05f, 0.0f, FLT_MAX, 0, 0);
  panels[i++] = create_slider(ui, "Camera Speed", &(instance->scene.camera.wasd_speed), 0, 0.0001f, 0.0f, FLT_MAX, 0, 0);
  panels[i++] = create_slider(ui, "Mouse Sensitivity", &(instance->scene.camera.mouse_speed), 0, 0.0001f, 0.0f, FLT_MAX, 0, 0);
  panels[i++] = create_check(ui, "Smooth Camera Movement", &(instance->scene.camera.smooth_movement), 0);
  panels[i++] = create_slider(ui, "Smoothing Factor", &(instance->scene.camera.smoothing_factor), 0, 0.0001f, 0.0f, 1.0f, 0, 0);
  panels[i++] =
    create_slider(ui, "Russian Roulette Bias", &(instance->scene.camera.russian_roulette_bias), 1, 0.0001f, 0.001f, FLT_MAX, 0, 0);

  tab.panels      = panels;
  tab.panel_count = i;
  tab.panels      = safe_realloc(tab.panels, sizeof(UIPanel) * tab.panel_count);

  return tab;
}

static UITab create_camera_post_panels(UI* ui, RaytraceInstance* instance) {
  UITab tab;

  tab.count   = 1;
  tab.subtabs = (UITab*) 0;

  UIPanel* panels = (UIPanel*) malloc(sizeof(UIPanel) * TAB_PANEL_DEFAULT_ALLOCATION);

  int i = 0;

  panels[i++] = create_tab(ui, &(ui->tab), "General\nCamera\nSky\nOcean\nToy");
  panels[i++] = create_tab(ui, &(ui->subtab), "Properties\nPost Processing");
  panels[i++] = create_dropdown(ui, "Tone Mapping", &(instance->scene.camera.tonemap), 0, 4, "None\0ACES\0Reinhard\0Uncharted 2", 2);
  panels[i++] =
    create_dropdown(ui, "Filter", &(instance->scene.camera.filter), 0, 7, "None\0Gray\0Sepia\0Gameboy\0002 Bit Gray\0CRT\0Black/White", 3);
  panels[i++] = create_check(ui, "Auto Exposure", &(instance->scene.camera.auto_exposure), 0);
  panels[i++] = create_slider(ui, "Exposure", &(instance->scene.camera.exposure), 0, 0.0005f, 0.0f, FLT_MAX, 1, 0);
  panels[i++] = create_check(ui, "Bloom", &(instance->scene.camera.bloom), 0);
  panels[i++] = create_slider(ui, "Bloom Strength", &(instance->scene.camera.bloom_strength), 0, 0.0005f, 0.0f, FLT_MAX, 0, 0);
  panels[i++] = create_slider(ui, "Bloom Threshold", &(instance->scene.camera.bloom_threshold), 0, 0.0005f, 0.0f, FLT_MAX, 0, 0);
  panels[i++] = create_check(ui, "Dithering", &(instance->scene.camera.dithering), 0);
  panels[i++] = create_slider(ui, "Temporal Blend Factor", &(instance->scene.camera.temporal_blend_factor), 1, 0.0005f, 0.0f, 1.0f, 0, 0);
  panels[i++] = create_check(ui, "Purkinje Shift", &(instance->scene.camera.purkinje), 0);
  panels[i++] = create_slider(ui, "Purkinje Blueness", &(instance->scene.camera.purkinje_kappa1), 0, 0.0001f, 0.0f, FLT_MAX, 0, 0);
  panels[i++] = create_slider(ui, "Purkinje Brightness", &(instance->scene.camera.purkinje_kappa2), 0, 0.0001f, 0.0f, FLT_MAX, 0, 0);

  tab.panels      = panels;
  tab.panel_count = i;
  tab.panels      = safe_realloc(tab.panels, sizeof(UIPanel) * tab.panel_count);

  return tab;
}

static UITab create_camera_panels(UI* ui, RaytraceInstance* instance) {
  UITab tab;

  tab.count       = 2;
  tab.panel_count = 0;
  tab.panels      = (UIPanel*) 0;

  UITab* tabs = (UITab*) malloc(sizeof(UITab) * tab.count);

  tabs[0] = create_camera_prop_panels(ui, instance);
  tabs[1] = create_camera_post_panels(ui, instance);

  tab.subtabs = tabs;

  return tab;
}

static UITab create_sky_celestial_panels(UI* ui, RaytraceInstance* instance) {
  UITab tab;

  tab.count   = 1;
  tab.subtabs = (UITab*) 0;

  UIPanel* panels = (UIPanel*) malloc(sizeof(UIPanel) * TAB_PANEL_DEFAULT_ALLOCATION);

  int i = 0;

  panels[i++] = create_tab(ui, &(ui->tab), "General\nCamera\nSky\nOcean\nToy");
  panels[i++] = create_tab(ui, &(ui->subtab), "Celestial\nAtmosphere\nClouds\nFog");
  panels[i++] = create_slider(ui, "Geometry Offset X", &(instance->scene.sky.geometry_offset.x), 1, 0.001f, -FLT_MAX, FLT_MAX, 1, 0);
  panels[i++] = create_slider(ui, "Geometry Offset Y", &(instance->scene.sky.geometry_offset.y), 1, 0.001f, -FLT_MAX, FLT_MAX, 1, 0);
  panels[i++] = create_slider(ui, "Geometry Offset Z", &(instance->scene.sky.geometry_offset.z), 1, 0.001f, -FLT_MAX, FLT_MAX, 1, 0);
  panels[i++] = create_slider(ui, "Sun Azimuth", &(instance->scene.sky.azimuth), 1, 0.0001f, -FLT_MAX, FLT_MAX, 1, 0);
  panels[i++] = create_slider(ui, "Sun Altitude", &(instance->scene.sky.altitude), 1, 0.0001f, -FLT_MAX, FLT_MAX, 1, 0);
  panels[i++] = create_slider(ui, "Sun Intensity", &(instance->scene.sky.sun_strength), 1, 0.001f, 0.0f, FLT_MAX, 0, 0);
  panels[i++] = create_slider(ui, "Moon Azimuth", &(instance->scene.sky.moon_azimuth), 1, 0.0001f, -FLT_MAX, FLT_MAX, 0, 0);
  panels[i++] = create_slider(ui, "Moon Altitude", &(instance->scene.sky.moon_altitude), 1, 0.0001f, -FLT_MAX, FLT_MAX, 0, 0);
  panels[i++] = create_slider(ui, "Moon Albedo", &(instance->scene.sky.moon_albedo), 1, 0.0001f, 0.0f, 1.0f, 0, 0);
  panels[i++] = create_slider(ui, "Stars Count", &(instance->scene.sky.settings_stars_count), 0, 1.0f, 0.0f, FLT_MAX, 0, 1);
  panels[i++] = create_slider(ui, "Stars Seed", &(instance->scene.sky.stars_seed), 0, 1.0f, 0.0f, FLT_MAX, 0, 1);
  panels[i++] = create_slider(ui, "Stars Intensity", &(instance->scene.sky.stars_intensity), 1, 0.0001f, 0.0f, FLT_MAX, 0, 0);
  panels[i++] = create_button(ui, "Generate Stars", instance, (void (*)(void*)) generate_stars, 1);

  tab.panels      = panels;
  tab.panel_count = i;
  tab.panels      = safe_realloc(tab.panels, sizeof(UIPanel) * tab.panel_count);

  return tab;
}

static UITab create_sky_atmo_panels(UI* ui, RaytraceInstance* instance) {
  UITab tab;

  tab.count   = 1;
  tab.subtabs = (UITab*) 0;

  UIPanel* panels = (UIPanel*) malloc(sizeof(UIPanel) * TAB_PANEL_DEFAULT_ALLOCATION);

  int i = 0;

  panels[i++] = create_tab(ui, &(ui->tab), "General\nCamera\nSky\nOcean\nToy");
  panels[i++] = create_tab(ui, &(ui->subtab), "Celestial\nAtmosphere\nClouds\nFog");
  panels[i++] = create_slider(ui, "Ray March Steps", &(instance->scene.sky.steps), 1, 0.005f, 0.0f, FLT_MAX, 0, 1);
  panels[i++] = create_check(ui, "Ozone Absorption", &(instance->atmo_settings.ozone_absorption), 0);
  panels[i++] = create_slider(ui, "Density", &(instance->atmo_settings.base_density), 0, 0.001f, 0.0001f, FLT_MAX, 0, 0);
  panels[i++] = create_slider(ui, "Density (Rayleigh)", &(instance->atmo_settings.rayleigh_density), 0, 0.001f, 0.0001f, FLT_MAX, 0, 0);
  panels[i++] = create_slider(ui, "Density (Mie)", &(instance->atmo_settings.mie_density), 0, 0.001f, 0.0001f, FLT_MAX, 0, 0);
  panels[i++] = create_slider(ui, "Density (Ozone)", &(instance->atmo_settings.ozone_density), 0, 0.001f, 0.0001f, FLT_MAX, 0, 0);
  panels[i++] =
    create_slider(ui, "Height Falloff (Rayleigh)", &(instance->atmo_settings.rayleigh_falloff), 0, 0.001f, 0.0001f, FLT_MAX, 0, 0);
  panels[i++] = create_slider(ui, "Height Falloff (Mie)", &(instance->atmo_settings.mie_falloff), 0, 0.001f, 0.0001f, FLT_MAX, 0, 0);
  panels[i++] = create_slider(ui, "Ground Visibility", &(instance->atmo_settings.ground_visibility), 0, 0.001f, 0.0001f, FLT_MAX, 0, 0);
  panels[i++] =
    create_slider(ui, "Ozone Layer Thickness", &(instance->atmo_settings.ozone_layer_thickness), 0, 0.001f, 0.01f, FLT_MAX, 0, 0);
  panels[i++] = create_slider(ui, "Mie Phase G", &(instance->atmo_settings.mie_g), 0, 0.001f, -1.0f, 1.0f, 0, 0);
  panels[i++] =
    create_slider(ui, "Multiscattering Factor", &(instance->atmo_settings.multiscattering_factor), 0, 0.001f, 0.01f, FLT_MAX, 0, 0);
  panels[i++] = create_button(ui, "Apply Settings", instance, (void (*)(void*)) device_sky_generate_LUTs, 1);

  tab.panels      = panels;
  tab.panel_count = i;
  tab.panels      = safe_realloc(tab.panels, sizeof(UIPanel) * tab.panel_count);

  return tab;
}

static UITab create_sky_cloud_panels(UI* ui, RaytraceInstance* instance) {
  UITab tab;

  tab.count   = 1;
  tab.subtabs = (UITab*) 0;

  UIPanel* panels = (UIPanel*) malloc(sizeof(UIPanel) * TAB_PANEL_DEFAULT_ALLOCATION);

  int i = 0;

  panels[i++] = create_tab(ui, &(ui->tab), "General\nCamera\nSky\nOcean\nToy");
  panels[i++] = create_tab(ui, &(ui->subtab), "Celestial\nAtmosphere\nClouds\nFog");
  panels[i++] = create_check(ui, "Active", &(instance->scene.sky.cloud.active), 1);
  panels[i++] = create_slider(ui, "Steps", &(instance->scene.sky.cloud.steps), 1, 0.01f, 0.0f, 128.0f, 0, 1);
  panels[i++] = create_slider(ui, "Mipmap Bias", &(instance->scene.sky.cloud.mipmap_bias), 1, 0.001f, -FLT_MAX, FLT_MAX, 0, 0);
  panels[i++] = create_slider(ui, "Offset X", &(instance->scene.sky.cloud.offset_x), 1, 0.001f, -FLT_MAX, FLT_MAX, 0, 0);
  panels[i++] = create_slider(ui, "Offset Z", &(instance->scene.sky.cloud.offset_z), 1, 0.001f, -FLT_MAX, FLT_MAX, 0, 0);
  panels[i++] = create_slider(ui, "Height Minimum", &(instance->scene.sky.cloud.height_min), 1, 0.001f, 0.0f, FLT_MAX, 0, 0);
  panels[i++] = create_slider(ui, "Height Maximum", &(instance->scene.sky.cloud.height_max), 1, 0.001f, 0.0f, FLT_MAX, 0, 0);
  panels[i++] = create_slider(ui, "Noise Shape Scale", &(instance->scene.sky.cloud.noise_shape_scale), 1, 0.01f, 0.0f, FLT_MAX, 0, 0);
  panels[i++] = create_slider(ui, "Noise Detail Scale", &(instance->scene.sky.cloud.noise_detail_scale), 1, 0.01f, 0.0f, FLT_MAX, 0, 0);
  panels[i++] = create_slider(ui, "Noise Weather Scale", &(instance->scene.sky.cloud.noise_weather_scale), 1, 0.01f, 0.0f, FLT_MAX, 0, 0);
  panels[i++] = create_slider(ui, "Noise Curl Scale", &(instance->scene.sky.cloud.noise_curl_scale), 1, 0.01f, 0.0f, FLT_MAX, 0, 0);
  panels[i++] = create_slider(ui, "Density", &(instance->scene.sky.cloud.density), 1, 0.001f, 0.0f, FLT_MAX, 0, 0);
  panels[i++] = create_slider(ui, "Coverage", &(instance->scene.sky.cloud.coverage), 1, 0.001f, 0.0f, FLT_MAX, 0, 0);
  panels[i++] = create_slider(ui, "Minimum Coverage", &(instance->scene.sky.cloud.coverage_min), 1, 0.001f, 0.0f, 1.0f, 0, 0);
  panels[i++] = create_slider(ui, "Type", &(instance->scene.sky.cloud.type), 1, 0.001f, 0.0f, FLT_MAX, 0, 0);
  panels[i++] = create_slider(ui, "Minimum Type", &(instance->scene.sky.cloud.type_min), 1, 0.001f, 0.0f, 1.0f, 0, 0);
  panels[i++] = create_slider(ui, "Anvil Overhang", &(instance->scene.sky.cloud.anvil), 1, 0.001f, 0.0f, FLT_MAX, 0, 0);
  panels[i++] = create_slider(ui, "Forward Scattering", &(instance->scene.sky.cloud.forward_scattering), 1, 0.001f, -1.0f, 1.0f, 0, 0);
  panels[i++] = create_slider(ui, "Backward Scattering", &(instance->scene.sky.cloud.backward_scattering), 1, 0.001f, -1.0f, 1.0f, 0, 0);
  panels[i++] = create_slider(ui, "Direction Lerp", &(instance->scene.sky.cloud.lobe_lerp), 1, 0.001f, 0.0f, 1.0f, 0, 0);
  panels[i++] = create_slider(ui, "Wetness", &(instance->scene.sky.cloud.wetness), 1, 0.001f, 0.0f, 1.0f, 0, 0);
  panels[i++] = create_slider(ui, "Beer Powder Effect", &(instance->scene.sky.cloud.powder), 1, 0.001f, 0.0f, 1.0f, 0, 0);
  panels[i++] = create_slider(ui, "Shadow Steps", &(instance->scene.sky.cloud.shadow_steps), 1, 0.01f, 0.0f, 128.0f, 0, 1);
  panels[i++] = create_slider(ui, "Seed", &(instance->scene.sky.cloud.seed), 0, 0.005f, 0.0f, FLT_MAX, 0, 1);
  panels[i++] = create_button(ui, "Generate Noise Maps", instance, (void (*)(void*)) device_cloud_noise_generate, 1);

  tab.panels      = panels;
  tab.panel_count = i;
  tab.panels      = safe_realloc(tab.panels, sizeof(UIPanel) * tab.panel_count);

  return tab;
}

static UITab create_sky_fog_panels(UI* ui, RaytraceInstance* instance) {
  UITab tab;

  tab.count   = 1;
  tab.subtabs = (UITab*) 0;

  UIPanel* panels = (UIPanel*) malloc(sizeof(UIPanel) * TAB_PANEL_DEFAULT_ALLOCATION);

  int i = 0;

  panels[i++] = create_tab(ui, &(ui->tab), "General\nCamera\nSky\nOcean\nToy");
  panels[i++] = create_tab(ui, &(ui->subtab), "Celestial\nAtmosphere\nClouds\nFog");
  panels[i++] = create_check(ui, "Active", &(instance->scene.fog.active), 1);
  panels[i++] = create_slider(ui, "Density", &(instance->scene.fog.density), 1, 0.001f, 0.001f, FLT_MAX, 0, 0);
  panels[i++] = create_slider(ui, "Anisotropy", &(instance->scene.fog.anisotropy), 1, 0.001f, -0.95f, 0.95f, 0, 0);
  panels[i++] = create_slider(ui, "Distance", &(instance->scene.fog.dist), 1, 0.005f, 0.0f, FLT_MAX, 0, 0);
  panels[i++] = create_slider(ui, "Height", &(instance->scene.fog.height), 1, 0.005f, 0.0f, FLT_MAX, 0, 0);

  tab.panels      = panels;
  tab.panel_count = i;
  tab.panels      = safe_realloc(tab.panels, sizeof(UIPanel) * tab.panel_count);

  return tab;
}

static UITab create_sky_panels(UI* ui, RaytraceInstance* instance) {
  UITab tab;

  tab.count       = 4;
  tab.panel_count = 0;
  tab.panels      = (UIPanel*) 0;

  UITab* tabs = (UITab*) malloc(sizeof(UITab) * tab.count);

  tabs[0] = create_sky_celestial_panels(ui, instance);
  tabs[1] = create_sky_atmo_panels(ui, instance);
  tabs[2] = create_sky_cloud_panels(ui, instance);
  tabs[3] = create_sky_fog_panels(ui, instance);

  tab.subtabs = tabs;

  return tab;
}

static UITab create_ocean_panels(UI* ui, RaytraceInstance* instance) {
  UITab tab;

  tab.count   = 1;
  tab.subtabs = (UITab*) 0;

  UIPanel* panels = (UIPanel*) malloc(sizeof(UIPanel) * TAB_PANEL_DEFAULT_ALLOCATION);

  int i = 0;

  panels[i++] = create_tab(ui, &(ui->tab), "General\nCamera\nSky\nOcean\nToy");
  panels[i++] = create_check(ui, "Active", &(instance->scene.ocean.active), 1);
  panels[i++] = create_slider(ui, "Height", &(instance->scene.ocean.height), 1, 0.005f, -FLT_MAX, FLT_MAX, 0, 0);
  panels[i++] = create_slider(ui, "Amplitude", &(instance->scene.ocean.amplitude), 1, 0.005f, 0.0f, FLT_MAX, 0, 0);
  panels[i++] = create_slider(ui, "Choppyness", &(instance->scene.ocean.choppyness), 1, 0.005f, 0.0f, FLT_MAX, 0, 0);
  panels[i++] = create_slider(ui, "Frequency", &(instance->scene.ocean.frequency), 1, 0.005f, 0.0f, FLT_MAX, 0, 0);
  panels[i++] = create_color(ui, "Albedo", (float*) &(instance->scene.ocean.albedo));
  panels[i++] = create_slider(ui, "  Red", &(instance->scene.ocean.albedo.r), 1, 0.001f, 0.0f, 1.0f, 0, 0);
  panels[i++] = create_slider(ui, "  Green", &(instance->scene.ocean.albedo.g), 1, 0.001f, 0.0f, 1.0f, 0, 0);
  panels[i++] = create_slider(ui, "  Blue", &(instance->scene.ocean.albedo.b), 1, 0.001f, 0.0f, 1.0f, 0, 0);
  panels[i++] = create_slider(ui, "Refractive Index", &(instance->scene.ocean.refractive_index), 1, 0.001f, 1.0f, FLT_MAX, 0, 0);
  panels[i++] = create_check(ui, "Emissive", &(instance->scene.ocean.emissive), 1);
  panels[i++] = create_slider(ui, "Alpha", &(instance->scene.ocean.albedo.a), 1, 0.001f, 0.0f, 1.0f, 0, 0);
  panels[i++] = create_check(ui, "Animated", &(instance->scene.ocean.update), 1);
  panels[i++] = create_slider(ui, "Speed", &(instance->scene.ocean.speed), 1, 0.005f, 0.0f, FLT_MAX, 0, 0);
  panels[i++] = create_slider(ui, "Pollution", &(instance->scene.ocean.pollution), 1, 0.005f, 0.0f, 1.0f, 0, 0);
  panels[i++] = create_color(ui, "Scattering", (float*) &(instance->scene.ocean.scattering));
  panels[i++] = create_slider(ui, "  Red", &(instance->scene.ocean.scattering.r), 1, 0.001f, 0.0f, 1.0f, 0, 0);
  panels[i++] = create_slider(ui, "  Green", &(instance->scene.ocean.scattering.g), 1, 0.001f, 0.0f, 1.0f, 0, 0);
  panels[i++] = create_slider(ui, "  Blue", &(instance->scene.ocean.scattering.b), 1, 0.001f, 0.0f, 1.0f, 0, 0);
  panels[i++] = create_slider(ui, "Absorption Strength", &(instance->scene.ocean.absorption_strength), 1, 0.005f, 0.001f, FLT_MAX, 0, 0);
  panels[i++] = create_color(ui, "Absorption", (float*) &(instance->scene.ocean.absorption));
  panels[i++] = create_slider(ui, "  Red", &(instance->scene.ocean.absorption.r), 1, 0.001f, 0.001f, 1.0f, 0, 0);
  panels[i++] = create_slider(ui, "  Green", &(instance->scene.ocean.absorption.g), 1, 0.001f, 0.001f, 1.0f, 0, 0);
  panels[i++] = create_slider(ui, "  Blue", &(instance->scene.ocean.absorption.b), 1, 0.001f, 0.001f, 1.0f, 0, 0);
  panels[i++] = create_slider(ui, "Anisotropy", &(instance->scene.ocean.anisotropy), 1, 0.005f, -1.0f, 1.0f, 0, 0);

  tab.panels      = panels;
  tab.panel_count = i;
  tab.panels      = safe_realloc(tab.panels, sizeof(UIPanel) * tab.panel_count);

  return tab;
}

static UITab create_toy_panels(UI* ui, RaytraceInstance* instance) {
  UITab tab;

  tab.count   = 1;
  tab.subtabs = (UITab*) 0;

  UIPanel* panels = (UIPanel*) malloc(sizeof(UIPanel) * TAB_PANEL_DEFAULT_ALLOCATION);

  int i = 0;

  panels[i++] = create_tab(ui, &(ui->tab), "General\nCamera\nSky\nOcean\nToy");
  panels[i++] = create_check(ui, "Active", &(instance->scene.toy.active), 1);
  panels[i++] = create_dropdown(ui, "Shape", &(instance->scene.toy.shape), 1, 1, "Sphere", 2);
  panels[i++] = create_button(ui, "Center at Camera", instance, (void (*)(void*)) raytrace_center_toy_at_camera, 1);
  panels[i++] = create_check(ui, "Flashlight Mode", &(instance->scene.toy.flashlight_mode), 1);
  panels[i++] = create_slider(ui, "Position X", &(instance->scene.toy.position.x), 1, 0.005f, -FLT_MAX, FLT_MAX, 1, 0);
  panels[i++] = create_slider(ui, "Position Y", &(instance->scene.toy.position.y), 1, 0.005f, -FLT_MAX, FLT_MAX, 1, 0);
  panels[i++] = create_slider(ui, "Position Z", &(instance->scene.toy.position.z), 1, 0.005f, -FLT_MAX, FLT_MAX, 1, 0);
  panels[i++] = create_slider(ui, "Rotation X", &(instance->scene.toy.rotation.x), 1, 0.005f, -FLT_MAX, FLT_MAX, 0, 0);
  panels[i++] = create_slider(ui, "Rotation Y", &(instance->scene.toy.rotation.y), 1, 0.005f, -FLT_MAX, FLT_MAX, 0, 0);
  panels[i++] = create_slider(ui, "Rotation Z", &(instance->scene.toy.rotation.z), 1, 0.005f, -FLT_MAX, FLT_MAX, 0, 0);
  panels[i++] = create_slider(ui, "Scale", &(instance->scene.toy.scale), 1, 0.005f, 0.0f, FLT_MAX, 0, 0);
  panels[i++] = create_color(ui, "Albedo", (float*) &(instance->scene.toy.albedo));
  panels[i++] = create_slider(ui, "  Red", &(instance->scene.toy.albedo.r), 1, 0.001f, 0.0f, 1.0f, 0, 0);
  panels[i++] = create_slider(ui, "  Green", &(instance->scene.toy.albedo.g), 1, 0.001f, 0.0f, 1.0f, 0, 0);
  panels[i++] = create_slider(ui, "  Blue", &(instance->scene.toy.albedo.b), 1, 0.001f, 0.0f, 1.0f, 0, 0);
  panels[i++] = create_slider(ui, "  Alpha", &(instance->scene.toy.albedo.a), 1, 0.001f, 0.0f, 1.0f, 0, 0);
  panels[i++] = create_slider(ui, "Smoothness", &(instance->scene.toy.material.r), 1, 0.001f, 0.0f, 1.0f, 0, 0);
  panels[i++] = create_slider(ui, "Metallic", &(instance->scene.toy.material.g), 1, 0.001f, 0.0f, 1.0f, 0, 0);
  panels[i++] = create_slider(ui, "Refractive Index", &(instance->scene.toy.refractive_index), 1, 0.001f, 1.0f, FLT_MAX, 0, 0);
  panels[i++] = create_check(ui, "Emissive", &(instance->scene.toy.emissive), 1);
  panels[i++] = create_color(ui, "Emission", (float*) &(instance->scene.toy.emission));
  panels[i++] = create_slider(ui, "  Red", &(instance->scene.toy.emission.r), 1, 0.001f, 0.0f, 1.0f, 0, 0);
  panels[i++] = create_slider(ui, "  Green", &(instance->scene.toy.emission.g), 1, 0.001f, 0.0f, 1.0f, 0, 0);
  panels[i++] = create_slider(ui, "  Blue", &(instance->scene.toy.emission.b), 1, 0.001f, 0.0f, 1.0f, 0, 0);
  panels[i++] = create_slider(ui, "Light Intensity", &(instance->scene.toy.material.b), 1, 0.001f, 0.0f, FLT_MAX, 0, 0);

  tab.panels      = panels;
  tab.panel_count = i;
  tab.panels      = safe_realloc(tab.panels, sizeof(UIPanel) * tab.panel_count);

  return tab;
}

UI init_UI(RaytraceInstance* instance, WindowInstance* window) {
  UI ui;
  ui.active = 0;
  ui.tab    = 0;
  ui.subtab = 0;

  ui.x     = 100;
  ui.y     = 100;
  ui.max_x = window->width;
  ui.max_y = window->height;

  ui.mouse_flags = 0;
  ui.scroll_pos  = 0;

  ui.panel_hover  = -1;
  ui.border_hover = 0;

  ui.last_panel = (UIPanel*) 0;
  ui.dropdown   = (UIPanel*) 0;

  ui.pixels      = (uint8_t*) malloc(sizeof(uint8_t) * UI_WIDTH * UI_HEIGHT_BUFFER * 4);
  ui.pixels_mask = (uint8_t*) malloc(sizeof(uint8_t) * UI_WIDTH * UI_HEIGHT_BUFFER * 4);

  ui.temporal_frames = &(instance->temporal_frames);

  size_t scratch_size = compute_scratch_space();
  ui.scratch          = malloc(scratch_size);
  init_text(&ui);

  ui.tabs      = malloc(sizeof(UITab) * UI_PANELS_TAB_COUNT);
  ui.tab_count = UI_PANELS_TAB_COUNT;

  ui.tabs[0] = create_general_panels(&ui, instance);
  ui.tabs[1] = create_camera_panels(&ui, instance);
  ui.tabs[2] = create_sky_panels(&ui, instance);
  ui.tabs[3] = create_ocean_panels(&ui, instance);
  ui.tabs[4] = create_toy_panels(&ui, instance);

  SDL_SetRelativeMouseMode(!ui.active);

  return ui;
}

static UITab create_post_process_menu_panels(UI* ui, RaytraceInstance* instance) {
  UITab tab;

  tab.count   = 1;
  tab.subtabs = (UITab*) 0;

  UIPanel* panels = (UIPanel*) malloc(sizeof(UIPanel) * TAB_PANEL_DEFAULT_ALLOCATION);

  int i = 0;

  panels[i++] = create_dropdown(ui, "Tone Mapping", &(instance->scene.camera.tonemap), 0, 4, "None\0ACES\0Reinhard\0Uncharted 2", 0);
  panels[i++] =
    create_dropdown(ui, "Filter", &(instance->scene.camera.filter), 0, 7, "None\0Gray\0Sepia\0Gameboy\0002 Bit Gray\0CRT\0Black/White", 1);
  panels[i++] = create_slider(ui, "Exposure", &(instance->scene.camera.exposure), 0, 0.0005f, 0.0f, FLT_MAX, 1, 0);
  panels[i++] = create_check(ui, "Bloom", &(instance->scene.camera.bloom), 0);
  panels[i++] = create_slider(ui, "Bloom Strength", &(instance->scene.camera.bloom_strength), 0, 0.0005f, 0.0f, FLT_MAX, 0, 0);
  panels[i++] = create_slider(ui, "Bloom Threshold", &(instance->scene.camera.bloom_threshold), 0, 0.0005f, 0.0f, FLT_MAX, 0, 0);
  panels[i++] = create_check(ui, "Dithering", &(instance->scene.camera.dithering), 0);
  panels[i++] = create_check(ui, "Purkinje Shift", &(instance->scene.camera.purkinje), 0);
  panels[i++] = create_slider(ui, "Purkinje Blueness", &(instance->scene.camera.purkinje_kappa1), 0, 0.0001f, 0.0f, FLT_MAX, 0, 0);
  panels[i++] = create_slider(ui, "Purkinje Brightness", &(instance->scene.camera.purkinje_kappa2), 0, 0.0001f, 0.0f, FLT_MAX, 0, 0);
  panels[i++] = create_button(ui, "Finish", (void*) instance, (void (*)(void*)) offline_exit_post_process_menu, 0);

  tab.panels      = panels;
  tab.panel_count = i;
  tab.panels      = safe_realloc(tab.panels, sizeof(UIPanel) * tab.panel_count);

  return tab;
}

UI init_post_process_UI(RaytraceInstance* instance, WindowInstance* window) {
  UI ui;
  ui.active = 1;
  ui.tab    = 0;
  ui.subtab = 0;

  ui.x     = 100;
  ui.y     = 100;
  ui.max_x = window->width;
  ui.max_y = window->height;

  ui.mouse_flags = 0;
  ui.scroll_pos  = 0;

  ui.panel_hover  = -1;
  ui.border_hover = 0;

  ui.last_panel = (UIPanel*) 0;
  ui.dropdown   = (UIPanel*) 0;

  ui.pixels      = (uint8_t*) malloc(sizeof(uint8_t) * UI_WIDTH * UI_HEIGHT_BUFFER * 4);
  ui.pixels_mask = (uint8_t*) malloc(sizeof(uint8_t) * UI_WIDTH * UI_HEIGHT_BUFFER * 4);

  ui.temporal_frames = &(instance->temporal_frames);

  size_t scratch_size = compute_scratch_space();
  ui.scratch          = malloc(scratch_size);
  init_text(&ui);

  ui.tabs      = malloc(sizeof(UITab));
  ui.tab_count = 1;

  ui.tabs[0] = create_post_process_menu_panels(&ui, instance);

  SDL_SetRelativeMouseMode(0);

  return ui;
}

void toggle_UI(UI* ui) {
  ui->active ^= 1;
  SDL_SetRelativeMouseMode(!ui->active);
}

void set_input_events_UI(UI* ui, int mouse_xrel, int mouse_wheel) {
  ui->mouse_xrel  = mouse_xrel;
  ui->mouse_wheel = mouse_wheel;
}

static void UI_clamp_position(UI* ui) {
  ui->x = max(0, min(ui->x, ui->max_x - UI_BORDER_SIZE));
  ui->y = max(0, min(ui->y, ui->max_y - UI_BORDER_SIZE));
}

void handle_mouse_UI(UI* ui) {
  if (!ui->active)
    return;

  int x, y;
  int d_x, d_y;

  uint32_t state = SDL_GetMouseState(&x, &y);
  SDL_GetRelativeMouseState(&d_x, &d_y);

  if (!(SDL_BUTTON_LMASK & state)) {
    ui->mouse_flags &= ~MOUSE_LEFT_BLOCKED;
    ui->mouse_flags &= ~MOUSE_DRAGGING_SLIDER;

    SDL_SetRelativeMouseMode(SDL_FALSE);
  }

  ui->scroll_pos -= MOUSE_SCROLL_SPEED * ui->mouse_wheel;

  UITab* active_tab = (ui->tabs[ui->tab].count == 1) ? ui->tabs + ui->tab : ui->tabs[ui->tab].subtabs + ui->subtab;

  int max_scroll = PANEL_HEIGHT * (active_tab->panel_count - UI_HEIGHT_IN_PANELS);

  clamp(max_scroll, 0, INT_MAX);
  clamp(ui->scroll_pos, 0, max_scroll);

  if (ui->mouse_flags & MOUSE_DRAGGING_WINDOW) {
    ui->x += d_x;
    ui->y += d_y;

    UI_clamp_position(ui);

    ui->mouse_flags |= MOUSE_LEFT_BLOCKED;
    ui->mouse_flags ^= MOUSE_DRAGGING_WINDOW;
  }

  for (int i = 0; i < active_tab->panel_count; i++) {
    active_tab->panels[i].hover = 0;
  }

  x -= ui->x;
  y -= ui->y;

  UIPanel* panel = (UIPanel*) 0;

  if (ui->mouse_flags & MOUSE_DRAGGING_SLIDER) {
    panel = ui->last_panel;
  }
  else if (ui->dropdown && get_intersection_dropdown(ui->dropdown, x, ui->scroll_pos + y)) {
    if (ui->mouse_flags & MOUSE_LEFT_BLOCKED) {
      state &= ~SDL_BUTTON_LMASK;
    }

    ui->dropdown->handle_mouse(ui, ui->dropdown, state, x, (ui->scroll_pos + y));
    ui->panel_hover = -1;
  }
  else if (x > 0 && x < UI_WIDTH && y > 0 && y < UI_HEIGHT + UI_BORDER_SIZE) {
    if (y < UI_BORDER_SIZE) {
      ui->panel_hover  = -1;
      ui->border_hover = 1;

      if (SDL_BUTTON_LMASK & state) {
        ui->mouse_flags |= MOUSE_DRAGGING_WINDOW;
      }
    }
    else {
      y -= UI_BORDER_SIZE;
      ui->panel_hover  = (y >= PANEL_HEIGHT) ? (ui->scroll_pos + y) / PANEL_HEIGHT : 0;
      ui->border_hover = 0;

      if (ui->mouse_flags & MOUSE_LEFT_BLOCKED) {
        state &= ~SDL_BUTTON_LMASK;
      }

      if (ui->panel_hover < active_tab->panel_count)
        panel = active_tab->panels + ui->panel_hover;
    }
  }
  else {
    ui->panel_hover  = -1;
    ui->border_hover = 0;
  }

  if (SDL_BUTTON_LMASK & state) {
    ui->mouse_flags |= MOUSE_LEFT_BLOCKED;
  }

  if (panel && panel->handle_mouse) {
    panel->handle_mouse(ui, panel, state, x, (ui->scroll_pos + y) % PANEL_HEIGHT);
    if (panel->type == PANEL_SLIDER) {
      ui->mouse_flags &= ~MOUSE_LEFT_BLOCKED;

      if (SDL_BUTTON_LMASK & state) {
        ui->mouse_flags |= MOUSE_DRAGGING_SLIDER;
      }
    }

    ui->last_panel = panel;
  }

  if (SDL_BUTTON_LMASK & state && ui->dropdown) {
    if (!panel || panel->type != PANEL_DROPDOWN) {
      ui->dropdown->prop2 = 0;
      ui->dropdown        = (UIPanel*) 0;
    }
  }
}

static void render_tab(UI* ui, UITab* tab) {
  const int first_panel = 1 + ui->scroll_pos / PANEL_HEIGHT;
  const int last_panel  = first_panel + UI_HEIGHT_IN_PANELS;
  int y                 = PANEL_HEIGHT;
  tab->panels[0].render(ui, tab->panels, 0);
  for (int i = first_panel; i < min(last_panel, tab->panel_count); i++) {
    if (tab->panels[i].render) {
      tab->panels[i].render(ui, tab->panels + i, y);
    }
    y += PANEL_HEIGHT;
  }
}

void render_UI(UI* ui) {
  if (!ui->active)
    return;

  memset(ui->pixels, 0, sizeof(uint8_t) * UI_WIDTH * UI_HEIGHT_BUFFER * 4);
  memset(ui->pixels_mask, 0, sizeof(uint8_t) * UI_WIDTH * UI_HEIGHT_BUFFER * 4);

  UITab* active_tab = (ui->tabs[ui->tab].count == 1) ? ui->tabs + ui->tab : ui->tabs[ui->tab].subtabs + ui->subtab;

  render_tab(ui, active_tab);

  if (ui->dropdown) {
    const int first_panel = 1 + ui->scroll_pos / PANEL_HEIGHT;
    render_dropdown(ui, ui->dropdown, first_panel - 1);
  }
}

void blit_UI(UI* ui, uint8_t* target, int width, int height, int ld) {
  if (!ui->active)
    return;

  ui->max_x = width;
  ui->max_y = height;

  UI_clamp_position(ui);

  blur_background(ui, target, width, height, ld);
  blit_UI_internal(ui, target, width, height, ld);
}

void free_UI(UI* ui) {
  free(ui->pixels);
  free(ui->pixels_mask);
  free(ui->scratch);

  for (int i = 0; i < ui->tab_count; i++) {
    if (ui->tabs[i].count == 1) {
      free(ui->tabs[i].panels);
    }
    else {
      for (int j = 0; j < ui->tabs[i].count; j++) {
        free(ui->tabs[i].subtabs[j].panels);
      }
      free(ui->tabs[i].subtabs);
    }
  }

  free(ui->tabs);
}

#include "UI.h"

#include <float.h>
#include <immintrin.h>
#include <stdlib.h>
#include <string.h>

#include "UI_blit.h"
#include "UI_blur.h"
#include "UI_dropdown.h"
#include "UI_info.h"
#include "UI_panel.h"
#include "UI_text.h"
#include "baked.h"
#include "raytrace.h"
#include "scene.h"

#define MOUSE_LEFT_BLOCKED 0b1
#define MOUSE_DRAGGING_WINDOW 0b10
#define MOUSE_DRAGGING_SLIDER 0b100

#define MOUSE_SCROLL_SPEED 10

static size_t compute_scratch_space() {
  size_t val = blur_scratch_needed();

  return val;
}

/*
 * Requires the font of the UI to be initialized.
 */
static UIPanel* create_general_panels(UI* ui, RaytraceInstance* instance) {
  UIPanel* panels = (UIPanel*) malloc(sizeof(UIPanel) * UI_PANELS_GENERAL_COUNT);

  int i = 0;

  panels[i++] = create_tab(ui, &(ui->tab));
  panels[i++] = create_slider(ui, "Width", &(instance->settings.width), 0, 0.9f, 16.0f, 16384.0f, 0, 1);
  panels[i++] = create_slider(ui, "Height", &(instance->settings.height), 0, 0.9f, 16.0f, 16384.0f, 0, 1);
  panels[i++] = create_slider(ui, "Max Ray Depth", &(instance->settings.max_ray_depth), 0, 0.02f, 0.0f, 1024.0f, 0, 1);
  panels[i++] = create_button(ui, "Reset Renderer", instance, (void (*)(void*)) reset_raytracing, 1);
  panels[i++] = create_info(ui, "Triangle Count", &(instance->scene_gpu.triangles_length), PANEL_INFO_TYPE_INT32, PANEL_INFO_STATIC);
  panels[i++] = create_dropdown(ui, "Snapshot Resolution", &(instance->snap_resolution), 0, 2, "Window\0Render", 6);
  panels[i++] = create_check(ui, "Optix Denoiser", &(instance->use_denoiser), 0);
  panels[i++] = create_dropdown(ui, "Accumulation Mode", &(instance->accum_mode), 1, 3, "Off\0Accumulation\0Reprojection", 8);
  panels[i++] = create_info(ui, "Temporal Frames", &(instance->temporal_frames), PANEL_INFO_TYPE_INT32, PANEL_INFO_DYNAMIC);
  panels[i++] = create_info(ui, "Light Source Count", &(instance->scene_gpu.lights_length), PANEL_INFO_TYPE_INT32, PANEL_INFO_STATIC);
  panels[i++] = create_check(ui, "Lights", &(instance->lights_active), 1);
  panels[i++] =
    create_dropdown(ui, "Shading Mode", &(instance->shading_mode), 1, 6, "Default\0Albedo\0Depth\0Normal\0Trace Heatmap\0Wireframe", 12);
  panels[i++] = create_slider(ui, "Default Smoothness", &(instance->default_material.r), 1, 0.001f, 0.0f, 1.0f, 0, 0);
  panels[i++] = create_slider(ui, "Default Metallic", &(instance->default_material.g), 1, 0.001f, 0.0f, 1.0f, 0, 0);
  panels[i++] = create_slider(ui, "Default Light Intensity", &(instance->default_material.b), 1, 0.001f, 0.0f, FLT_MAX, 0, 0);
  panels[i++] = create_slider(ui, "Camera Speed", &(instance->scene_gpu.camera.wasd_speed), 0, 0.0001f, 0.0f, FLT_MAX, 0, 0);
  panels[i++] = create_slider(ui, "Mouse Sensitivity", &(instance->scene_gpu.camera.mouse_speed), 0, 0.0001f, 0.0f, FLT_MAX, 0, 0);
  panels[i++] = create_check(ui, "Smooth Camera Movement", &(instance->scene_gpu.camera.smooth_movement), 0);
  panels[i++] = create_slider(ui, "Smoothing Factor", &(instance->scene_gpu.camera.smoothing_factor), 0, 0.0001f, 0.0f, 1.0f, 0, 0);
  panels[i++] = create_button(ui, "Export Settings", instance, (void (*)(void*)) serialize_scene, 0);
  panels[i++] = create_button(ui, "Export Baked File", instance, (void (*)(void*)) serialize_baked, 0);

  return panels;
}

static UIPanel* create_camera_panels(UI* ui, RaytraceInstance* instance) {
  UIPanel* panels = (UIPanel*) malloc(sizeof(UIPanel) * UI_PANELS_CAMERA_COUNT);

  int i = 0;

  panels[i++] = create_tab(ui, &(ui->tab));
  panels[i++] = create_slider(ui, "Position X", &(instance->scene_gpu.camera.pos.x), 1, 0.001f, -FLT_MAX, FLT_MAX, 1, 0);
  panels[i++] = create_slider(ui, "Position Y", &(instance->scene_gpu.camera.pos.y), 1, 0.001f, -FLT_MAX, FLT_MAX, 1, 0);
  panels[i++] = create_slider(ui, "Position Z", &(instance->scene_gpu.camera.pos.z), 1, 0.001f, -FLT_MAX, FLT_MAX, 1, 0);
  panels[i++] = create_slider(ui, "Rotation X", &(instance->scene_gpu.camera.rotation.x), 1, 0.001f, -FLT_MAX, FLT_MAX, 1, 0);
  panels[i++] = create_slider(ui, "Rotation Y", &(instance->scene_gpu.camera.rotation.y), 1, 0.001f, -FLT_MAX, FLT_MAX, 1, 0);
  panels[i++] = create_slider(ui, "Rotation Z", &(instance->scene_gpu.camera.rotation.z), 1, 0.001f, -FLT_MAX, FLT_MAX, 1, 0);
  panels[i++] = create_dropdown(ui, "Tone Mapping", &(instance->scene_gpu.camera.tonemap), 0, 4, "None\0ACES\0Reinhard\0Uncharted 2", 7);
  panels[i++] = create_dropdown(
    ui, "Filter", &(instance->scene_gpu.camera.filter), 0, 7, "None\0Gray\0Sepia\0Gameboy\0002 Bit Gray\0CRT\0Black/White", 8);
  panels[i++] = create_slider(ui, "Field of View", &(instance->scene_gpu.camera.fov), 1, 0.001f, 0.0001f, FLT_MAX, 0, 0);
  panels[i++] = create_check(ui, "Auto Exposure", &(instance->scene_gpu.camera.auto_exposure), 0);
  panels[i++] = create_slider(ui, "Exposure", &(instance->scene_gpu.camera.exposure), 0, 0.0005f, 0.0f, FLT_MAX, 1, 0);
  panels[i++] = create_slider(ui, "Aperture Size", &(instance->scene_gpu.camera.aperture_size), 1, 0.0005f, 0.0f, FLT_MAX, 0, 0);
  panels[i++] = create_slider(ui, "Focal Length", &(instance->scene_gpu.camera.focal_length), 1, 0.001f, 0.0f, FLT_MAX, 0, 0);
  panels[i++] = create_slider(ui, "Far Clip Distance", &(instance->scene_gpu.camera.far_clip_distance), 1, 0.05f, 0.0f, FLT_MAX, 0, 0);
  panels[i++] = create_check(ui, "Bloom", &(instance->scene_gpu.camera.bloom), 0);
  panels[i++] = create_slider(ui, "Bloom Strength", &(instance->scene_gpu.camera.bloom_strength), 0, 0.0005f, 0.0f, 1.0f, 0, 0);
  panels[i++] = create_check(ui, "Dithering", &(instance->scene_gpu.camera.dithering), 0);
  panels[i++] = create_slider(ui, "Alpha Cutoff", &(instance->scene_gpu.camera.alpha_cutoff), 1, 0.0005f, 0.0f, 1.0f, 0, 0);

  return panels;
}

static UIPanel* create_sky_panels(UI* ui, RaytraceInstance* instance) {
  UIPanel* panels = (UIPanel*) malloc(sizeof(UIPanel) * UI_PANELS_SKY_COUNT);

  int i = 0;

  panels[i++] = create_tab(ui, &(ui->tab));
  panels[i++] = create_color(ui, "Sun Color", (float*) &(instance->scene_gpu.sky.sun_color));
  panels[i++] = create_slider(ui, "  Red", &(instance->scene_gpu.sky.sun_color.r), 1, 0.001f, 0.0f, 1.0f, 0, 0);
  panels[i++] = create_slider(ui, "  Green", &(instance->scene_gpu.sky.sun_color.g), 1, 0.001f, 0.0f, 1.0f, 0, 0);
  panels[i++] = create_slider(ui, "  Blue", &(instance->scene_gpu.sky.sun_color.b), 1, 0.001f, 0.0f, 1.0f, 0, 0);
  panels[i++] = create_slider(ui, "Azimuth", &(instance->scene_gpu.sky.azimuth), 1, 0.001f, -FLT_MAX, FLT_MAX, 1, 0);
  panels[i++] = create_slider(ui, "Altitude", &(instance->scene_gpu.sky.altitude), 1, 0.001f, -FLT_MAX, FLT_MAX, 1, 0);
  panels[i++] = create_slider(ui, "Density", &(instance->scene_gpu.sky.base_density), 1, 0.001f, 0.0f, FLT_MAX, 0, 0);
  panels[i++] = create_slider(ui, "Rayleigh Falloff", &(instance->scene_gpu.sky.rayleigh_falloff), 1, 0.001f, 0.0f, FLT_MAX, 0, 0);
  panels[i++] = create_slider(ui, "Mie Falloff", &(instance->scene_gpu.sky.mie_falloff), 1, 0.001f, 0.0f, FLT_MAX, 0, 0);
  panels[i++] = create_slider(ui, "Sun Intensity", &(instance->scene_gpu.sky.sun_strength), 1, 0.001f, 0.0f, FLT_MAX, 0, 0);
  panels[i++] = create_check(ui, "Fog Active", &(instance->scene_gpu.fog.active), 1);
  panels[i++] = create_slider(ui, "Fog Absorption", &(instance->scene_gpu.fog.absorption), 1, 0.001f, 0.0f, FLT_MAX, 0, 0);
  panels[i++] = create_slider(ui, "Fog Scattering", &(instance->scene_gpu.fog.scattering), 1, 0.001f, 0.0f, FLT_MAX, 0, 0);
  panels[i++] = create_slider(ui, "Fog Anisotropy", &(instance->scene_gpu.fog.anisotropy), 1, 0.001f, -0.95f, 0.95f, 0, 0);
  panels[i++] = create_slider(ui, "Fog Distance", &(instance->scene_gpu.fog.dist), 1, 0.005f, 0.0f, FLT_MAX, 0, 0);
  panels[i++] = create_slider(ui, "Fog Height", &(instance->scene_gpu.fog.height), 1, 0.005f, -FLT_MAX, FLT_MAX, 0, 0);
  panels[i++] = create_slider(ui, "Fog Height Falloff", &(instance->scene_gpu.fog.falloff), 1, 0.005f, 0.0f, FLT_MAX, 0, 0);

  return panels;
}

static UIPanel* create_ocean_panels(UI* ui, RaytraceInstance* instance) {
  UIPanel* panels = (UIPanel*) malloc(sizeof(UIPanel) * UI_PANELS_OCEAN_COUNT);

  int i = 0;

  panels[i++] = create_tab(ui, &(ui->tab));
  panels[i++] = create_check(ui, "Active", &(instance->scene_gpu.ocean.active), 1);
  panels[i++] = create_slider(ui, "Height", &(instance->scene_gpu.ocean.height), 1, 0.005f, -FLT_MAX, FLT_MAX, 0, 0);
  panels[i++] = create_slider(ui, "Amplitude", &(instance->scene_gpu.ocean.amplitude), 1, 0.005f, 0.0f, FLT_MAX, 0, 0);
  panels[i++] = create_slider(ui, "Choppyness", &(instance->scene_gpu.ocean.choppyness), 1, 0.005f, 0.0f, FLT_MAX, 0, 0);
  panels[i++] = create_slider(ui, "Frequency", &(instance->scene_gpu.ocean.frequency), 1, 0.005f, 0.0f, FLT_MAX, 0, 0);
  panels[i++] = create_color(ui, "Albedo", (float*) &(instance->scene_gpu.ocean.albedo));
  panels[i++] = create_slider(ui, "  Red", &(instance->scene_gpu.ocean.albedo.r), 1, 0.001f, 0.0f, 1.0f, 0, 0);
  panels[i++] = create_slider(ui, "  Green", &(instance->scene_gpu.ocean.albedo.g), 1, 0.001f, 0.0f, 1.0f, 0, 0);
  panels[i++] = create_slider(ui, "  Blue", &(instance->scene_gpu.ocean.albedo.b), 1, 0.001f, 0.0f, 1.0f, 0, 0);
  panels[i++] = create_slider(ui, "Refractive Index", &(instance->scene_gpu.ocean.refractive_index), 1, 0.001f, 1.0f, FLT_MAX, 0, 0);
  panels[i++] = create_check(ui, "Emissive", &(instance->scene_gpu.ocean.emissive), 1);
  panels[i++] = create_slider(ui, "Alpha", &(instance->scene_gpu.ocean.albedo.a), 1, 0.001f, 0.0f, 1.0f, 0, 0);
  panels[i++] = create_check(ui, "Animated", &(instance->scene_gpu.ocean.update), 1);
  panels[i++] = create_slider(ui, "Speed", &(instance->scene_gpu.ocean.speed), 1, 0.005f, 0.0f, FLT_MAX, 0, 0);

  return panels;
}

static UIPanel* create_toy_panels(UI* ui, RaytraceInstance* instance) {
  UIPanel* panels = (UIPanel*) malloc(sizeof(UIPanel) * UI_PANELS_TOY_COUNT);

  int i = 0;

  panels[i++] = create_tab(ui, &(ui->tab));
  panels[i++] = create_check(ui, "Active", &(instance->scene_gpu.toy.active), 1);
  panels[i++] = create_dropdown(ui, "Shape", &(instance->scene_gpu.toy.shape), 1, 1, "Sphere", 2);
  panels[i++] = create_button(ui, "Center at Camera", instance, (void (*)(void*)) center_toy_at_camera, 1);
  panels[i++] = create_slider(ui, "Position X", &(instance->scene_gpu.toy.position.x), 1, 0.005f, -FLT_MAX, FLT_MAX, 1, 0);
  panels[i++] = create_slider(ui, "Position Y", &(instance->scene_gpu.toy.position.y), 1, 0.005f, -FLT_MAX, FLT_MAX, 1, 0);
  panels[i++] = create_slider(ui, "Position Z", &(instance->scene_gpu.toy.position.z), 1, 0.005f, -FLT_MAX, FLT_MAX, 1, 0);
  panels[i++] = create_slider(ui, "Rotation X", &(instance->scene_gpu.toy.rotation.x), 1, 0.005f, -FLT_MAX, FLT_MAX, 0, 0);
  panels[i++] = create_slider(ui, "Rotation Y", &(instance->scene_gpu.toy.rotation.y), 1, 0.005f, -FLT_MAX, FLT_MAX, 0, 0);
  panels[i++] = create_slider(ui, "Rotation Z", &(instance->scene_gpu.toy.rotation.z), 1, 0.005f, -FLT_MAX, FLT_MAX, 0, 0);
  panels[i++] = create_slider(ui, "Scale", &(instance->scene_gpu.toy.scale), 1, 0.005f, 0.0f, FLT_MAX, 0, 0);
  panels[i++] = create_color(ui, "Albedo", (float*) &(instance->scene_gpu.toy.albedo));
  panels[i++] = create_slider(ui, "  Red", &(instance->scene_gpu.toy.albedo.r), 1, 0.001f, 0.0f, 1.0f, 0, 0);
  panels[i++] = create_slider(ui, "  Green", &(instance->scene_gpu.toy.albedo.g), 1, 0.001f, 0.0f, 1.0f, 0, 0);
  panels[i++] = create_slider(ui, "  Blue", &(instance->scene_gpu.toy.albedo.b), 1, 0.001f, 0.0f, 1.0f, 0, 0);
  panels[i++] = create_slider(ui, "  Alpha", &(instance->scene_gpu.toy.albedo.a), 1, 0.001f, 0.0f, 1.0f, 0, 0);
  panels[i++] = create_slider(ui, "Smoothness", &(instance->scene_gpu.toy.material.r), 1, 0.001f, 0.0f, 1.0f, 0, 0);
  panels[i++] = create_slider(ui, "Metallic", &(instance->scene_gpu.toy.material.g), 1, 0.001f, 0.0f, 1.0f, 0, 0);
  panels[i++] = create_slider(ui, "Refractive Index", &(instance->scene_gpu.toy.refractive_index), 1, 0.001f, 1.0f, FLT_MAX, 0, 0);
  panels[i++] = create_check(ui, "Emissive", &(instance->scene_gpu.toy.emissive), 1);
  panels[i++] = create_color(ui, "Emission", (float*) &(instance->scene_gpu.toy.emission));
  panels[i++] = create_slider(ui, "  Red", &(instance->scene_gpu.toy.emission.r), 1, 0.001f, 0.0f, 1.0f, 0, 0);
  panels[i++] = create_slider(ui, "  Green", &(instance->scene_gpu.toy.emission.g), 1, 0.001f, 0.0f, 1.0f, 0, 0);
  panels[i++] = create_slider(ui, "  Blue", &(instance->scene_gpu.toy.emission.b), 1, 0.001f, 0.0f, 1.0f, 0, 0);
  panels[i++] = create_slider(ui, "Light Intensity", &(instance->scene_gpu.toy.material.b), 1, 0.001f, 0.0f, FLT_MAX, 0, 0);

  return panels;
}

UI init_UI(RaytraceInstance* instance, RealtimeInstance* realtime) {
  UI ui;
  ui.active = 0;
  ui.tab    = UI_PANELS_GENERAL_TAB;

  ui.x     = 100;
  ui.y     = 100;
  ui.max_x = realtime->width - UI_WIDTH;
  ui.max_y = realtime->height - UI_HEIGHT - UI_BORDER_SIZE;

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

  ui.general_panels = create_general_panels(&ui, instance);
  ui.camera_panels  = create_camera_panels(&ui, instance);
  ui.sky_panels     = create_sky_panels(&ui, instance);
  ui.ocean_panels   = create_ocean_panels(&ui, instance);
  ui.toy_panels     = create_toy_panels(&ui, instance);

  SDL_SetRelativeMouseMode(!ui.active);

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

  int max_scroll = 0;

  switch (ui->tab) {
    case UI_PANELS_GENERAL_TAB:
      max_scroll = UI_PANELS_GENERAL_COUNT;
      break;
    case UI_PANELS_CAMERA_TAB:
      max_scroll = UI_PANELS_CAMERA_COUNT;
      break;
    case UI_PANELS_SKY_TAB:
      max_scroll = UI_PANELS_SKY_COUNT;
      break;
    case UI_PANELS_OCEAN_TAB:
      max_scroll = UI_PANELS_OCEAN_COUNT;
      break;
    case UI_PANELS_TOY_TAB:
      max_scroll = UI_PANELS_TOY_COUNT;
      break;
  }

  max_scroll -= UI_HEIGHT_IN_PANELS;
  max_scroll *= PANEL_HEIGHT;

  clamp(max_scroll, 0, INT_MAX);
  clamp(ui->scroll_pos, 0, max_scroll);

  if (ui->mouse_flags & MOUSE_DRAGGING_WINDOW) {
    ui->x += d_x;
    ui->y += d_y;

    clamp(ui->x, 0, ui->max_x);
    clamp(ui->y, 0, ui->max_y);

    ui->mouse_flags |= MOUSE_LEFT_BLOCKED;
    ui->mouse_flags ^= MOUSE_DRAGGING_WINDOW;
  }

  switch (ui->tab) {
    case UI_PANELS_GENERAL_TAB: {
      for (int i = 0; i < UI_PANELS_GENERAL_COUNT; i++) {
        ui->general_panels[i].hover = 0;
      }
    } break;
    case UI_PANELS_CAMERA_TAB: {
      for (int i = 0; i < UI_PANELS_CAMERA_COUNT; i++) {
        ui->camera_panels[i].hover = 0;
      }
    } break;
    case UI_PANELS_SKY_TAB: {
      for (int i = 0; i < UI_PANELS_SKY_COUNT; i++) {
        ui->sky_panels[i].hover = 0;
      }
    } break;
    case UI_PANELS_OCEAN_TAB: {
      for (int i = 0; i < UI_PANELS_OCEAN_COUNT; i++) {
        ui->ocean_panels[i].hover = 0;
      }
    } break;
    case UI_PANELS_TOY_TAB: {
      for (int i = 0; i < UI_PANELS_TOY_COUNT; i++) {
        ui->toy_panels[i].hover = 0;
      }
    } break;
  }

  x -= ui->x;
  y -= ui->y;

  UIPanel* panel = (UIPanel*) 0;

  if (ui->mouse_flags & MOUSE_DRAGGING_SLIDER) {
    panel = ui->last_panel;
  }
  else if (ui->dropdown && get_intersection_dropdown(ui, ui->dropdown, x, ui->scroll_pos + y)) {
    if (ui->mouse_flags & MOUSE_LEFT_BLOCKED) {
      state &= ~SDL_BUTTON_LMASK;
    }

    handle_mouse_UIPanel(ui, ui->dropdown, state, x, (ui->scroll_pos + y));

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

      UIPanel* panels_tab = ui->general_panels;
      int panel_count     = UI_PANELS_GENERAL_COUNT;

      switch (ui->tab) {
        case UI_PANELS_GENERAL_TAB: {
          panels_tab  = ui->general_panels;
          panel_count = UI_PANELS_GENERAL_COUNT;
        } break;
        case UI_PANELS_CAMERA_TAB: {
          panels_tab  = ui->camera_panels;
          panel_count = UI_PANELS_CAMERA_COUNT;
        } break;
        case UI_PANELS_SKY_TAB: {
          panels_tab  = ui->sky_panels;
          panel_count = UI_PANELS_SKY_COUNT;
        } break;
        case UI_PANELS_OCEAN_TAB: {
          panels_tab  = ui->ocean_panels;
          panel_count = UI_PANELS_OCEAN_COUNT;
        } break;
        case UI_PANELS_TOY_TAB: {
          panels_tab  = ui->toy_panels;
          panel_count = UI_PANELS_TOY_COUNT;
        } break;
      }

      if (ui->panel_hover < panel_count)
        panel = panels_tab + ui->panel_hover;
    }
  }
  else {
    ui->panel_hover  = -1;
    ui->border_hover = 0;
  }

  if (SDL_BUTTON_LMASK & state) {
    ui->mouse_flags |= MOUSE_LEFT_BLOCKED;
  }

  if (panel) {
    handle_mouse_UIPanel(ui, panel, state, x, (ui->scroll_pos + y) % PANEL_HEIGHT);
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

void render_UI(UI* ui) {
  if (!ui->active)
    return;

  memset(ui->pixels, 0, sizeof(uint8_t) * UI_WIDTH * UI_HEIGHT_BUFFER * 4);
  memset(ui->pixels_mask, 0, sizeof(uint8_t) * UI_WIDTH * UI_HEIGHT_BUFFER * 4);

  const int first_panel = 1 + ui->scroll_pos / PANEL_HEIGHT;
  const int last_panel  = first_panel + UI_HEIGHT_IN_PANELS;
  int y                 = PANEL_HEIGHT;

  switch (ui->tab) {
    case UI_PANELS_GENERAL_TAB: {
      render_UIPanel(ui, ui->general_panels, 0);
      for (int i = first_panel; i < min(last_panel, UI_PANELS_GENERAL_COUNT); i++) {
        render_UIPanel(ui, ui->general_panels + i, y);
        y += PANEL_HEIGHT;
      }
    } break;
    case UI_PANELS_CAMERA_TAB: {
      render_UIPanel(ui, ui->camera_panels, 0);
      for (int i = first_panel; i < min(last_panel, UI_PANELS_CAMERA_COUNT); i++) {
        render_UIPanel(ui, ui->camera_panels + i, y);
        y += PANEL_HEIGHT;
      }
    } break;
    case UI_PANELS_SKY_TAB: {
      render_UIPanel(ui, ui->sky_panels, 0);
      for (int i = first_panel; i < min(last_panel, UI_PANELS_SKY_COUNT); i++) {
        render_UIPanel(ui, ui->sky_panels + i, y);
        y += PANEL_HEIGHT;
      }
    } break;
    case UI_PANELS_OCEAN_TAB: {
      render_UIPanel(ui, ui->ocean_panels, 0);
      for (int i = first_panel; i < min(last_panel, UI_PANELS_OCEAN_COUNT); i++) {
        render_UIPanel(ui, ui->ocean_panels + i, y);
        y += PANEL_HEIGHT;
      }
    } break;
    case UI_PANELS_TOY_TAB: {
      render_UIPanel(ui, ui->toy_panels, 0);
      for (int i = first_panel; i < min(last_panel, UI_PANELS_TOY_COUNT); i++) {
        render_UIPanel(ui, ui->toy_panels + i, y);
        y += PANEL_HEIGHT;
      }
    } break;
  }

  if (ui->dropdown) {
    render_dropdown(ui, ui->dropdown, first_panel - 1);
  }
}

void blit_UI(UI* ui, uint8_t* target, int width, int height) {
  if (!ui->active)
    return;

  blur_background(ui, target, width, height);
  blit_UI_internal(ui, target, width, height);
}

void free_UI(UI* ui) {
  free(ui->pixels);
  free(ui->pixels_mask);
  free(ui->scratch);

  for (int i = 0; i < UI_PANELS_GENERAL_COUNT; i++) {
    free_UIPanel(ui->general_panels + i);
  }
  for (int i = 0; i < UI_PANELS_CAMERA_COUNT; i++) {
    free_UIPanel(ui->camera_panels + i);
  }
  for (int i = 0; i < UI_PANELS_SKY_COUNT; i++) {
    free_UIPanel(ui->sky_panels + i);
  }
  for (int i = 0; i < UI_PANELS_OCEAN_COUNT; i++) {
    free_UIPanel(ui->ocean_panels + i);
  }
  for (int i = 0; i < UI_PANELS_TOY_COUNT; i++) {
    free_UIPanel(ui->toy_panels + i);
  }

  free(ui->general_panels);
  free(ui->camera_panels);
  free(ui->sky_panels);
  free(ui->ocean_panels);
  free(ui->toy_panels);
}

#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <immintrin.h>
#include "UI.h"
#include "UI_blur.h"
#include "UI_blit.h"
#include "UI_text.h"
#include "UI_panel.h"
#include "UI_info.h"

#define MOUSE_LEFT_BLOCKED 0b1
#define MOUSE_DRAGGING_WINDOW 0b10
#define MOUSE_DRAGGING_SLIDER 0b100

static size_t compute_scratch_space() {
  size_t val = blur_scratch_needed();

  return val;
}

/*
 * Requires the font of the UI to be initialized.
 */
static UIPanel* create_general_panels(UI* ui, RaytraceInstance* instance) {
  UIPanel* panels = (UIPanel*) malloc(sizeof(UIPanel) * UI_PANELS_GENERAL_COUNT);

  panels[0] = create_tab(ui, 0, &(ui->tab));
  panels[1] =
    create_info(ui, 1, "Width", &(instance->width), PANEL_INFO_TYPE_INT32, PANEL_INFO_STATIC);
  panels[2] =
    create_info(ui, 2, "Height", &(instance->height), PANEL_INFO_TYPE_INT32, PANEL_INFO_STATIC);
  panels[3] = create_info(
    ui, 3, "Triangle Count", &(instance->scene_gpu.triangles_length), PANEL_INFO_TYPE_INT32,
    PANEL_INFO_STATIC);
  panels[4] = create_check(ui, 4, "Optix Denoiser", &(instance->use_denoiser), 0);
  panels[5] = create_info(
    ui, 5, "Temporal Frames", &(instance->temporal_frames), PANEL_INFO_TYPE_INT32,
    PANEL_INFO_DYNAMIC);
  panels[6] = create_info(
    ui, 6, "Light Source Count", &(instance->scene_gpu.lights_length), PANEL_INFO_TYPE_INT32,
    PANEL_INFO_STATIC);
  panels[7] = create_check(ui, 7, "Lights", &(instance->lights_active), 1);
  panels[8] = create_slider(
    ui, 8, "Default Smoothness", &(instance->default_material.r), 1, 0.001f, 0.0f, 1.0f);
  panels[9] = create_slider(
    ui, 9, "Default Metallic", &(instance->default_material.g), 1, 0.001f, 0.0f, 1.0f);
  panels[10] = create_slider(
    ui, 10, "Default Light Intensity", &(instance->default_material.b), 1, 0.001f, 0.0f, FLT_MAX);

  return panels;
}

static UIPanel* create_camera_panels(UI* ui, RaytraceInstance* instance) {
  UIPanel* panels = (UIPanel*) malloc(sizeof(UIPanel) * UI_PANELS_CAMERA_COUNT);

  panels[0] = create_tab(ui, 0, &(ui->tab));
  panels[1] = create_info(
    ui, 1, "Position X", &(instance->scene_gpu.camera.pos.x), PANEL_INFO_TYPE_FP32,
    PANEL_INFO_DYNAMIC);
  panels[2] = create_info(
    ui, 2, "Position Y", &(instance->scene_gpu.camera.pos.y), PANEL_INFO_TYPE_FP32,
    PANEL_INFO_DYNAMIC);
  panels[3] = create_info(
    ui, 3, "Position Z", &(instance->scene_gpu.camera.pos.z), PANEL_INFO_TYPE_FP32,
    PANEL_INFO_DYNAMIC);
  panels[4] = create_info(
    ui, 4, "Rotation X", &(instance->scene_gpu.camera.rotation.x), PANEL_INFO_TYPE_FP32,
    PANEL_INFO_DYNAMIC);
  panels[5] = create_info(
    ui, 5, "Rotation Y", &(instance->scene_gpu.camera.rotation.y), PANEL_INFO_TYPE_FP32,
    PANEL_INFO_DYNAMIC);
  panels[6] = create_info(
    ui, 6, "Rotation Z", &(instance->scene_gpu.camera.rotation.z), PANEL_INFO_TYPE_FP32,
    PANEL_INFO_DYNAMIC);
  panels[7] = create_slider(
    ui, 7, "Field of View", &(instance->scene_gpu.camera.fov), 1, 0.001f, 0.0001f, FLT_MAX);
  panels[8] = create_check(ui, 8, "Auto Exposure", &(instance->scene_gpu.camera.auto_exposure), 0);
  panels[9] = create_info(
    ui, 9, "Exposure", &(instance->scene_gpu.camera.exposure), PANEL_INFO_TYPE_FP32,
    PANEL_INFO_DYNAMIC);
  panels[10] = create_slider(
    ui, 10, "Aperture Size", &(instance->scene_gpu.camera.aperture_size), 1, 0.0005f, 0.0f,
    FLT_MAX);
  panels[11] = create_slider(
    ui, 11, "Focal Length", &(instance->scene_gpu.camera.focal_length), 1, 0.001f, 0.0f, FLT_MAX);
  panels[12] = create_slider(
    ui, 12, "Far Clip Distance", &(instance->scene_gpu.camera.far_clip_distance), 1, 0.05f, 0.0f,
    FLT_MAX);
  panels[13] = create_check(ui, 13, "Bloom", &(instance->use_bloom), 0);
  panels[14] = create_slider(
    ui, 14, "Alpha Cutoff", &(instance->scene_gpu.camera.alpha_cutoff), 1, 0.0005f, 0.0f, 1.0f);

  return panels;
}

static UIPanel* create_sky_panels(UI* ui, RaytraceInstance* instance) {
  UIPanel* panels = (UIPanel*) malloc(sizeof(UIPanel) * UI_PANELS_SKY_COUNT);

  panels[0] = create_tab(ui, 0, &(ui->tab));
  panels[1] = create_info(
    ui, 1, "Azimuth", &(instance->scene_gpu.sky.azimuth), PANEL_INFO_TYPE_FP32, PANEL_INFO_DYNAMIC);
  panels[2] = create_info(
    ui, 2, "Altitude", &(instance->scene_gpu.sky.altitude), PANEL_INFO_TYPE_FP32,
    PANEL_INFO_DYNAMIC);
  panels[3] = create_slider(
    ui, 3, "Density", &(instance->scene_gpu.sky.base_density), 1, 0.001f, 0.0f, FLT_MAX);
  panels[4] = create_slider(
    ui, 4, "Rayleigh Falloff", &(instance->scene_gpu.sky.rayleigh_falloff), 1, 0.001f, 0.0f,
    FLT_MAX);
  panels[5] = create_slider(
    ui, 5, "Mie Falloff", &(instance->scene_gpu.sky.mie_falloff), 1, 0.001f, 0.0f, FLT_MAX);
  panels[6] = create_slider(
    ui, 6, "Sun Intensity", &(instance->scene_gpu.sky.sun_strength), 1, 0.001f, 0.0f, FLT_MAX);

  return panels;
}

static UIPanel* create_ocean_panels(UI* ui, RaytraceInstance* instance) {
  UIPanel* panels = (UIPanel*) malloc(sizeof(UIPanel) * UI_PANELS_OCEAN_COUNT);

  panels[0] = create_tab(ui, 0, &(ui->tab));
  panels[1] = create_check(ui, 1, "Active", &(instance->scene_gpu.ocean.active), 1);
  panels[2] = create_slider(
    ui, 2, "Height", &(instance->scene_gpu.ocean.height), 1, 0.005f, -FLT_MAX, FLT_MAX);
  panels[3] = create_slider(
    ui, 3, "Amplitude", &(instance->scene_gpu.ocean.amplitude), 1, 0.005f, 0.0f, FLT_MAX);
  panels[4] = create_slider(
    ui, 4, "Choppyness", &(instance->scene_gpu.ocean.choppyness), 1, 0.005f, 0.0f, FLT_MAX);
  panels[5] = create_slider(
    ui, 5, "Frequency", &(instance->scene_gpu.ocean.frequency), 1, 0.005f, 0.0f, FLT_MAX);
  panels[6] = create_color(ui, 6, "Albedo", (float*) &(instance->scene_gpu.ocean.albedo));
  panels[7] =
    create_slider(ui, 7, "  Red", &(instance->scene_gpu.ocean.albedo.r), 1, 0.001f, 0.0f, 1.0f);
  panels[8] =
    create_slider(ui, 8, "  Green", &(instance->scene_gpu.ocean.albedo.g), 1, 0.001f, 0.0f, 1.0f);
  panels[9] =
    create_slider(ui, 9, "  Blue", &(instance->scene_gpu.ocean.albedo.b), 1, 0.001f, 0.0f, 1.0f);
  panels[10] = create_check(ui, 10, "Emissive", &(instance->scene_gpu.ocean.emissive), 1);
  panels[11] =
    create_slider(ui, 11, "Alpha", &(instance->scene_gpu.ocean.albedo.a), 1, 0.001f, 0.0f, 1.0f);
  panels[12] = create_check(ui, 12, "Animated", &(instance->scene_gpu.ocean.update), 1);
  panels[13] =
    create_slider(ui, 13, "Speed", &(instance->scene_gpu.ocean.speed), 1, 0.005f, 0.0f, FLT_MAX);

  return panels;
}

static UIPanel* create_toy_panels(UI* ui, RaytraceInstance* instance) {
  UIPanel* panels = (UIPanel*) malloc(sizeof(UIPanel) * UI_PANELS_TOY_COUNT);

  panels[0] = create_tab(ui, 0, &(ui->tab));

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

  ui.pixels      = (uint8_t*) malloc(sizeof(uint8_t) * UI_WIDTH * UI_HEIGHT * 3);
  ui.pixels_mask = (uint8_t*) malloc(sizeof(uint8_t) * UI_WIDTH * UI_HEIGHT * 3);

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
      ui->panel_hover  = y / PANEL_HEIGHT;
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
    handle_mouse_UIPanel(ui, panel, state, x, y % PANEL_HEIGHT);
    if (panel->type == PANEL_SLIDER) {
      ui->mouse_flags &= ~MOUSE_LEFT_BLOCKED;

      if (SDL_BUTTON_LMASK & state) {
        ui->mouse_flags |= MOUSE_DRAGGING_SLIDER;
      }
    }

    ui->last_panel = panel;
  }
}

void render_UI(UI* ui) {
  if (!ui->active)
    return;

  memset(ui->pixels, 0, sizeof(uint8_t) * UI_WIDTH * UI_HEIGHT * 3);
  memset(ui->pixels_mask, 0, sizeof(uint8_t) * UI_WIDTH * UI_HEIGHT * 3);

  switch (ui->tab) {
  case UI_PANELS_GENERAL_TAB: {
    for (int i = 0; i < UI_PANELS_GENERAL_COUNT; i++) {
      render_UIPanel(ui, ui->general_panels + i);
    }
  } break;
  case UI_PANELS_CAMERA_TAB: {
    for (int i = 0; i < UI_PANELS_CAMERA_COUNT; i++) {
      render_UIPanel(ui, ui->camera_panels + i);
    }
  } break;
  case UI_PANELS_SKY_TAB: {
    for (int i = 0; i < UI_PANELS_SKY_COUNT; i++) {
      render_UIPanel(ui, ui->sky_panels + i);
    }
  } break;
  case UI_PANELS_OCEAN_TAB: {
    for (int i = 0; i < UI_PANELS_OCEAN_COUNT; i++) {
      render_UIPanel(ui, ui->ocean_panels + i);
    }
  } break;
  case UI_PANELS_TOY_TAB: {
    for (int i = 0; i < UI_PANELS_TOY_COUNT; i++) {
      render_UIPanel(ui, ui->toy_panels + i);
    }
  } break;
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

  free(ui->general_panels);
}
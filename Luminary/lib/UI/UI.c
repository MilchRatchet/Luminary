#include <stdlib.h>
#include <string.h>
#include <immintrin.h>
#include "UI.h"
#include "UI_blur.h"
#include "UI_blit.h"
#include "UI_text.h"
#include "UI_panel.h"
#include "UI_info.h"

#define MOUSE_LEFT_BLOCKED 0b1
#define MOUSE_DRAGGING_WINDOW 0b10

static size_t compute_scratch_space() {
  size_t val = blur_scratch_needed();

  return val;
}

/*
 * Requires the font of the UI to be initialized.
 */
static UIPanel* create_general_panels(UI* ui, RaytraceInstance* instance) {
  UIPanel* panels = (UIPanel*) malloc(sizeof(UIPanel) * UI_PANELS_GENERAL_COUNT);

  panels[0] = create_tab(ui, 0, &(ui->tab), UI_PANELS_GENERAL_TAB);
  panels[1] = create_check(ui, 1, "Optix Denoiser", &(instance->use_denoiser), 0);
  panels[2] = create_check(ui, 2, "Auto Exposure", &(instance->scene_gpu.camera.auto_exposure), 0);
  panels[3] = create_info(
    ui, 3, "Exposure", &(instance->scene_gpu.camera.exposure), PANEL_INFO_TYPE_FP32,
    PANEL_INFO_DYNAMIC);
  panels[4] = create_info(
    ui, 4, "Triangle Count", &(instance->scene_gpu.triangles_length), PANEL_INFO_TYPE_INT32,
    PANEL_INFO_STATIC);
  panels[5] = create_info(
    ui, 5, "Temporal Frames", &(instance->temporal_frames), PANEL_INFO_TYPE_INT32,
    PANEL_INFO_DYNAMIC);
  panels[6] = create_check(ui, 6, "Lights", &(instance->lights_active), 1);
  panels[7] = create_check(ui, 7, "Bloom", &(instance->use_bloom), 0);

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

  ui.pixels      = (uint8_t*) malloc(sizeof(uint8_t) * UI_WIDTH * UI_HEIGHT * 3);
  ui.pixels_mask = (uint8_t*) malloc(sizeof(uint8_t) * UI_WIDTH * UI_HEIGHT * 3);

  ui.temporal_frames = &(instance->temporal_frames);

  size_t scratch_size = compute_scratch_space();
  ui.scratch          = malloc(scratch_size);
  init_text(&ui);

  ui.general_panels = create_general_panels(&ui, instance);

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
  }

  if (ui->mouse_flags & MOUSE_DRAGGING_WINDOW) {
    ui->x += d_x;
    ui->y += d_y;

    clamp(ui->x, 0, ui->max_x);
    clamp(ui->y, 0, ui->max_y);

    ui->mouse_flags |= MOUSE_LEFT_BLOCKED;
    ui->mouse_flags ^= MOUSE_DRAGGING_WINDOW;
  }

  x -= ui->x;
  y -= ui->y;

  if (x > 0 && x < UI_WIDTH && y > 0 && y < UI_HEIGHT + UI_BORDER_SIZE) {
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

      if (ui->panel_hover < UI_PANELS_GENERAL_COUNT)
        handle_mouse_UIPanel(ui, ui->general_panels + ui->panel_hover, state, x, y % PANEL_HEIGHT);
    }
  }
  else {
    ui->panel_hover  = -1;
    ui->border_hover = 0;
  }

  if (SDL_BUTTON_LMASK & state) {
    ui->mouse_flags |= MOUSE_LEFT_BLOCKED;
  }
}

void render_UI(UI* ui) {
  if (!ui->active)
    return;

  memset(ui->pixels, 0, sizeof(uint8_t) * UI_WIDTH * UI_HEIGHT * 3);
  memset(ui->pixels_mask, 0, sizeof(uint8_t) * UI_WIDTH * UI_HEIGHT * 3);

  switch (ui->tab) {
  case UI_PANELS_GENERAL_TAB:
    for (int i = 0; i < UI_PANELS_GENERAL_COUNT; i++) {
      render_UIPanel(ui, ui->general_panels + i);
    }
    break;
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

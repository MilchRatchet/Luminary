#include <stdlib.h>
#include <string.h>
#include <immintrin.h>
#include "UI.h"
#include "UI_blur.h"
#include "UI_blit.h"
#include "UI_text.h"
#include "UI_panel.h"
#include "UI_info.h"

static size_t compute_scratch_space() {
  size_t val = blur_scratch_needed();

  return val;
}

/*
 * Requires the font of the UI to be initialized.
 */
static UIPanel* create_general_panels(UI* ui, RaytraceInstance* instance) {
  UIPanel* panels = (UIPanel*) malloc(sizeof(UIPanel) * UI_PANELS_GENERAL_COUNT);

  panels[0] = create_slider(ui, 0, "Far Clip Distance", 0);
  panels[1] = create_check(ui, 1, "Optix Denoiser", &(instance->use_denoiser));
  panels[2] = create_check(ui, 2, "Auto Exposure", &(instance->scene_gpu.camera.auto_exposure));
  panels[3] = create_info(
    ui, 3, "Exposure", &(instance->scene_gpu.camera.exposure), PANEL_INFO_TYPE_FP32,
    PANEL_INFO_DYNAMIC);
  panels[4] = create_info(
    ui, 4, "Triangle Count", &(instance->scene_gpu.triangles_length), PANEL_INFO_TYPE_INT32,
    PANEL_INFO_STATIC);

  return panels;
}

UI init_UI(RaytraceInstance* instance) {
  UI ui;
  ui.active = 0;

  ui.x          = 100;
  ui.y          = 100;
  ui.scroll_pos = 0;

  ui.panel_hover  = -1;
  ui.border_hover = 0;

  ui.pixels      = (uint8_t*) malloc(sizeof(uint8_t) * UI_WIDTH * UI_HEIGHT * 3);
  ui.pixels_mask = (uint8_t*) malloc(sizeof(uint8_t) * UI_WIDTH * UI_HEIGHT * 3);

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
  int x, y;

  uint32_t state = SDL_GetMouseState(&x, &y);

  x -= ui->x;
  y -= ui->y;

  if (x > 0 && x < UI_WIDTH && y > 0 && y < UI_HEIGHT + UI_BORDER_SIZE) {
    if (y < UI_BORDER_SIZE) {
      ui->panel_hover  = -1;
      ui->border_hover = 1;
    }
    else {
      y -= UI_BORDER_SIZE;
      ui->panel_hover  = y / PANEL_HEIGHT;
      ui->border_hover = 0;
    }
  }
  else {
    ui->panel_hover  = -1;
    ui->border_hover = 0;
  }
}

void render_UI(UI* ui) {
  if (!ui->active)
    return;

  memset(ui->pixels, 0, sizeof(uint8_t) * UI_WIDTH * UI_HEIGHT * 3);
  memset(ui->pixels_mask, 0, sizeof(uint8_t) * UI_WIDTH * UI_HEIGHT * 3);

  for (int i = 0; i < UI_PANELS_GENERAL_COUNT; i++) {
    render_UIPanel(ui, ui->general_panels + i);
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
